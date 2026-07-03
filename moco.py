"""
MoCo-v2 wrapper + projection head for protein contrastive learning
(plan_implementation_3 §2 & §3).

  * ProjectionHead (§3): MLP dim->dim->dim (SiLU) + L2-norm. The contrastive loss
    operates on g(h); the EVAL embedding is normalize(h) with g discarded — this is
    the key anti-(dimensional-)collapse piece (2110.09348).
  * MoCo (§2): an online encoder f_q (trained) and a momentum encoder f_k (EMA,
    no grad), plus a FIFO queue of negative keys. Decouples #negatives from the
    (small, MPS-bound) batch size and removes hard-neg mining (which caused the
    run-1 false-negatives + collapse). No ShuffleBN needed — the encoders use
    LayerNorm / GVP-norm, not BatchNorm.

The encoder is any Deltafold model exposing ``forward(features, return_repr=True)``
(AsymmetricTopoNet / EquivariantTopoNet). MoCo runs view1 through f_q and view2
through f_k (no_grad); the collate must keep the two views separate.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP (dim->dim->dim, SiLU) + L2 normalization. Discarded at eval."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, h):
        return F.normalize(self.net(h), p=2, dim=-1)


class MoCo(nn.Module):
    """Momentum-contrast wrapper around a Deltafold encoder.

    Args
    ----
    encoder : the trained online encoder f_q (an nn.Module with
              ``forward(features, return_repr=True) -> (B, dim)``).
    dim     : representation width (encoder repr dim, e.g. scalar_dim).
    K       : negative-queue length (plan default 8192; lower if OOM).
    m       : momentum for the EMA key encoder (0.99..0.999).
    tau     : InfoNCE temperature (plan default 0.2; 0.1 favoured collapse in run 1).
    """

    def __init__(self, encoder, dim=128, K=8192, m=0.99, tau=0.2, proj_hidden=None):
        super().__init__()
        self.K, self.m, self.tau, self.dim = K, m, tau, dim

        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.proj_q = ProjectionHead(dim, proj_hidden)
        self.proj_k = copy.deepcopy(self.proj_q)
        for p in self.encoder_k.parameters():
            p.requires_grad_(False)
        for p in self.proj_k.parameters():
            p.requires_grad_(False)

        # Negative queue (dim x K), L2-normalized columns; FIFO via a pointer.
        # Initialized with random keys (standard MoCo) so there are K negatives
        # from step 0 — they get overwritten by real keys as training proceeds.
        self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    # -- EMA + queue bookkeeping ------------------------------------------------
    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)
        for pq, pk in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        b = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + b <= self.K:
            self.queue[:, ptr:ptr + b] = keys.T
        else:                                   # wrap around
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :b - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + b) % self.K

    # -- forward / loss ---------------------------------------------------------
    def forward(self, feats_q, feats_k, real_B=None, update=True):
        """feats_q / feats_k : two collated view batches (same B proteins, same
        order). ``real_B`` slices off bucket-padding dummy rows. ``update=False``
        (validation) skips the EMA + enqueue so val data never mutates the key
        encoder or pollutes the negative queue. Returns the loss."""
        q = self.proj_q(self._repr(self.encoder_q, feats_q))        # (B, dim), grad
        if real_B is not None:
            q = q[:real_B]
        with torch.no_grad():
            self.encoder_k.eval(); self.proj_k.eval()               # deterministic keys
            if update:
                self._momentum_update()
            k = self.proj_k(self._repr(self.encoder_k, feats_k))    # (B, dim), no grad
            if real_B is not None:
                k = k[:real_B]
            k = k.detach()

        # Clone the queue (a no-grad buffer) so autograd saves a stable copy for
        # backward — the enqueue below modifies self.queue in-place.
        l_pos = (q * k).sum(dim=-1, keepdim=True)                   # (B, 1)
        l_neg = q @ self.queue.clone()                             # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)  # pos = col 0
        loss = F.cross_entropy(logits, labels)

        if update:
            self._dequeue_and_enqueue(k)
        return loss

    @staticmethod
    def _repr(encoder, feats):
        z = encoder(feats, return_repr=True)
        return z.unsqueeze(0) if z.dim() == 1 else z

    @torch.no_grad()
    def embed(self, feats, real_B=None):
        """Eval embedding = normalize(h) from the online encoder (head discarded)."""
        z = self._repr(self.encoder_q, feats)
        if real_B is not None:
            z = z[:real_B]
        return F.normalize(z, p=2, dim=-1)


# --- smoke test (plan_implementation_3 §Tests 2 & 3) -------------------------
if __name__ == "__main__":
    import glob
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import custom_collate
    from contrastive_data import pad_to_buckets
    from asymmetric_topotein import AsymmetricTopoNet
    from substructure import SubstructureViews

    pts = sorted(glob.glob('data/hoan_processed/*.pt'))[:8]
    raw = [torch.load(p, map_location='cpu', weights_only=False) for p in pts]
    sampler = SubstructureViews(f_range=(0.4, 0.6), seed=1)
    v1s, v2s = [], []
    for d in raw:
        a, b = sampler(d)
        v1s.append(a); v2s.append(b)

    def collate(views):
        f = custom_collate([(v, str(i)) for i, v in enumerate(views)])
        f, _ = pad_to_buckets(f)
        return f

    enc = AsymmetricTopoNet(scalar_dim=128, dist_encoding='rbf', rbf_dim=16,
                            use_positional_encoding=False)
    moco = MoCo(enc, dim=128, K=512, m=0.99, tau=0.2)

    # [2] eval embedding uses h (pre-head); the head must NOT leak into h.
    fb = collate(v1s)
    h = enc(fb, return_repr=True)
    emb = moco.embed(fb)
    leak = (moco.proj_q(h) - h).abs().mean().item()   # head output differs from h
    print(f"[2] repr h dim={tuple(h.shape)} eval-emb dim={tuple(emb.shape)} "
          f"head!=h (no leak): {leak > 1e-3}")

    # [3] queue fills, f_k tracks f_q via EMA, loss is non-trivial (task is hard).
    before = [p.detach().clone() for p in moco.encoder_k.parameters()]
    losses = []
    moco.train()
    opt = torch.optim.Adam([p for p in moco.parameters() if p.requires_grad], lr=1e-3)
    for step in range(6):
        fq, fk = collate(v1s), collate(v2s)
        loss = moco(fq, fk)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    moved = max((p - b).abs().max().item() for p, b in zip(moco.encoder_k.parameters(), before))
    print(f"[3] queue ptr advanced to {int(moco.queue_ptr)} (K={moco.K}) | "
          f"f_k moved via EMA: {moved > 0} (max {moved:.2e})")
    print(f"    losses: {[round(l, 3) for l in losses]}")
    nontrivial = losses[0] > 0.1          # not collapsing to ~0 immediately
    print(f"[A] task non-trivial (loss not ~0): {nontrivial}")
    ok = leak > 1e-3 and int(moco.queue_ptr) > 0 and moved > 0 and nontrivial
    print("[smoke]", "PASS" if ok else "FAIL")
