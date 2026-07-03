"""
RBF distance-encoding smoke test + diagnostics (plan_implementation §1, steps 2–3).

Validates Modification 1 (Gaussian-RBF distance encoding) on three axes:

  A. Forward smoke   — AsymmetricTopoNet (sinusoidal vs rbf) and EquivariantTopoNet
                       (rbf) run on a real collated batch and return finite,
                       L2-normalized graph embeddings of the right shape.
  B. Reconstruction  — a tiny MLP recovers the raw distance d from enc(d). The
     (plan step 2)     encoding should be informative/invertible on the contact
                       range; RBF should beat sinusoidal (whose high bands are dead).
  C. Dimension usage — per-dimension std (count of ~dead dims) and effective rank
     (plan step 3)     (participation ratio) over the REAL edge-distance
                       distribution. Sinusoidal is expected to have ~half its dims
                       near-constant; RBF should use its dimensions far better.

Run:  python scripts/utilities/rbf_diagnostics.py
"""
import glob
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rbf import GaussianRBF  # noqa: E402


def sample_edges(n_proteins=400, seed=0):
    """Edge distances + their stored sinusoidal encodings from real .pt files."""
    random.seed(seed)
    sub = "data/subbase_corrected_train.txt"
    if os.path.exists(sub):
        paths = [l.strip() for l in open(sub)][:n_proteins]
    else:
        paths = random.sample(sorted(glob.glob("data/hoan_processed/*.pt")), n_proteins)
    dists, sins = [], []
    for p in paths:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        dists.append(d["rank1"]["distance"].flatten())
        sins.append(d["rank1"]["distance_encoding"].flatten(0, 1))  # (E,16)
    return torch.cat(dists), torch.cat(sins), paths


def effective_rank(enc):
    """Participation ratio of singular values of the centered encoding matrix:
    (Σσ)² / Σσ²  -> a soft count of 'used' dimensions (1..D)."""
    x = enc - enc.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.double())
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))


def dead_dims(enc, frac=0.02):
    """Dimensions whose std is < frac of the max per-dim std (near-constant)."""
    sd = enc.std(0)
    return int((sd < frac * sd.max()).sum()), sd


def reconstruction_mae(enc, d, steps=400, seed=0):
    """Train a tiny MLP enc->d; return held-out MAE in Å (lower = more invertible)."""
    torch.manual_seed(seed)
    n = d.shape[0]
    idx = torch.randperm(n)
    ntr = int(0.8 * n)
    tr, te = idx[:ntr], idx[ntr:]
    xtr, ytr = enc[tr], d[tr].unsqueeze(-1)
    xte, yte = enc[te], d[te].unsqueeze(-1)
    net = torch.nn.Sequential(torch.nn.Linear(enc.shape[1], 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((net(xte) - yte).abs().mean())


def forward_smoke():
    """Run both models on a real collated+bucketed batch in each encoding mode."""
    from train import custom_collate
    from contrastive_data import pad_to_buckets
    from asymmetric_topotein import AsymmetricTopoNet
    from equivariant_topotein import EquivariantTopoNet

    files = sorted(glob.glob("data/hoan_processed/*.pt"))[:6]
    batch = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d["rank1"]["source"].shape[1] == 16:
            batch.append((d, f))
    feats = custom_collate(batch)
    feats, realB = pad_to_buckets(feats)

    def check(name, model):
        model.eval()
        with torch.no_grad():
            z = model(feats)
        ok = torch.isfinite(z).all() and abs(float(z.norm(dim=-1).mean()) - 1.0) < 1e-3
        print(f"    {name:42s} out={tuple(z.shape)} finite={bool(torch.isfinite(z).all())} "
              f"|z|≈{float(z.norm(dim=-1).mean()):.3f}  {'OK' if ok else 'FAIL'}")
        return ok

    print("\n[A] Forward smoke (real batch):")
    ok = True
    ok &= check("AsymmetricTopoNet dist_encoding='sinusoidal'",
                AsymmetricTopoNet(scalar_dim=128, dist_encoding="sinusoidal"))
    ok &= check("AsymmetricTopoNet dist_encoding='rbf' K=16",
                AsymmetricTopoNet(scalar_dim=128, dist_encoding="rbf", rbf_dim=16))
    ok &= check("AsymmetricTopoNet dist_encoding='rbf' K=32",
                AsymmetricTopoNet(scalar_dim=128, dist_encoding="rbf", rbf_dim=32))
    ok &= check("EquivariantTopoNet rbf_dim=16 scalarize='frame'",
                EquivariantTopoNet(scalar_dim=128, vector_dim=16, num_layers=3,
                                   scalarize="frame", rbf_dim=16))
    return ok


def main():
    print("Sampling real edge distances...")
    d, sin, paths = sample_edges()
    print(f"  {d.numel():,} edges from {len(paths)} proteins; "
          f"d range [{d.min():.2f}, {d.max():.2f}] Å, median {d.median():.2f}")

    rbf16 = GaussianRBF(16)(d)
    rbf32 = GaussianRBF(32)(d)

    print("\n[C] Dimension usage over the real distance distribution:")
    print(f"    {'encoding':22s} {'dims':>5} {'dead':>5} {'eff.rank':>9}")
    for name, enc in [("sinusoidal (stored)", sin), ("RBF K=16", rbf16), ("RBF K=32", rbf32)]:
        dead, _ = dead_dims(enc)
        print(f"    {name:22s} {enc.shape[1]:>5} {dead:>5} {effective_rank(enc):>9.2f}")

    print("\n[B] Reconstruction MAE (recover d from enc(d); lower = more invertible):")
    for name, enc in [("sinusoidal (stored)", sin), ("RBF K=16", rbf16), ("RBF K=32", rbf32)]:
        print(f"    {name:22s} MAE = {reconstruction_mae(enc, d):.3f} Å")

    ok = forward_smoke()
    print(f"\n{'PASS' if ok else 'FAIL'} — RBF encoding smoke test")


if __name__ == "__main__":
    main()
