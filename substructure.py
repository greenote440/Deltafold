"""
Substructure sampling for contrastive positives (plan_implementation_3 §1).

Replaces the trivial jitter/crop/mask augmentation (run-1 post-mortem H1: two
near-identical noised copies -> loss ~1e-5, nothing learned) with **two distinct
connected substructures of the same protein** as the positive pair (Hermosilla &
Ropinski, 2205.15675). Two different regions of the same fold is a non-trivial
task that cannot be solved by noise-invariance.

A substructure is a **CA-only re-lifting restricted to a residue subset S**: the
per-residue features are sub-set, and the geometry (rank-1 kNN graph, rank-2 SSE
PCA, rank-3 globals) is *recomputed* on ``ca_coords[S]`` with the same formulas as
``topotein_lifter.py``. The output dict has the exact format ``custom_collate``
and the models consume, so it is a drop-in for the dataset transform.

Default mode is ``contiguous`` (a chain segment — preserves sequence locality and
keeps SSE membership contiguous, so the rank-2 sub-ranges stay valid). Size is the
critical hyperparameter (InfoMin sweet spot): a fraction ``f`` of residues drawn
per view; too small -> views no longer share the fold (false positives), too large
-> views near-identical (back to the run-1 problem). Floor of 17 residues keeps the
k=16 kNN graph valid.

Runs on CPU in the DataLoader workers (like the old jitter), so everything here is
device-agnostic CPU tensors.
"""
import math
import random

import torch

K_NN = 16          # contact-graph degree (must match the collate's shape[1]==16 check)
MIN_SUB = K_NN + 1  # >=17 residues so the kNN graph has 16 neighbours


# --- geometry helpers (mirror topotein_lifter.py, restricted to a subset) ----
def _knn_graph(ca, k=K_NN):
    """kNN contact graph on CA coords -> (source, target, distance, vector),
    each shaped (n, k); mirrors lift_rank1_edges (self-loop dropped)."""
    n = ca.shape[0]
    k_act = min(k + 1, n)
    dmat = torch.cdist(ca, ca)
    distances, indices = torch.topk(dmat, k=k_act, largest=False, dim=1)
    distances, indices = distances[:, 1:], indices[:, 1:]          # drop self (d=0)
    sources = torch.arange(n).view(-1, 1).expand(-1, k_act - 1)
    vectors = ca[indices] - ca[sources]
    return sources.contiguous(), indices.contiguous(), distances.contiguous(), vectors.contiguous()


def _sin_encoding(values, d_model=16):
    """Sinusoidal encoding of a tensor of scalars -> (*values.shape, d_model).
    Matches the stored distance_encoding / positional_encoding formula."""
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    flat = values.reshape(-1, 1).float()
    pe = torch.zeros(flat.shape[0], d_model)
    pe[:, 0::2] = torch.sin(flat * div)
    pe[:, 1::2] = torch.cos(flat * div)
    return pe.view(*values.shape, d_model)


def _pca(coords):
    """(eigenvalues[3] desc, shape_descriptors[5]); mirrors get_pca_features."""
    if coords.shape[0] < 3:
        return torch.zeros(3), torch.zeros(5)
    com = coords.mean(dim=0)
    centered = coords - com
    cov = (centered.T @ centered) / (coords.shape[0] - 1)
    ev = torch.linalg.eigh(cov).eigenvalues          # CPU; ascending
    ev = ev.flip(0)                                  # descending
    e1, e2, e3 = ev + 1e-8
    sd = torch.stack([(e1 - e2) / e1,
                      2 * (e2 - e3) / (e1 + e2),
                      3 * e3 / (e1 + e2 + e3),
                      torch.sign(e1 * e2 * e3) * torch.abs(e1 * e2 * e3).pow(1.0 / 3.0),
                      (e1 - e3) / e1])
    return ev, sd


def _per_residue_sse(rank2, n):
    """Per-residue (sse_local_id, type_onehot) from the rank-2 SSE list. SSEs
    partition the chain (DSSP H/E/C), so every residue gets exactly one."""
    labels = [-1] * n
    types = [None] * n
    for j, sse in enumerate(rank2):
        for r in range(int(sse['start_idx']), int(sse['end_idx']) + 1):
            if 0 <= r < n:
                labels[r] = j
                types[r] = sse['type']
    return labels, types


# --- substructure construction ----------------------------------------------
def sample_substructure(data, f_range=(0.5, 0.8), mode='contiguous', rng=None,
                        start=None):
    """Return a CA-only re-lifted PCC dict for a connected residue subset.

    data : a full PCC dict (rank0/rank1/rank2/rank3) as stored in the .pt files.
    f_range : (lo, hi); the subset size is round(U(lo,hi) * N), floored at 17.
    mode : 'contiguous' (chain segment, default) or 'ball' (spatial kNN of a centre).
    start : optional fixed segment start (contiguous mode) — used to force two
            distinct views.
    """
    rng = rng or random
    r0 = data['rank0']
    N = r0['aa'].shape[0]
    if N <= MIN_SUB + 1:                       # too small to sub-sample meaningfully
        return _clone(data)

    n_sub = int(round(rng.uniform(*f_range) * N))
    n_sub = max(MIN_SUB, min(n_sub, N - 1))

    if mode == 'ball':
        idx = _ball_indices(r0['ca_coords'], n_sub, rng)
    else:
        if start is None:
            start = rng.randint(0, N - n_sub)
        idx = torch.arange(start, start + n_sub)

    return _relift(data, idx)


def _ball_indices(ca, n_sub, rng):
    """Spatial substructure: the n_sub nearest 3D neighbours of a random centre.
    Returned SORTED (so SSE sub-ranges stay as contiguous as possible)."""
    centre = rng.randint(0, ca.shape[0] - 1)
    d = torch.cdist(ca[centre:centre + 1], ca).squeeze(0)
    idx = torch.topk(d, k=n_sub, largest=False).indices
    return idx.sort().values


def _relift(data, idx):
    """Build the sub-PCC for residue indices ``idx`` (1-D LongTensor)."""
    r0, r2 = data['rank0'], data['rank2']
    idx = idx.long()
    n = idx.shape[0]
    ca = r0['ca_coords'][idx].float()

    # rank0: subset every per-residue feature
    new_r0 = {k: v[idx].clone() for k, v in r0.items()}

    # rank1: recompute the kNN contact graph on the subset
    src, tgt, dist, vec = _knn_graph(ca, K_NN)
    new_r1 = {'source': src, 'target': tgt, 'distance': dist, 'vector': vec,
              'distance_encoding': _sin_encoding(dist, 16)}

    # rank2: regroup the subset's residues by their (original) SSE id into
    # contiguous sub-SSEs, recompute PCA features on the member CAs.
    labels, types = _per_residue_sse(r2, r0['aa'].shape[0])
    sub_labels = [labels[int(i)] for i in idx]
    sub_types = [types[int(i)] for i in idx]
    new_r2 = []
    p = 0
    while p < n:
        lab = sub_labels[p]
        q = p
        while q + 1 < n and sub_labels[q + 1] == lab:
            q += 1
        if lab != -1:
            ev, sd = _pca(ca[p:q + 1])
            new_r2.append({'type': sub_types[p].clone(), 'size': q - p + 1,
                           'start_idx': p, 'end_idx': q,
                           'eigenvalues': ev, 'shape_descriptors': sd})
        p = q + 1

    # rank3: recompute globals on the subset
    ev, sd = _pca(ca)
    centered = ca - ca.mean(dim=0)
    rog = torch.sqrt((centered ** 2).sum(dim=1).mean())
    new_r3 = {'protein_size': n, 'radius_of_gyration': rog,
              'global_eigenvalues': ev, 'global_shape_descriptors': sd}

    return {'rank0': new_r0, 'rank1': new_r1, 'rank2': new_r2, 'rank3': new_r3}


def _clone(obj):
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    return obj


class SubstructureViews:
    """Dataset transform: a single PCC dict -> two DISTINCT substructure views
    (the positive pair). Drop-in for ``StructuralAugmentations`` (same __call__
    contract). Optional light feature masking is available but off by default."""

    def __init__(self, f_range=(0.5, 0.8), mode='contiguous', mask_ratio=0.0, seed=None):
        self.f_range = f_range
        self.mode = mode
        self.mask_ratio = mask_ratio
        # A Random INSTANCE (never the bare `random` module) so the transform stays
        # picklable for DataLoader worker spawn. seed=None -> per-process entropy
        # (workers draw different substructures, which is what we want).
        self._rng = random.Random(seed)

    def _maybe_mask(self, view):
        if self.mask_ratio <= 0:
            return view
        n = view['rank0']['aa'].shape[0]
        k = int(n * self.mask_ratio)
        if k > 0:
            m = self._rng.sample(range(n), k)
            view['rank0']['aa'][m] = 0.0
            view['rank0']['3di'][m] = 0.0
        return view

    def __call__(self, pcc):
        N = pcc['rank0']['aa'].shape[0]
        v1 = sample_substructure(pcc, self.f_range, self.mode, self._rng)
        v2 = sample_substructure(pcc, self.f_range, self.mode, self._rng)
        # Ensure the two views are genuinely distinct (contiguous mode): if they
        # collide, reshift the second segment.
        if self.mode == 'contiguous' and N > MIN_SUB + 1:
            tries = 0
            while torch.equal(v1['rank0']['ca_coords'], v2['rank0']['ca_coords']) and tries < 5:
                v2 = sample_substructure(pcc, self.f_range, self.mode, self._rng)
                tries += 1
        return self._maybe_mask(v1), self._maybe_mask(v2)


# --- smoke test (plan_implementation_3 §Tests 1) -----------------------------
if __name__ == "__main__":
    import glob
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    pt = sorted(glob.glob('data/hoan_processed/*.pt'))
    if pt:
        data = torch.load(pt[0], map_location='cpu', weights_only=False)
        for k in ('aa', 'ca_coords'):
            data['rank0'][k] = data['rank0'][k].cpu()
    else:
        raise SystemExit("no .pt files to smoke-test on")

    N = data['rank0']['aa'].shape[0]
    rng = random.Random(0)
    sampler = SubstructureViews(f_range=(0.5, 0.8), seed=0)
    v1, v2 = sampler(data)
    n1, n2 = v1['rank0']['aa'].shape[0], v2['rank0']['aa'].shape[0]
    print(f"[1] protein N={N} -> view1 {n1} res / {len(v1['rank2'])} SSE, "
          f"view2 {n2} res / {len(v2['rank2'])} SSE")

    # PCC validity: kNN degree 16, dims consistent
    for nm, v in (('v1', v1), ('v2', v2)):
        assert v['rank1']['source'].shape[1] == 16, f"{nm} k!=16"
        assert v['rank1']['vector'].shape == (v['rank0']['aa'].shape[0], 16, 3)
        feat = torch.cat([v['rank2'][0]['type'], v['rank2'][0]['shape_descriptors'],
                          v['rank2'][0]['eigenvalues']], dim=-1)
        assert feat.shape[0] == 12, f"{nm} rank2 feat dim {feat.shape}"
    print("[2] PCC valid: k=16, rank1 vector (n,16,3), rank2 feat dim 12")

    # Distinct views + overlap (Jaccard) on contiguous indices
    import numpy as np
    # recover which residues each view covers by matching ca rows back to the parent
    def coverage(view):
        pca = data['rank0']['ca_coords']
        vca = view['rank0']['ca_coords']
        idx = []
        for row in vca:
            hit = (pca == row).all(dim=1).nonzero(as_tuple=True)[0]
            if len(hit):
                idx.append(int(hit[0]))
        return set(idx)
    s1, s2 = coverage(v1), coverage(v2)
    jac = len(s1 & s2) / max(1, len(s1 | s2))
    print(f"[3] views distinct={s1 != s2}  Jaccard overlap={jac:.2f}  "
          f"(want moderate: not 0, not 1)")

    # The sub-dicts must drive the model; check the equivariant model invariance
    # still holds on a substructure (geometry is self-consistent).
    from train import custom_collate
    from contrastive_data import pad_to_buckets
    from equivariant_topotein import EquivariantTopoNet
    import copy
    feats = custom_collate([(v1, 'a'), (v2, 'b')])
    feats, _ = pad_to_buckets(feats)
    model = EquivariantTopoNet(scalar_dim=64, vector_dim=8, num_layers=2,
                               scalarize='frame', rbf_dim=16).eval()
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    f2 = copy.deepcopy(feats)
    f2['rank0']['ca_coords'] = feats['rank0']['ca_coords'] @ Q.T
    f2['rank1']['vector'] = feats['rank1']['vector'] @ Q.T
    with torch.no_grad():
        za, zb = model(feats), model(f2)
    err = (za - zb).abs().max().item()
    print(f"[4] model runs on substructures; SE(3) invariance err={err:.2e}")
    ok = n1 >= 17 and n2 >= 17 and s1 != s2 and 0.0 < jac < 1.0 and err < 1e-4
    print("[smoke]", "PASS" if ok else "FAIL")
