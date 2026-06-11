"""
Data pipeline for contrastive training: batch construction, shape bucketing, and
collation.

Three concerns live here, all aimed at making per-step compute on MPS predictable:

  * `pad_to_buckets`          — pad collated batches up to a handful of fixed shapes
                                so MPSGraph reuses compiled kernels instead of
                                recompiling per unique shape.
  * `HardNegativeBatchSampler`— group superficially-similar proteins (length +
                                secondary-structure composition) into each batch so
                                InfoNCE cannot be solved by cheap separators, while
                                budgeting total residues per batch to bound compute.
  * `contrastive_collate` / `worker_init_fn` — assemble the two augmented views per
                                protein and keep DataLoader workers single-threaded.

`extract_batch_keys` mines the per-protein metadata the sampler needs and caches it.
"""
import os

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from train import custom_collate


def _ceil_to(x, m):
    return ((x + m - 1) // m) * m


def pad_to_buckets(features, node_bucket=1024, sse_bucket=256, prot_bucket=8):
    """Pad a collated PCC batch up to bucketed dimensions so the model sees only a
    SMALL set of distinct tensor shapes across the whole run.

    Why: residue-budgeted batches have a different node/edge/SSE/protein count every
    step. On MPS, MPSGraph compiles and caches a kernel per unique shape; with
    hundreds of unique shapes per epoch the kernel cache bloats (memory + dispatch
    overhead) and each epoch gets slower. Bucketing the shapes bounds the number of
    compiled kernels (~a handful) so they get reused.

    Correctness: padding is fully isolated from the real proteins. Dummy nodes are
    routed to a dummy protein slot (batch_idx_0), dummy edges self-loop on dummy
    nodes (so no real node receives messages from padding), and dummy SSEs collect
    only dummy nodes. Every op in the model is either per-element (LayerNorm/FFN) or
    a segment reduction keyed by these indices, so real outputs are unchanged.
    Returns (padded_features, real_B) where real_B = original protein count; slice
    the model output to [:real_B] before the loss.
    """
    r0, r1, r3 = features['rank0'], features['rank1'], features['rank3']
    dev = r0['aa'].device
    N = r0['aa'].shape[0]
    S = features['rank2_features'].shape[0]
    B = r3['protein_size'].shape[0]

    N_pad = _ceil_to(N, node_bucket)
    S_pad = _ceil_to(S + 1, sse_bucket)   # +1 guarantees a dummy SSE slot
    B_pad = _ceil_to(B + 1, prot_bucket)  # +1 guarantees a dummy protein slot
    dummy_prot, dummy_sse = B, S
    nd, ns, nb = N_pad - N, S_pad - S, B_pad - B

    # ---- rank0: zero-pad per-node features ----
    new_r0 = {}
    for k, v in r0.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N and nd > 0:
            pad = torch.zeros((nd,) + tuple(v.shape[1:]), dtype=v.dtype, device=dev)
            new_r0[k] = torch.cat([v, pad], dim=0)
        else:
            new_r0[k] = v

    # ---- rank1: pad edges; dummy edges self-loop on their own dummy node ----
    new_r1 = {}
    for k, v in r1.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N and nd > 0:
            if k in ('source', 'target'):
                pad = torch.arange(N, N_pad, device=dev).view(-1, 1).expand(-1, v.shape[1])
                new_r1[k] = torch.cat([v, pad], dim=0)
            else:
                pad = torch.zeros((nd,) + tuple(v.shape[1:]), dtype=v.dtype, device=dev)
                new_r1[k] = torch.cat([v, pad], dim=0)
        else:
            new_r1[k] = v

    # ---- rank2 features + nodes_per_sse ----
    r2f = features['rank2_features']
    nps = features['nodes_per_sse']
    if ns > 0:
        r2f = torch.cat([r2f, torch.zeros((ns,) + tuple(r2f.shape[1:]), dtype=r2f.dtype, device=dev)], dim=0)
        nps = torch.cat([nps, torch.ones((ns,) + tuple(nps.shape[1:]), dtype=nps.dtype, device=dev)], dim=0)

    # ---- index maps: dummy nodes -> dummy_sse / dummy_prot; dummy sses -> dummy_prot ----
    sm = features['sse_map_0']
    bi0 = features['batch_idx_0']
    bi2 = features['batch_idx_2']
    if nd > 0:
        sm = torch.cat([sm, torch.full((nd,), dummy_sse, dtype=sm.dtype, device=dev)], dim=0)
        bi0 = torch.cat([bi0, torch.full((nd,), dummy_prot, dtype=bi0.dtype, device=dev)], dim=0)
    if ns > 0:
        bi2 = torch.cat([bi2, torch.full((ns,), dummy_prot, dtype=bi2.dtype, device=dev)], dim=0)

    # ---- rank3: pad dummy proteins (discarded after the readout) ----
    new_r3 = {}
    for k, v in r3.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == B and nb > 0:
            fill = 1.0 if k == 'protein_size' else 0.0  # avoid div-by-zero on size
            pad = torch.full((nb,) + tuple(v.shape[1:]), fill, dtype=v.dtype, device=dev)
            new_r3[k] = torch.cat([v, pad], dim=0)
        else:
            new_r3[k] = v

    out = dict(features)
    out['rank0'], out['rank1'], out['rank3'] = new_r0, new_r1, new_r3
    out['rank2_features'], out['nodes_per_sse'] = r2f, nps
    out['sse_map_0'], out['batch_idx_0'], out['batch_idx_2'] = sm, bi0, bi2
    return out, B


def extract_batch_keys(files, cache_path=None):
    """For each training file, extract (protein_size, helix_ratio, sheet_ratio) from
    its Rank-2 SSE list. These drive the hard-negative sampler so that batches group
    proteins that are superficially similar (length + secondary-structure composition).
    Results are cached to avoid re-reading every .pt on subsequent runs."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, weights_only=False)
        except Exception:
            cache = {}
    keys, dirty = [], False
    for f in tqdm(files, desc="Mining batch metadata"):
        if f in cache:
            keys.append(cache[f]); continue
        try:
            try:
                d = torch.load(f, map_location='cpu', weights_only=False)
            except TypeError:
                d = torch.load(f, map_location='cpu')
            N = int(d['rank3']['protein_size'])
            h = e = 0
            for sse in d['rank2']:
                t = sse['type']
                sz = int(sse.get('size', sse['end_idx'] - sse['start_idx'] + 1))
                if t[0] == 1:
                    h += sz
                elif t[1] == 1:
                    e += sz
            k = (N, h / max(N, 1), e / max(N, 1))
        except Exception:
            k = (0, 0.0, 0.0)
        cache[f] = k
        keys.append(k)
        dirty = True
    if cache_path and dirty:
        try:
            torch.save(cache, cache_path)
        except Exception:
            pass
    return keys


class HardNegativeBatchSampler(Sampler):
    """Hard-negative mining via batch construction (report 7, "the single most
    important intervention"). Groups dataset indices into length bins, and within
    each bin orders by helix ratio, so every contiguous batch contains proteins of
    similar length and similar secondary-structure composition -- superficially
    alike but topologically distinct. This denies the model cheap separators
    (length / composition) and forces it to use fold topology to solve InfoNCE.

    Per-epoch noise on the bin assignment and ordering keeps batches varied across
    epochs; call set_epoch(epoch) before each epoch.

    `max_residues` caps the TOTAL residues per batch (not just the protein count):
    model compute scales with residues/edges, so without this cap the batches that
    group the longest proteins explode to 25-30k residues -> tens of seconds/step
    (and can even abort the MPS backend). Budgeting by residues makes every step
    roughly equal, small compute and also stabilizes tensor shapes (fewer MPS
    kernel recompiles). batch_size acts as a hard upper bound on the count."""
    def __init__(self, keys, batch_size, seed=42, length_jitter=0.05, ratio_jitter=0.02,
                 max_residues=10000):
        self.keys = keys
        self.batch_size = max(2, batch_size)
        self.n = len(keys)
        self.seed = seed
        self.length_jitter = length_jitter
        self.ratio_jitter = ratio_jitter
        self.max_residues = max_residues
        self.epoch = 0
        self._total_res = sum(int(k[0]) for k in keys)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        # Approximate batch count under the residue budget (whichever binds first).
        by_residues = self._total_res // max(1, self.max_residues)
        by_count = self.n // self.batch_size
        return max(1, by_residues, by_count)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        lengths = np.array([k[0] for k in self.keys], dtype=float)
        hr = np.array([k[1] for k in self.keys], dtype=float)

        # Rank-based equal-size length bins (~4 batches per bin), with per-epoch jitter
        n_bins = max(1, self.n // (self.batch_size * 4))
        lj = lengths * (1 + rng.normal(0, self.length_jitter, self.n))
        rank = np.argsort(np.argsort(lj))
        bins = (rank * n_bins // max(self.n, 1)).astype(int)

        order = []
        for b in range(n_bins):
            idx = np.where(bins == b)[0]
            if len(idx) == 0:
                continue
            hj = hr[idx] + rng.normal(0, self.ratio_jitter, len(idx))
            order.extend(idx[np.argsort(hj)].tolist())

        # Build batches by accumulating along the length-sorted order until either
        # the residue budget or the count cap is hit. This bounds per-step compute.
        batches, cur, cur_res = [], [], 0
        for idx in order:
            r = int(lengths[idx])
            if cur and (cur_res + r > self.max_residues or len(cur) >= self.batch_size):
                batches.append(cur)
                cur, cur_res = [], 0
            cur.append(idx)
            cur_res += r
        if cur:
            batches.append(cur)
        batches = [b for b in batches if len(b) >= 2]  # InfoNCE needs >=2 anchors
        rng.shuffle(batches)
        for b in batches:
            yield b


def contrastive_collate(batch):
    # Filter invalid/small proteins
    valid_batch = [b for b in batch if b[0] is not None and b[0][0] is not None
                   and b[0][0]['rank1']['source'].shape[1] == 16]

    if not valid_batch: return None

    # Concatenate view1s and view2s into a single combined list of 2*B proteins
    # b[0] is (view1, view2), b[1] is path
    views_1 = [b[0][0] for b in valid_batch]
    views_2 = [b[0][1] for b in valid_batch]
    paths = [b[1] for b in valid_batch] + [b[1] for b in valid_batch]

    return custom_collate(views_1 + views_2), paths


def worker_init_fn(worker_id):
    import torch
    torch.set_num_threads(1)
