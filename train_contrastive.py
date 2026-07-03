"""
Contrastive Learning Training Loop
"""
import os
import glob
import re
import csv
import ctypes
import gzip
import json
import torch
import torch.profiler
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import numpy as np
import random
import gc
import os
import psutil

from topotein import Topotein
from asymmetric_topotein import AsymmetricTopoNet
from contrastive_engine import StructuralAugmentations, NTXentLoss
from substructure import SubstructureViews
from moco import MoCo
from train import PCCDataset, custom_collate, to_device, get_cluster_aware_split, DEVICE, PROC_DIR, CHECKPOINT_DIR, CLUSTER_TSV

_PROC = psutil.Process(os.getpid())

# Number of train proteins in the original downsampled run. Kept in sync with
# extract_embeddings.py so `--downsampled` here and there select the same subset.
DOWNSAMPLED_DATASET_SIZE = 2918

# How often (in gradient steps) to write a structured per-step record to the
# training log (loss, lr, footprint, collapse-health). Env-tunable so the
# overnight sweep can dial granularity without code changes. 0 disables.
STEP_LOG_EVERY = int(os.environ.get('DELTAFOLD_LOG_EVERY', '20'))


# ── macOS phys_footprint via Mach task_info ─────────────────────────────────
# On Apple Silicon unified memory, MPS (GPU) allocations are managed by the
# driver in a separate pool that is NOT counted in psutil RSS. For example,
# after a real model forward pass RSS reports ~0.3 GB while Activity Monitor
# (and this API) shows ~1.5 GB — the gap is 100% MPS allocations.  phys_
# footprint = what Activity Monitor calls "Memory" and what triggers swap.
class _TaskVMInfo(ctypes.Structure):
    _fields_ = [
        ('virtual_size',                ctypes.c_uint64),
        ('region_count',                ctypes.c_int32),
        ('page_size',                   ctypes.c_int32),
        ('resident_size',               ctypes.c_uint64),
        ('resident_size_peak',          ctypes.c_uint64),
        ('device',                      ctypes.c_uint64),
        ('device_peak',                 ctypes.c_uint64),
        ('internal',                    ctypes.c_uint64),
        ('internal_peak',               ctypes.c_uint64),
        ('external',                    ctypes.c_uint64),
        ('external_peak',               ctypes.c_uint64),
        ('reusable',                    ctypes.c_uint64),
        ('reusable_peak',               ctypes.c_uint64),
        ('purgeable_volatile_pmap',     ctypes.c_uint64),
        ('purgeable_volatile_resident', ctypes.c_uint64),
        ('purgeable_volatile_virtual',  ctypes.c_uint64),
        ('compressed',                  ctypes.c_uint64),
        ('compressed_peak',             ctypes.c_uint64),
        ('compressed_lifetime',         ctypes.c_uint64),
        ('phys_footprint',              ctypes.c_uint64),
    ]

_TASK_VM_INFO = 22
try:
    _libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    _tvi_count = ctypes.c_uint32(ctypes.sizeof(_TaskVMInfo) // 4)
    _HAS_MACH = True
except Exception:
    _HAS_MACH = False


def _phys_footprint_gb():
    """Physical memory footprint (GiB) matching Activity Monitor's 'Memory' column.
    Includes both CPU-side RSS and MPS (GPU) driver allocations on Apple Silicon.
    Falls back to psutil RSS on non-macOS."""
    if _HAS_MACH:
        try:
            info = _TaskVMInfo()
            c = ctypes.c_uint32(_tvi_count.value)
            if _libc.task_info(_libc.mach_task_self(), _TASK_VM_INFO,
                               ctypes.byref(info), ctypes.byref(c)) == 0:
                return info.phys_footprint / (1024 ** 3)
        except Exception:
            pass
    return _PROC.memory_info().rss / (1024 ** 3)


def _report_memory(step, epoch):
    return _phys_footprint_gb()


class MemoryGovernor:
    """Keeps the process RSS under a hard cap on this 16GB machine.

    The training loop leaks across an epoch (DataLoader worker buffers + a growing/
    fragmented MPS allocator pool), creeping to ~27GB by epoch 5 -> swap -> 40s/step.
    Swapping is far slower than rebuilding the workers, so this governor escalates:

      every `cleanup_every` steps        -> baseline gc + empty_cache (cheap throttle)
      RSS > soft_gb                       -> force gc + empty_cache NOW (off-schedule)
      RSS > hard_gb (after a reclaim try) -> signal a COLD RESTART of the DataLoader
                                             workers and SHRINK the residue budget so
                                             subsequent batches allocate less.

    The residue budget recovers slowly across clean epochs so throughput is not
    permanently sacrificed after a transient spike."""

    def __init__(self, soft_gb=11.0, hard_gb=14.0, sampler=None, cleanup_every=50,
                 min_residues=2000, shrink=0.85, grow=1.1, no_budget_adapt=False):
        self.soft_gb = soft_gb
        self.hard_gb = hard_gb
        self.sampler = sampler
        self.cleanup_every = cleanup_every
        self.min_residues = min_residues
        self.shrink = shrink
        self.grow = grow
        self.no_budget_adapt = no_budget_adapt
        self.base_residues = getattr(sampler, 'max_residues', None)
        self.last_fp = 0.0
        self.restarts = 0
        self.soft_hits = 0
        self._epoch_restarts = 0
        self._last_restart_step = -100  # prevent restart spam (cooldown)

    def _reclaim(self):
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def after_step(self, step):
        """Call once per training step. Returns 'restart' if the caller should cold-
        restart the DataLoader workers, else None.

        Prevents restart spam: only allow restart if >50 steps since last restart.
        If memory stays high after restart, shrinking budget won't help; warn instead.
        """
        enabled = self.hard_gb and self.hard_gb > 0
        # Baseline throttled cleanup (unconditional, matches the old behaviour).
        if DEVICE.type == 'mps' and self.cleanup_every > 0 and (step % self.cleanup_every == 0):
            self._reclaim()
        if not enabled:
            self.last_fp = _phys_footprint_gb()
            return None
        rss = _phys_footprint_gb()
        if rss > self.hard_gb:
            self._reclaim()
            rss = _phys_footprint_gb()
            if rss > self.hard_gb:
                self.last_fp = rss
                # Prevent restart spam: only allow restart if 50+ steps since last one
                if step - self._last_restart_step >= 50:
                    self.restarts += 1
                    self._epoch_restarts += 1
                    if self.sampler is not None and self.base_residues and not self.no_budget_adapt:
                        self.sampler.max_residues = max(
                            self.min_residues, int(self.sampler.max_residues * self.shrink))
                    self._last_restart_step = step
                    return 'restart'
                # else: silently skip restart (memory governor is exhausted)
        elif rss > self.soft_gb:
            self.soft_hits += 1
            self._reclaim()
            rss = _phys_footprint_gb()
        self.last_fp = rss
        return None

    def end_of_epoch(self):
        """Slowly grow the residue budget back toward its base after a clean epoch; reset
        per-epoch counters. Returns a short status string for logging."""
        grew = False
        if self.sampler is not None and self.base_residues:
            if self._epoch_restarts == 0 and self.sampler.max_residues < self.base_residues:
                self.sampler.max_residues = min(
                    self.base_residues, int(self.sampler.max_residues * self.grow))
                grew = True
        msg = (f"restarts={self._epoch_restarts} soft_hits={self.soft_hits} "
               f"budget={getattr(self.sampler, 'max_residues', None)}"
               + (" (grown)" if grew else ""))
        self._epoch_restarts = 0
        self.soft_hits = 0
        return msg


# ── Consolidated training log ─────────────────────────────────────────────────
# Replaces the four separate verbose CSVs (contrastive_losses, collapse_metrics,
# ari_log, epoch_eval) with a single gzipped JSONL file written once per epoch.
# Each record is one JSON line — trivial to read, token-efficient to benchmark.
#
#   Per-epoch record ("epoch"):
#     ep, steps, train_loss, val_loss, ari (kmeans-val), fp_gb (peak footprint),
#     restarts, budget, eval {ari, nmi, tm_rho, recall, k_clusters, singletons}
#     + optional loss_range [min,max] for a quick sanity check
#
#   Sparse event records ("restart", "collapse"):
#     only written when something notable happens — keeps the file short.
#
# Usage (read everything):
#   python -c "import gzip,json; [print(json.loads(l)) for l in gzip.open('checkpoints/training_log.jsonl.gz')]"
#
# Filter to epoch summaries only:
#   python -c "import gzip,json; [print(l) for l in (json.loads(x) for x in gzip.open('checkpoints/training_log.jsonl.gz')) if l['t']=='epoch']"

class TrainingLog:
    """Buffers log events during an epoch and appends them as a gzip stream at
    epoch-end.  Multi-stream gzip is transparent to Python's gzip.open reader."""

    def __init__(self, path):
        self.path = path
        self._buf = []   # (dict,) records collected this epoch

    def event(self, rec):
        """Queue a sparse event record (restart, collapse, etc.)."""
        self._buf.append(rec)

    def flush_epoch(self, epoch_rec):
        """Write the epoch summary + any buffered events in one gzip stream."""
        records = [epoch_rec] + self._buf
        with gzip.open(self.path, 'ab') as fz:
            for rec in records:
                fz.write((json.dumps(rec, separators=(',', ':')) + '\n').encode())
        self._buf.clear()

    @staticmethod
    def read(path):
        """Return a list of all records from training_log.jsonl.gz."""
        recs = []
        with gzip.open(path, 'rb') as fz:
            for line in fz:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs


def free_memory():
    """Device-agnostic memory reclamation. Collects Python reference cycles and
    releases the framework's cached device allocations so that peak RAM does not
    creep upward from epoch to epoch (the leftover prefetch/worker buffers of a
    finished DataLoader iterator and a fragmented MPS/CUDA cache are the usual
    culprits behind 'each epoch gets slower'). The double gc.collect() handles
    objects whose __del__ resurrects references on the first pass."""
    gc.collect()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


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


# --- Per-epoch ARI logging helpers (numpy-only; sklearn is unavailable in ml_env) ---
def _adjusted_rand_index(labels_true, labels_pred):
    import numpy as np
    lt = np.asarray(labels_true); lp = np.asarray(labels_pred)
    _, t = np.unique(lt, return_inverse=True)
    _, p = np.unique(lp, return_inverse=True)
    n = t.shape[0]
    cont = np.zeros((t.max() + 1, p.max() + 1), dtype=np.int64)
    np.add.at(cont, (t, p), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    sc = c2(cont.sum(1)).sum(); sk = c2(cont.sum(0)).sum(); sij = c2(cont).sum()
    expected = sc * sk / c2(n) if n > 1 else 0.0
    maxidx = (sc + sk) / 2.0
    return 0.0 if maxidx - expected == 0 else (sij - expected) / (maxidx - expected)


def _spherical_kmeans(X, K, iters=25, seed=0):
    """Tiny cosine k-means (X assumed L2-normalized). Returns cluster assignments."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    first = int(rng.integers(n))
    centers_idx = [first]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, K):
        s = d2.sum()
        i = int(rng.choice(n, p=(d2 / s) if s > 0 else None))
        centers_idx.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    C = X[centers_idx].copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        assign = (X @ Cn.T).argmax(1)
        newC = np.zeros_like(C)
        for k in range(K):
            m = X[assign == k]
            newC[k] = m.mean(0) if len(m) else X[int(rng.integers(n))]
        if np.allclose(newC, C):
            break
        C = newC
    return assign


def compute_ari(embs, labels, seed=0):
    """ARI between a cosine-kmeans clustering of `embs` and ground-truth `labels`,
    restricted to multi-member clusters (singletons make k-means-vs-truth ill-posed).
    Returns (ari, n_eval, n_clusters) or (nan, 0, 0) if not enough structure."""
    import numpy as np
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    multi = set(uniq[counts >= 2])
    keep = np.array([l in multi for l in labels])
    if keep.sum() < 4 or len(multi) < 2:
        return float('nan'), int(keep.sum()), len(multi)
    X = np.asarray(embs, dtype=np.float64)[keep]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y = labels[keep]
    K = len(np.unique(y))
    pred = _spherical_kmeans(X, K, seed=seed)
    return float(_adjusted_rand_index(y, pred)), int(keep.sum()), K

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO"  # mirrors topotein_lifter / evaluate_correlation


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


class ResidueBudgetSampler(Sampler):
    """Batch by a total-residue budget instead of a fixed protein count.

    Without this, the default (count-based) loader puts `batch_size` proteins in
    every batch regardless of their sizes, so a batch that happens to draw several
    large viral proteins explodes -- on the equivariant tcpnet model that overflows
    the ~20GB MPS allocator. Greedily packs a per-epoch shuffle of indices into
    batches whose summed residue count stays under `max_residues` (with
    `batch_size` as a hard count cap), keeping per-step memory/compute bounded and
    roughly uniform. Exposes `max_residues` + `set_epoch` for the MemoryGovernor.
    """
    def __init__(self, lengths, batch_size, seed=42, max_residues=2000):
        self.lengths = [int(x) for x in lengths]
        self.n = len(self.lengths)
        self.batch_size = max(2, batch_size)
        self.seed = seed
        self.max_residues = max_residues
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        total = sum(self.lengths)
        return max(1, total // max(1, self.max_residues), self.n // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches, cur, cur_res = [], [], 0
        for idx in rng.permutation(self.n):
            idx = int(idx)
            r = self.lengths[idx]
            # Close on the count cap, or on the residue budget but ONLY once the
            # batch already holds >=2 proteins. That way a lone large protein
            # isn't dropped -- it just gets one partner and mildly overflows the
            # budget (still well under the OOM cap), and InfoNCE keeps >=2 anchors.
            if cur and (len(cur) >= self.batch_size or
                        (len(cur) >= 2 and cur_res + r > self.max_residues)):
                batches.append(cur)
                cur, cur_res = [], 0
            cur.append(idx)
            cur_res += r
        if len(cur) >= 2:
            batches.append(cur)
        elif cur and batches:
            batches[-1].extend(cur)  # fold a trailing singleton into the last batch
        rng.shuffle(batches)
        for b in batches:
            yield b


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


def collapse_metrics(z):
    """Per-batch collapse signals (report 7, "collapse detection during training").
    Returns (embedding std, mean off-diagonal cosine similarity). A shrinking std
    or a mean cosine approaching 1.0 indicates representation collapse."""
    import torch.nn.functional as F
    with torch.no_grad():
        zn = F.normalize(z, p=2, dim=-1)
        emb_std = zn.std(dim=0).mean().item()
        n = zn.size(0)
        sims = zn @ zn.t()
        off = sims[~torch.eye(n, dtype=torch.bool, device=zn.device)]
        mean_cos = off.mean().item()
    return emb_std, mean_cos


def tm_score_aux_loss(z, features, num_pairs=8):
    """Auxiliary TM-score regression (report 7, "add a TM-score regression loss").
    Directly supervises embedding cosine similarity to track TM-score ordering,
    reducing the contrastive/TM-score objective mismatch (report 3.4). Computed on
    a few sampled in-batch pairs; tmtools alignment runs on CPU. Returns None if no
    pair could be aligned."""
    try:
        import tmtools
    except ImportError:
        return None
    import numpy as np
    import torch.nn.functional as F

    batch_idx = features['batch_idx_0']
    ca = features['rank0']['ca_coords']
    aa = features['rank0']['aa']
    B2 = z.size(0)

    coords, seqs = [], []
    for b in range(B2):
        m = (batch_idx == b)
        c = ca[m].detach().cpu().numpy().astype(np.float32)
        if c.shape[0] < 3:
            coords.append(None); seqs.append(None); continue
        idx = aa[m].detach().cpu().numpy().argmax(axis=1)
        seqs.append("".join(AA_ALPHABET[i] if i < len(AA_ALPHABET) else "X" for i in idx))
        coords.append(c)

    valid = [b for b in range(B2) if coords[b] is not None]
    if len(valid) < 2:
        return None

    zn = F.normalize(z, p=2, dim=-1)
    terms = []
    for _ in range(num_pairs):
        i, j = random.sample(valid, 2)
        try:
            res = tmtools.tm_align(coords[i], coords[j], seqs[i], seqs[j])
            tm = max(res.tm_norm_chain1, res.tm_norm_chain2)
        except Exception:
            continue
        target = 2.0 * tm - 1.0          # map TM [0,1] -> cosine target [-1,1]
        pred = (zn[i] * zn[j]).sum()     # cosine similarity (z is unit-normalized)
        terms.append((pred - target) ** 2)
    if not terms:
        return None
    return torch.stack(terms).mean()


def tm_score_aux_loss_cached(z, paths, tm_cache, num_pairs=16):
    """Cached variant of the TM-score regression auxiliary (analysis §5.2).

    Uses pre-computed pairwise TM-scores (see build_tm_cache.py) instead of paying
    ~0.5s/pair on-the-fly tmtools alignment, which is what kept tm_score_aux_loss
    disabled. Directly optimises rho: it pulls each pair's embedding cosine toward
    2*TM-1, so minimising it is minimising the squared deviation of cosine from
    TM-score. `paths` is the 2B view-path list (aligned to z); two augmented views
    of the same protein (equal basename) get target TM=1."""
    import torch.nn.functional as F
    B = z.size(0)
    if B < 2 or not tm_cache:
        return None
    zn = F.normalize(z, p=2, dim=-1)
    bns = [os.path.basename(p) for p in paths]
    terms = []
    for _ in range(num_pairs):
        i, j = random.sample(range(B), 2)
        if bns[i] == bns[j]:
            tm = 1.0  # two augmented views of the same protein
        else:
            tm = tm_cache.get((bns[i], bns[j]))
            if tm is None:
                tm = tm_cache.get((bns[j], bns[i]))
        if tm is None:
            continue
        target = 2.0 * float(tm) - 1.0      # TM [0,1] -> cosine target [-1,1]
        pred = (zn[i] * zn[j]).sum()
        terms.append((pred - target) ** 2)
    if not terms:
        return None
    return torch.stack(terms).mean()


def build_tm_matrix(paths, tm_cache, device):
    """Dense N x N TM-score matrix for the in-batch proteins from the sparse cache
    (analysis §5.6). Unknown pairs are NaN (soft_supcon_loss falls back to binary
    weights there); identical basenames (the two views of one protein) are 1.0."""
    n = len(paths)
    bns = [os.path.basename(p) for p in paths]
    mat = torch.full((n, n), float('nan'), device=device)
    for i in range(n):
        for j in range(i + 1, n):
            if bns[i] == bns[j]:
                tm = 1.0
            else:
                tm = tm_cache.get((bns[i], bns[j]))
                if tm is None:
                    tm = tm_cache.get((bns[j], bns[i]))
            if tm is not None:
                mat[i, j] = mat[j, i] = float(tm)
    return mat


def soft_supcon_loss(embeddings, labels, tm_matrix, temperature=0.1):
    """Continuous (soft) supervised contrastive loss (analysis §5.6).

    Identical structure to supervised_ntxent_loss, but positive pairs are weighted
    by their actual TM-score rather than a binary same-cluster label. This turns the
    objective from "collapse each cluster to a point" (rho ceiling = 0) into "place
    proteins at embedding distances proportional to their structural distance" --
    the correct objective for a TM-score-correlated metric. Same-cluster pairs with
    no cached TM-score fall back to binary weight 1.0."""
    import torch.nn.functional as F
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    N = embeddings.shape[0]
    dev = embeddings.device

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    self_mask = 1.0 - torch.eye(N, device=dev)
    labels = labels.contiguous().view(-1, 1)
    labels_eq = torch.eq(labels, labels.T).float().to(dev) * self_mask   # same cluster, not self

    # Positive weights: TM-score where same-cluster AND cached, else binary fallback.
    pos_weights = labels_eq.clone()
    known = ~torch.isnan(tm_matrix)
    if known.any():
        pos_weights[known] = labels_eq[known] * tm_matrix[known].clamp(0, 1)
    row_sum = pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pos_weights_norm = pos_weights / row_sum

    exp_logits = torch.exp(logits) * self_mask
    log_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    log_prob = logits - log_denom

    has_pos = labels_eq.sum(1) > 0
    loss = -(pos_weights_norm * log_prob).sum(1)
    return loss[has_pos].mean() if has_pos.any() else logits.sum() * 0.0


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

def moco_collate(batch):
    """MoCo collate: keep the two views SEPARATE (view1 -> f_q, view2 -> f_k).
    Returns (feats_v1, feats_v2, paths) — each feats_* is an independently
    collated PCC batch of the same B proteins in the same order."""
    valid = [b for b in batch if b[0] is not None and b[0][0] is not None
             and b[0][1] is not None
             and b[0][0]['rank1']['source'].shape[1] == 16
             and b[0][1]['rank1']['source'].shape[1] == 16]
    if not valid:
        return None
    v1 = [b[0][0] for b in valid]
    v2 = [b[0][1] for b in valid]
    paths = [b[1] for b in valid]
    return custom_collate(v1), custom_collate(v2), paths

def worker_init_fn(worker_id):
    import torch
    torch.set_num_threads(1)

def supervised_ntxent_loss(embeddings, labels, temperature=0.1, hard_neg_beta=0.0):
    """
    Supervised Contrastive Loss (SupCon) with optional hard-negative reweighting.

    hard_neg_beta > 0 up-weights negatives that are most similar to the anchor
    (the hardest ones) in the denominator, matching the same mechanism used by
    NTXentLoss on the unsupervised path.  beta=0 is the standard SupCon loss.
    """
    # embeddings: (N, dim), labels: (N,)
    N = embeddings.shape[0]
    dev = embeddings.device

    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(dev)       # 1 where same cluster

    logits = torch.matmul(embeddings, embeddings.T) / temperature

    # Numerical stability shift
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # self_mask[i,i] = 0, elsewhere 1
    self_mask = 1.0 - torch.eye(N, device=dev)
    # positive mask excludes self
    pos_mask = pos_mask * self_mask
    # negative mask: not self, not positive
    neg_mask = self_mask * (1.0 - pos_mask)

    if hard_neg_beta <= 0:
        # Standard SupCon denominator: all non-self terms
        exp_logits = torch.exp(logits) * self_mask
        log_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    else:
        # Reweight negatives toward hard ones (same log-space trick as NTXentLoss).
        # Positives keep weight 1; negatives get weight proportional to exp(beta*sim).
        num_neg = neg_mask.sum(1, keepdim=True).clamp(min=1)
        w_logits = (hard_neg_beta * logits.detach()).masked_fill(neg_mask == 0, -float('inf'))
        log_norm = torch.logsumexp(w_logits, dim=1, keepdim=True)
        log_w_neg = torch.log(num_neg) + w_logits - log_norm   # (N,N), -inf off-negatives

        # Weighted denominator: positives contribute exp(logit), negatives exp(logit+log_w)
        pos_contrib = (torch.exp(logits) * pos_mask).sum(1, keepdim=True)
        neg_contrib = torch.exp(logits + log_w_neg.masked_fill(neg_mask == 0, -float('inf')))
        neg_contrib = neg_contrib.sum(1, keepdim=True)
        log_denom = torch.log(pos_contrib + neg_contrib + 1e-6)

    log_prob = logits - log_denom
    # Mean log-likelihood over positives (ignore rows with no positive)
    n_pos = pos_mask.sum(1)
    has_pos = n_pos > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (n_pos + 1e-6)
    return -mean_log_prob_pos[has_pos].mean() if has_pos.any() else logits.sum() * 0.0

def train_contrastive(model_type='topotein', epochs=30, batch_size=16, lr=1e-4, accum_steps=1, dataset_size=None, profile_train=False,
                      use_positional_encoding=True, use_residue_features=True, use_3di_features=True,
                      hard_neg_beta=0.0, split='cluster', tm_aux_weight=0.0, supervised=True,
                      hard_neg_mining=False, cleanup_every=50, max_residues=4000, pad_buckets=True,
                      tm_cache_path=None, soft_supcon=False, use_crop=False, jitter_sigma=0.3,
                      edge_attn_softmax=True, dist_bias_gamma=0.1, detach_h3=True,
                      dist_encoding='sinusoidal', rbf_dim=16,
                      mem_soft_gb=11.0, mem_hard_gb=14.0, min_residues=2000,
                      no_budget_adapt=False, downsampled=False,
                      num_layers=None, emb_dim=None, knn=None,
                      num_message_layers=None, num_feedforward_layers=None,
                      temperature=0.1,
                      objective='infonce', moco_k=8192, moco_m=0.99,
                      sub_f_lo=0.5, sub_f_hi=0.8, sub_mode='contiguous',
                      scalarize='frame', vector_dim=16,
                      tensor_diagram='default', readout='node',
                      num_workers=None,
                      disable_typecheck=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tlog = TrainingLog(os.path.join(CHECKPOINT_DIR, 'training_log.jsonl.gz'))

    # Residue-budget defaults are model-aware. The equivariant tcpnet/topotein
    # model (per-edge + per-SSE attention, vector channels, 6 layers) costs far
    # more memory per residue on MPS than the asymmetric model, so it needs a much
    # smaller per-batch residue budget to stay under the ~20GB MPS allocator cap.
    # train.py passes None when the user didn't override; resolve it here.
    is_tcpnet_model = (model_type != 'asymmetric')
    if max_residues is None:
        max_residues = 1800 if is_tcpnet_model else 3500
    if min_residues is None:
        min_residues = 1200 if is_tcpnet_model else 3500
    print(f"Residue budget: max={max_residues} min={min_residues} (model={model_type})")

    from train import get_split
    train_files, val_files = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split=split)

    if downsampled:
        # Restrict to the same ~3647 proteins (train+val) as the original
        # downsampled run, reconstructed from the deterministic cluster-aware
        # split (seed=42). Mirrors extract_embeddings.py's `_downsampled_files()`
        # so training and embedding extraction operate on the identical subset.
        val_size = int(DOWNSAMPLED_DATASET_SIZE * (1.0 - 0.8) / 0.8)
        train_files = train_files[:DOWNSAMPLED_DATASET_SIZE]
        val_files = val_files[:val_size]
        print(f"Using downsampled sub-dataset: {len(train_files)} train, "
              f"{len(val_files)} val proteins (seed=42 cluster split).")

    if dataset_size is not None:
        print(f"Limiting dataset to {dataset_size} training samples.")
        train_files = train_files[:dataset_size]
        val_size = int(dataset_size * (1.0 - 0.8) / 0.8) # Keep split ratio
        val_files = val_files[:val_size]

    if not train_files:
        print(f"No .pt files found in {PROC_DIR}.")
        return

    # Load Cluster Labels for Supervised Contrastive
    acc_to_cluster = {}
    if os.path.exists(CLUSTER_TSV):
        from train import extract_accession
        with open(CLUSTER_TSV, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    acc_to_cluster[extract_accession(parts[1])] = extract_accession(parts[0])

    is_moco = (objective == 'moco')
    if is_moco:
        # plan_implementation_3: positives = two distinct connected substructures
        # (non-trivial task), replacing the trivial jitter/crop/mask views.
        aug = SubstructureViews(f_range=(sub_f_lo, sub_f_hi), mode=sub_mode)
        hard_neg_mining = False                     # MoCo queue replaces hard-neg
        print(f"Objective: MoCo (substructure {sub_mode} f∈[{sub_f_lo},{sub_f_hi}], "
              f"K={moco_k}, m={moco_m}, tau={temperature})")
    else:
        # §5.5: jitter-only augmentation by default (use_crop=False, sigma=0.3A). SSE
        # cropping is TM-score-destructive; it teaches invariance to domain-level
        # structural differences that TM-score is meant to measure.
        aug = StructuralAugmentations(jitter_sigma=jitter_sigma, drop_ratio_range=(0.1, 0.2),
                                      mask_ratio=0.15, use_crop=use_crop)

    # §5.1/5.2/5.6: load the pre-computed pairwise TM-score cache once, if provided
    # and actually needed (soft InfoNCE or the TM-aux regression).
    tm_cache = None
    if tm_cache_path and (soft_supcon or tm_aux_weight > 0.0):
        if os.path.exists(tm_cache_path):
            try:
                tm_cache = torch.load(tm_cache_path, weights_only=False)
            except TypeError:
                tm_cache = torch.load(tm_cache_path)
            print(f"Loaded TM-score cache: {len(tm_cache)} pairs from {tm_cache_path}")
        else:
            print(f"[!] TM-score cache {tm_cache_path} not found. Run build_tm_cache.py first. "
                  f"Proceeding without it (soft_supcon/cached-aux disabled).")
    if soft_supcon and tm_cache is None:
        print("[!] --soft-supcon requested but no TM cache loaded; falling back to binary SupCon.")
        soft_supcon = False

    train_dataset = PCCDataset(train_files, transform=aug)
    val_dataset = PCCDataset(val_files, transform=aug)

    # Single factory so train and val share every DataLoader knob. To add a new
    # flag (e.g. pin_memory policy, a different collate_fn, timeout) edit here once.
    # Val always uses num_workers=0 and no batch_sampler: parallel workers + a
    # separate sampler on MPS unified memory caused unbounded accumulation and
    # crashes at ~200-300 steps. num_workers=0 is safe; train parallelism is enough.
    # `num_workers` overridable (the --deltafold CUDA preset sets 16 for the 96-thread
    # box); default is cpu_count//2 (the old behaviour).
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // 2)
    print(f"DataLoader: {num_workers} train workers, pin_memory={DEVICE.type == 'cuda'}")

    def _make_loader(dataset, *, batch_sampler=None, workers=num_workers, shuffle=False):
        shared_kwargs = dict(
            collate_fn=moco_collate if is_moco else contrastive_collate,
            pin_memory=DEVICE.type == 'cuda',
            worker_init_fn=worker_init_fn if workers > 0 else None,
        )
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler,
                              num_workers=workers, prefetch_factor=2 if workers > 0 else None,
                              persistent_workers=False, **shared_kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, prefetch_factor=2 if workers > 0 else None,
                          persistent_workers=False, **shared_kwargs)

    train_sampler = None
    # `make_train_loader` is a FACTORY (not a single loader) so the epoch loop can
    # cold-restart workers under memory pressure. Each call reads
    # train_sampler.max_residues live so a shrunk budget takes effect immediately.
    if hard_neg_mining:
        cache_path = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
        keys = extract_batch_keys(train_files, cache_path)
        train_sampler = HardNegativeBatchSampler(keys, batch_size, seed=42, max_residues=max_residues)
        print(f"Hard-negative mining: residue budget {max_residues} (orig; ~{int(max_residues*1.75)} in-forward "
              f"after 2 views), count cap {batch_size}, ~{len(train_sampler)} batches/epoch")
        def make_train_loader():
            return _make_loader(train_dataset, batch_sampler=train_sampler)
    else:
        # Even without hard-negative mining, batch by residue budget (not protein
        # count) so a few large proteins can't overflow MPS memory. Lengths come
        # from the same cached metadata the hard-neg sampler uses.
        cache_path = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
        lengths = [k[0] for k in extract_batch_keys(train_files, cache_path)]
        train_sampler = ResidueBudgetSampler(lengths, batch_size, seed=42, max_residues=max_residues)
        print(f"Residue-budget batching: budget {max_residues} (orig; ~{int(max_residues*1.75)} in-forward "
              f"after 2 views), count cap {batch_size}, ~{len(train_sampler)} batches/epoch")
        def make_train_loader():
            return _make_loader(train_dataset, batch_sampler=train_sampler)
    train_loader = make_train_loader()  # one instance for len()/scheduler bookkeeping
    val_loader = _make_loader(val_dataset, workers=0)

    # Memory governor: caps process RSS at mem_hard_gb via off-schedule reclaim + cold
    # worker restarts + dynamic residue-budget shrink (see MemoryGovernor).
    governor = MemoryGovernor(soft_gb=mem_soft_gb, hard_gb=mem_hard_gb, sampler=train_sampler,
                              cleanup_every=cleanup_every, min_residues=min_residues,
                              no_budget_adapt=no_budget_adapt)
    print(f"Memory governor: soft={mem_soft_gb}GB hard={mem_hard_gb}GB "
          f"min_residues={min_residues} (cold-restarts workers + shrinks budget above hard cap)")
    if mem_hard_gb > 12.0:
        print(f"[!] WARNING: hard cap at {mem_hard_gb}GB is close to 16GB limit; "
              f"recommend --mem-hard-gb 12.0 to avoid swap and crashes")
    
    model_config = {
        'use_positional_encoding': use_positional_encoding,
        'use_residue_features': use_residue_features,
        'use_3di_features': use_3di_features,
    }
    if model_type == 'asymmetric':
        # §5.3/5.4/5.7 architecture fixes are asymmetric-only. Stored in model_config
        # so extract_embeddings.py / diagnostics.py rebuild the exact forward pass;
        # old checkpoints lack these keys and fall back to the original behavior.
        model_config.update({
            'edge_attn_softmax': edge_attn_softmax,
            'dist_bias_gamma': dist_bias_gamma,
            'detach_h3': detach_h3,
            # Distance encoding (§1, Mod 1). Persisted so extract_embeddings.py
            # rebuilds the identical rank1_emb (input dim depends on it).
            'dist_encoding': dist_encoding,
            'rbf_dim': rbf_dim,
        })
        model = AsymmetricTopoNet(scalar_dim=128, **model_config).to(DEVICE)
    elif model_type == 'equivariant':
        # SE(3)/O(3) geometric model (plan_experimentation_v2 §F). Consumes the same
        # rank-dict as `asymmetric` (drop-in in the loop; not path-based like tcpnet).
        # rbf_dim>0 selects RBF distance encoding (0 = sinusoidal). Persisted so
        # extract_embeddings rebuilds the identical architecture.
        from equivariant_topotein import EquivariantTopoNet
        model_config.update({
            'edge_attn_softmax': edge_attn_softmax,
            'dist_bias_gamma': dist_bias_gamma,
            'scalarize': scalarize,          # 'frame' (SE(3)+chiral) or 'norm' (O(3))
            'vector_dim': vector_dim,
            'rbf_dim': (rbf_dim if dist_encoding == 'rbf' else 0),
        })
        if num_layers is not None:
            model_config['num_layers'] = num_layers
        model = EquivariantTopoNet(scalar_dim=128, **model_config).to(DEVICE)
    else:
        # Full Topotein / TCPNet on the PCC (topotein.py), the faithful
        # implementation of the protocol's §Architecture. Consumes the same
        # rank-dict as `asymmetric` / `equivariant` (dict path in the loop; it
        # sets no `is_tcpnet` flag, so it is NOT the old PDB-re-lifting adapter).
        # Every shape-determining knob goes in model_config so the checkpoint
        # records it and extract_embeddings.py rebuilds the identical model.
        model_config.update({
            'scalarize': scalarize,               # 'frame' (edge-centric SE(3)+chiral) | 'norm' (O(3))
            'vector_dim': vector_dim,             # protocol d_v
            'tensor_diagram': tensor_diagram,     # message-passing order / channels (--tensor-diagram)
            'readout': readout,                   # 'node' pooling (default) | 'protein' cell
            'edge_attn_softmax': edge_attn_softmax,
            'rbf_dim': (rbf_dim if dist_encoding == 'rbf' else 0),
        })
        if num_layers is not None:
            model_config['num_layers'] = num_layers
        # scalar_dim is fixed at the protocol's d_s=128; the legacy --emb-dim knob
        # (old adapter width) does not apply to the new encoder.
        model = Topotein(scalar_dim=128, **model_config).to(DEVICE)

    print(f"Mitigations: PE={use_positional_encoding} residue={use_residue_features} "
          f"3di={use_3di_features} | hard_neg_beta={hard_neg_beta} | hard_neg_mining={hard_neg_mining} | "
          f"split={split} | tm_aux_weight={tm_aux_weight} | supervised={supervised} | pad_buckets={pad_buckets}")
    if model_type == 'asymmetric':
        print(f"Arch fixes (§5.3/5.4/5.7): edge_attn_softmax={edge_attn_softmax} "
              f"dist_bias_gamma={dist_bias_gamma} detach_h3={detach_h3}")
        print(f"Distance encoding (§1): {dist_encoding}"
              + (f" (K={rbf_dim})" if dist_encoding == 'rbf' else ""))
    if model_type == 'equivariant':
        print(f"Equivariant (§F): scalarize={scalarize} vector_dim={vector_dim} "
              f"rbf_dim={rbf_dim if dist_encoding=='rbf' else 0}")
    if model_type == 'topotein':
        print(f"Topotein/TCPNet: scalarize={scalarize} vector_dim={vector_dim} "
              f"tensor_diagram={tensor_diagram} readout={readout} "
              f"num_layers={num_layers or 4} rbf_dim={rbf_dim if dist_encoding=='rbf' else 0}")
    print(f"Aug (§5.5): use_crop={use_crop} jitter_sigma={jitter_sigma} | "
          f"soft_supcon (§5.6)={soft_supcon} | tm_cache={'yes' if tm_cache else 'no'}")

    # plan_implementation_3 §2/§3: wrap the encoder in MoCo (momentum key encoder +
    # negative queue + projection head). `model` stays == moco.encoder_q, so the
    # checkpoint / extract_embeddings path (which builds the bare encoder and loads
    # model_state_dict) is unchanged; the optimizer trains encoder_q + proj_q.
    moco = None
    opt_params = list(model.parameters())
    if is_moco:
        moco = MoCo(model, dim=128, K=moco_k, m=moco_m, tau=temperature).to(DEVICE)
        opt_params = [p for p in moco.parameters() if p.requires_grad]

    # Fused AdamW runs all parameter updates in a single kernel dispatch instead of
    # one-per-param, saving ~10ms/step on MPS. Falls back silently if unavailable.
    try:
        optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=1e-5, fused=True)
        print("Optimizer: AdamW (fused)")
    except (TypeError, RuntimeError):
        optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=1e-5)
        print("Optimizer: AdamW (eager)")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = NTXentLoss(temperature=temperature, hard_neg_beta=hard_neg_beta).to(DEVICE)
    
    start_epoch = 0
    best_loss = float('inf')
    best_ari = float('-inf')
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_contrastive_{model_type}_last.pth')

    if os.path.exists(last_ckpt_path):
        print(f"Loading checkpoint {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path, map_location=DEVICE)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"[!] model weights not loaded from checkpoint ({e}); starting from init.")
        # The optimizer param-groups change when the objective changes (e.g. infonce
        # -> moco adds projection-head params), so a mismatch must NOT crash — just
        # start the optimizer fresh. Same for resuming a run with a different setup.
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_ari = checkpoint.get('best_ari', float('-inf'))
        except (ValueError, KeyError) as e:
            print(f"[!] optimizer state not restored ({e}); fresh optimizer, epoch 0.")
    
    print(f"Starting Contrastive Training ({model_type}): {len(train_files)} train, {len(val_files)} val samples.")
    print(f"Logging to {tlog.path}")
    
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # re-vary hard-negative batches each epoch
        if profile_train and epoch == start_epoch: # Profile only the first epoch if enabled
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], # Removed MPS as it might not be available
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1), # Profile 10 active steps
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(CHECKPOINT_DIR, 'profiler_log')),
                with_stack=True,
                profile_memory=True
            ) as prof:
                best_loss, best_ari = _run_training_epoch(model, None, train_loader, val_loader, optimizer, scheduler, criterion, epoch, epochs, 0.0, accum_steps, tlog, True, best_loss, model_type, acc_to_cluster, profiler=prof, tm_aux_weight=tm_aux_weight, supervised=supervised, model_config=model_config, cleanup_every=cleanup_every, pad_buckets=pad_buckets, best_ari=best_ari, tm_cache=tm_cache, soft_supcon=soft_supcon, make_train_loader=make_train_loader, train_sampler=train_sampler, governor=governor, moco=moco)

            print("\n" + "="*30 + " PROFILER RESULTS " + "="*30)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        else:
            best_loss, best_ari = _run_training_epoch(model, None, train_loader, val_loader, optimizer, scheduler, criterion, epoch, epochs, 0.0, accum_steps, tlog, True, best_loss, model_type, acc_to_cluster, tm_aux_weight=tm_aux_weight, supervised=supervised, model_config=model_config, cleanup_every=cleanup_every, pad_buckets=pad_buckets, best_ari=best_ari, tm_cache=tm_cache, soft_supcon=soft_supcon, make_train_loader=make_train_loader, train_sampler=train_sampler, governor=governor, moco=moco)

        # Between-epoch cleanup: the just-finished DataLoader iterator (and its
        # worker/prefetch buffers) is now out of scope here, so collecting it now
        # plus releasing the device cache prevents cross-epoch RAM creep.
        ram_before = _report_memory(0, epoch + 1)
        free_memory()
        ram_after = _report_memory(0, epoch + 1)
        gov_status = governor.end_of_epoch()  # recover residue budget after a clean epoch
        print(f"  [mem] epoch {epoch+1} cleanup: {ram_before:.2f}GB -> {ram_after:.2f}GB footprint | {gov_status}")

    print("Training finished.")

# Refactored common epoch logic into a helper function
def _run_training_epoch(model, head, train_loader, val_loader, optimizer, scheduler, criterion, epoch, epochs, mask_ratio, accum_steps, tlog, is_contrastive, best_loss, model_type, acc_to_cluster=None, profiler=None, tm_aux_weight=0.0, supervised=True, model_config=None, cleanup_every=50, pad_buckets=True, best_ari=float('-inf'), tm_cache=None, soft_supcon=False, make_train_loader=None, train_sampler=None, governor=None, moco=None):
    model.train()
    if moco is not None: moco.train()
    if head: head.train()
    epoch_loss = 0.0
    step_losses = []
    optimizer.zero_grad()
    if governor is None:
        governor = MemoryGovernor(soft_gb=0, hard_gb=0, sampler=train_sampler, cleanup_every=cleanup_every)
    if make_train_loader is None:
        make_train_loader = lambda: train_loader

    # Per-step metrics tracked in memory; flushed once as part of the epoch record.
    step_loss_min = float('inf')
    step_loss_max = float('-inf')

    # Restartable training pass: we process `total_steps` gradient steps, but the
    # DataLoader is (re)built inside the generator below so a memory-pressure event can
    # cold-restart the workers (freeing their leaked buffers) without aborting the epoch.
    # Each rebuild re-seeds the sampler so a restart yields fresh batches rather than
    # re-doing the first ones. Setting `_rs['restart']=True` triggers a rebuild.
    total_steps = len(train_loader)
    _rs = {'restart': False}

    def _restartable_batches():
        restart_pass = 0
        while True:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch * 1000 + restart_pass)
            loader = make_train_loader()
            _rs['restart'] = False
            for b in loader:
                yield b
                if _rs['restart']:
                    break
            del loader
            # Force immediate worker cleanup: MPS worker memory may not release with just del.
            # Explicit multi-step reclaim prevents memory lingering after restart (restart spam root cause).
            gc.collect()
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            governor._reclaim()   # workers are now torn down; reclaim their pool too
            restart_pass += 1
            if not _rs['restart']:
                return            # loader exhausted naturally -> epoch pass complete

    pbar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{epochs}")
    done = 0          # gradient steps actually completed this epoch
    last_std = last_cos = float('nan')   # most recent collapse-health probe
    for batch in _restartable_batches():
        if batch is None:
            if profiler: profiler.step()
            continue
        done += 1

        if moco is not None:
            # MoCo path (plan_impl_3): batch = (feats_v1, feats_v2, paths); run
            # view1 -> f_q, view2 -> f_k + negative queue. No feature jitter/dropout
            # (the substructure views ARE the augmentation).
            feats_q_raw, feats_k_raw, paths = batch
            fq = to_device(feats_q_raw, DEVICE)
            fk = to_device(feats_k_raw, DEVICE)
            if pad_buckets:
                fq, real_B = pad_to_buckets(fq)
                fk, _ = pad_to_buckets(fk)
            else:
                real_B = fq['rank3']['protein_size'].shape[0]
            loss = moco(fq, fk, real_B=real_B)
            if done % 10 == 0:
                with torch.no_grad():
                    zc = moco.embed(fq, real_B=real_B)
                emb_std, mean_cos = collapse_metrics(zc)
                last_std, last_cos = emb_std, mean_cos
                if mean_cos > 0.9 or emb_std < 0.02:
                    print(f"\n[!] Possible collapse at epoch {epoch+1} step {done}: "
                          f"emb_std={emb_std:.4f} mean_cos={mean_cos:.4f}")
                    tlog.event({'t': 'collapse', 'ep': epoch+1, 's': done,
                                'std': round(emb_std, 5), 'cos': round(mean_cos, 5)})
                del zc
            features, model_in, z = fq, fk, None   # placeholders for the end-of-step del
        elif is_contrastive:
            batch_data, paths = batch
            features = to_device(batch_data, DEVICE)

            # Stronger jitter and Feature Dropout to prevent shortcut learning
            if model.training:
                r3 = features['rank3']
                # Apply 10% jitter to scale-sensitive features
                r3['radius_of_gyration'] *= (1 + 0.10 * torch.randn_like(r3['radius_of_gyration']))
                r3['global_shape_descriptors'] *= (1 + 0.05 * torch.randn_like(r3['global_shape_descriptors']))
                
                # Randomly "blind" the model to protein size (Shortcut Dropout)
                # In 20% of batches, set size to a constant to force reliance on topology
                if random.random() < 0.20:
                    r3['protein_size'] = torch.ones_like(r3['protein_size']) * 500.0
                    r3['radius_of_gyration'] = torch.zeros_like(r3['radius_of_gyration'])

            if getattr(model, 'is_tcpnet', False):
                # Equivariant TCPNet rebuilds geometry from the raw PDBs keyed by
                # `paths` (the rank-dict / bucket padding don't apply here).
                model_in = None  # bound so the end-of-step `del` works on this path
                z = model(paths=paths)
                real_B = z.size(0)
            else:
                # Bucket-pad shapes so MPS reuses compiled kernels (no per-step recompile
                # bloat). Real proteins are unchanged; slice the output back to real_B.
                if pad_buckets:
                    model_in, real_B = pad_to_buckets(features)
                else:
                    model_in, real_B = features, features['rank3']['protein_size'].shape[0]
                z = model(model_in)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z = z[:real_B]

            if (soft_supcon or supervised) and acc_to_cluster and is_contrastive:
                from train import extract_accession
                cluster_ids = [acc_to_cluster.get(extract_accession(os.path.basename(p)), p) for p in paths]
                # Convert cluster strings to integer labels for the SupCon mask
                label_map = {cid: i for i, cid in enumerate(set(cluster_ids))}
                labels = torch.tensor([label_map[cid] for cid in cluster_ids], device=DEVICE)
                if soft_supcon and tm_cache is not None:
                    # §5.6: continuous (TM-weighted) supervised contrastive objective.
                    tm_matrix = build_tm_matrix(paths, tm_cache, z.device)
                    loss = soft_supcon_loss(z, labels, tm_matrix, temperature=0.1)
                    del tm_matrix  # Free immediately; loss computation holds copy in autograd graph
                else:
                    loss = supervised_ntxent_loss(z, labels, temperature=0.1,
                                                  hard_neg_beta=criterion.hard_neg_beta)
                del cluster_ids, label_map  # Free CPU-side metadata
            else:
                B = z.size(0) // 2
                loss = criterion(z[:B], z[B:])

            # Optional TM-score regression auxiliary (§5.2 / 3.4). Prefer the cached
            # variant (keyed by view path basenames) when a TM cache is loaded;
            # otherwise fall back to on-the-fly tmtools alignment on unpadded Ca.
            if tm_aux_weight > 0.0:
                if tm_cache is not None:
                    tm_loss = tm_score_aux_loss_cached(z, paths, tm_cache)
                else:
                    tm_loss = tm_score_aux_loss(z, features)
                if tm_loss is not None:
                    loss = loss + tm_aux_weight * tm_loss

            # Collapse detection: check every 10 steps; only log a 'collapse' event
            # on threshold breach, but keep the latest probe for the per-step record.
            if done % 10 == 0:
                emb_std, mean_cos = collapse_metrics(z)
                last_std, last_cos = emb_std, mean_cos
                if mean_cos > 0.9 or emb_std < 0.02:
                    msg = f"emb_std={emb_std:.4f} mean_cos={mean_cos:.4f}"
                    print(f"\n[!] Possible collapse at epoch {epoch+1} step {done}: {msg}")
                    tlog.event({'t': 'collapse', 'ep': epoch+1, 's': done,
                                'std': round(emb_std, 5), 'cos': round(mean_cos, 5)})
        else:
            batch_data, _ = batch
            features = to_device(batch_data, DEVICE)
            r0 = features['rank0']
            n_res = r0['aa'].shape[0]
            if n_res == 0:
                if profiler: profiler.step()
                continue
            num_mask = max(1, int(n_res * mask_ratio))
            mask_indices = random.sample(range(n_res), num_mask)
            orig_3di = r0['3di'][mask_indices].clone()
            target_classes = torch.argmax(orig_3di, dim=1)
            r0['3di'][mask_indices] = 0.0
            r0['aa'][mask_indices] = 0.0
            global_emb, h0 = model(features, return_nodes=True)
            masked_h0 = h0[mask_indices]
            logits = head(masked_h0)
            loss = criterion(logits, target_classes)

        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        lv = loss.item()
        epoch_loss += lv
        step_losses.append(lv)
        if lv < step_loss_min: step_loss_min = lv
        if lv > step_loss_max: step_loss_max = lv

        if done % accum_steps == 0 or done == total_steps:
            if moco is not None:
                params_to_clip = [p for p in moco.parameters() if p.requires_grad and p.grad is not None]
            else:
                params_to_clip = [p for p in model.parameters() if p.grad is not None]
            if head: params_to_clip.extend([p for p in head.parameters() if p.grad is not None])
            
            if params_to_clip:
                if DEVICE.type == 'mps':
                    # torch.linalg.vector_norm in clip_grad_norm_ causes massive CPU sync bottlenecks on MPS.
                    # Using clip_grad_value_ entirely bypasses the global reduction syncs.
                    torch.nn.utils.clip_grad_value_(params_to_clip, clip_value=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=5.0)
                    
            optimizer.step()
            optimizer.zero_grad()

        if done % 10 == 0 or done == total_steps:
            current_loss = step_losses[-1]
            mem_gb = governor.last_fp or _report_memory(done, epoch + 1)
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'fp': f"{mem_gb:.1f}GB"})
            step_losses.clear()

        # Structured per-batch record (loss / lr / footprint / collapse-health) for
        # offline analysis. Buffered in the TrainingLog, flushed with the epoch.
        if STEP_LOG_EVERY and (done % STEP_LOG_EVERY == 0 or done == total_steps):
            tlog.event({'t': 'step', 'ep': epoch + 1, 's': done,
                        'loss': round(lv, 5),
                        'lr': round(optimizer.param_groups[0]['lr'], 8),
                        'fp': round(governor.last_fp or 0.0, 2),
                        'std': round(last_std, 5) if last_std == last_std else None,
                        'cos': round(last_cos, 5) if last_cos == last_cos else None})

        if profiler: profiler.step()

        if is_contrastive:
            del features, model_in, z, batch
        else:
            del features, r0, orig_3di, target_classes, global_emb, h0, masked_h0, logits, batch
        del loss, scaled_loss

        # Memory governance: throttled MPS reclaim, off-schedule reclaim under soft
        # pressure, and a cold worker-restart + residue-budget shrink if RSS still
        # exceeds the hard cap (cheaper than swapping). See MemoryGovernor.
        if governor.after_step(done) == 'restart':
            new_budget = getattr(train_sampler, 'max_residues', 'n/a')
            pbar.write(f"  [mem] footprint {governor.last_fp:.1f}GB > {governor.hard_gb}GB hard cap "
                       f"-> cold-restarting DataLoader workers; residue budget -> {new_budget}")
            tlog.event({'t': 'restart', 'ep': epoch+1, 's': done,
                        'fp': round(governor.last_fp, 2),
                        'budget': getattr(train_sampler, 'max_residues', None)})
            # Safety: if we've already restarted too many times, stop this epoch to avoid crash
            if governor._epoch_restarts > 5:
                pbar.write(f"  [!] ABORT: >5 restarts in this epoch; memory governor exhausted")
                pbar.close()
                print(f"[!] Epoch {epoch+1} aborted after {done} steps due to memory exhaustion")
                break
            _rs['restart'] = True
        pbar.update(1)
    pbar.close()
    train_steps_done = done
    peak_fp = _phys_footprint_gb()

    # Validation Pass
    model.eval()
    if head: head.eval()
    val_loss_epoch = 0.0
    ari_embs, ari_paths = [], []   # collect view-1 embeddings for per-epoch ARI
    # Validation has no residue-budgeted sampler and (previously) no per-step reclaim, so
    # the MPS allocator pool grew across the whole val set and spilled 16GB into swap
    # (observed ~27GB, 44s/it). Throttle a gc+empty_cache the way the training loop does.
    val_cleanup = max(1, cleanup_every // 2) if cleanup_every > 0 else 25
    with torch.no_grad():
        for v_step, v_batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            if v_batch is None: continue
            if moco is not None:
                # update=False: validation must NOT mutate the key encoder or the
                # negative queue. ARI/TM embedding = normalize(h) from f_q (view1).
                vq = to_device(v_batch[0], DEVICE)
                vk = to_device(v_batch[1], DEVICE)
                if pad_buckets:
                    vq, v_realB = pad_to_buckets(vq)
                    vk, _ = pad_to_buckets(vk)
                else:
                    v_realB = vq['rank3']['protein_size'].shape[0]
                val_loss_epoch += moco(vq, vk, real_B=v_realB, update=False).item()
                if acc_to_cluster and v_realB > 0:
                    ari_embs.append(moco.embed(vq, real_B=v_realB).detach().cpu().numpy())
                    ari_paths.extend(v_batch[2][:v_realB])
                del vq, vk, v_batch
            elif is_contrastive:
                v_feat = to_device(v_batch[0], DEVICE)
                if getattr(model, 'is_tcpnet', False):
                    v_in = None  # bound so the end-of-step `del` works on this path
                    vz = model(paths=v_batch[1])
                    v_realB = vz.size(0)
                else:
                    if pad_buckets:
                        v_in, v_realB = pad_to_buckets(v_feat)
                    else:
                        v_in, v_realB = v_feat, v_feat['rank3']['protein_size'].shape[0]
                    vz = model(v_in)
                if vz.dim() == 1: vz = vz.unsqueeze(0)
                vz = vz[:v_realB]
                VB = vz.size(0) // 2
                val_loss_epoch += criterion(vz[:VB], vz[VB:]).item()
                # one embedding (view 1) + cluster path per protein for ARI
                if acc_to_cluster and VB > 0:
                    ari_embs.append(vz[:VB].detach().cpu().numpy())
                    ari_paths.extend(v_batch[1][:VB])
                del v_feat, v_in, vz, v_batch
            else:
                v_feat = to_device(v_batch, DEVICE)
                v_r0 = v_feat['rank0']
                v_n = v_r0['aa'].shape[0]
                if v_n == 0: continue
                v_num_mask = max(1, int(v_n * mask_ratio))
                v_mask_idx = random.sample(range(v_n), v_num_mask)
                v_targets = torch.argmax(v_r0['3di'][v_mask_idx], dim=1)
                v_r0['3di'][v_mask_idx] = 0.0
                v_r0['aa'][v_mask_idx] = 0.0
                _, v_h0 = model(v_feat, return_nodes=True)
                v_logits = head(v_h0[v_mask_idx])
                val_loss_epoch += criterion(v_logits, v_targets).item()
                del v_feat, v_r0, v_targets, v_h0, v_logits, v_batch

            # Periodic reclaim INSIDE the val loop — this is the fix for the 27GB swap blowup.
            # Plus an off-schedule reclaim whenever RSS crosses the soft cap, so validation
            # (num_workers=0, so nothing to cold-restart) also honours the memory ceiling.
            if DEVICE.type == 'mps' and (v_step + 1) % val_cleanup == 0:
                gc.collect()
                torch.mps.empty_cache()
            elif governor.soft_gb and _phys_footprint_gb() > governor.soft_gb:
                governor._reclaim()

        if DEVICE.type == 'mps':
            gc.collect()
            torch.mps.empty_cache()

    avg_train_loss = epoch_loss / max(1, train_steps_done)
    avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0.0
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Per-epoch ARI: cluster the val embeddings and score against cluster.tsv labels.
    # Rising ARI = the model is learning fold structure (watch this during training).
    ari_val = float('nan')
    eval_metrics = {}
    if is_contrastive and acc_to_cluster and ari_embs:
        import numpy as np
        from train import extract_accession
        embs = np.concatenate(ari_embs, axis=0)
        labels = [acc_to_cluster.get(extract_accession(os.path.basename(p)), p) for p in ari_paths]
        # Average over multiple k-means seeds to reduce init variance.
        ari_runs = [compute_ari(embs, labels, seed=epoch * 20 + s)[0] for s in range(15)]
        ari_runs = [a for a in ari_runs if not (a != a)]  # drop NaNs
        ari_val = sum(ari_runs) / len(ari_runs) if ari_runs else float('nan')
        _, n_eval, n_clusters = compute_ari(embs, labels, seed=epoch)
        print(f"  Cluster ARI (val, 5-seed avg): {ari_val:.4f}  "
              f"[{n_eval} proteins in {n_clusters} multi-member clusters]")

        # Deeper per-epoch health check: HDBSCAN cluster + TM-rho / homology recall.
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'analysis'))
            import epoch_eval
            ev_ids = [os.path.basename(p).replace('.pt', '') for p in ari_paths]
            eval_metrics = epoch_eval.evaluate(embs, ev_ids, labels)
            print(f"  Epoch eval: {epoch_eval.format_line(eval_metrics)}")
        except Exception as e:
            print(f"  [epoch_eval skipped: {e}]")
        del embs, labels  # Free large ARI embedding arrays after evaluation

    # Track best by val loss AND by ARI separately, since changing hard_neg_beta
    # shifts the loss scale and makes val-loss-only checkpointing unreliable.
    current_ari = ari_val if (is_contrastive and acc_to_cluster and ari_embs) else float('nan')
    is_best_loss = avg_val_loss < best_loss
    if is_best_loss:
        best_loss = avg_val_loss
    is_best_ari = (not (current_ari != current_ari)) and current_ari > best_ari
    if is_best_ari:
        best_ari = current_ari
    is_best = is_best_loss or is_best_ari

    # Define checkpoint_data outside the is_best block
    checkpoint_data = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'best_loss': best_loss,
        'ari': current_ari,
        'best_ari': best_ari,
        'model_config': model_config or {},
    }

    # Save last checkpoint (rolling pointer to the most recent epoch).
    last_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_contrastive_{model_type}_last.pth')
    torch.save(checkpoint_data, last_path)

    # Save a per-epoch checkpoint that is never overwritten, so every epoch's
    # weights stay available (e.g. for the §3 epoch-wise evaluation / picking a
    # checkpoint after the fact rather than trusting only best-loss/ARI).
    epoch_path = os.path.join(
        CHECKPOINT_DIR, f'checkpoint_contrastive_{model_type}_epoch{epoch + 1:03d}.pth')
    torch.save(checkpoint_data, epoch_path)
    print(f"Saved epoch checkpoint -> {os.path.basename(epoch_path)}")

    # Save best checkpoint separately (triggers on best val loss OR best ARI)
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_contrastive_{model_type}_best.pth')
        torch.save(checkpoint_data, best_path)
        reason = []
        if is_best_loss: reason.append(f"loss={best_loss:.4f}")
        if is_best_ari:  reason.append(f"ARI={best_ari:.4f}")
        print(f"New best checkpoint saved ({', '.join(reason)})")

    # Flush the consolidated epoch record to training_log.jsonl.gz.
    # One compact JSON line per epoch; any buffered events (restarts, collapses)
    # are appended in the same gzip stream so the file stays small.
    def _safe(v): return round(v, 5) if isinstance(v, float) and v == v else None
    ev_compact = {}
    if eval_metrics:
        ev_compact = {k: _safe(eval_metrics[k]) if isinstance(eval_metrics[k], float)
                      else eval_metrics[k]
                      for k in ('hdbscan_ari','hdbscan_nmi','tm_rho','tm_recall',
                                'n_clusters','singleton_frac',
                                # §6.1 health gate + Mod 3/4 (plan metrics)
                                'tm_alignment','effective_rank','emb_std','mean_cos','uniformity',
                                'homogeneity','completeness','v_measure','fowlkes_mallows',
                                'fragmentation','fusion','perm_ari') if k in eval_metrics}
    epoch_rec = {
        't': 'epoch', 'ep': epoch + 1,
        'steps': train_steps_done,
        'loss': round(avg_train_loss, 5),
        'vloss': round(avg_val_loss, 5),
        'ari': _safe(ari_val),
        'fp': round(peak_fp, 2),
        'restarts': governor._epoch_restarts,          # already reset by end_of_epoch, read before
        'budget': getattr(train_sampler, 'max_residues', None),
        'loss_range': [round(step_loss_min, 4), round(step_loss_max, 4)],
    }
    if ev_compact:
        epoch_rec['eval'] = ev_compact
    tlog.flush_epoch(epoch_rec)

    # Explicitly clear checkpoint objects and release the device cache
    del checkpoint_data
    free_memory()

    return best_loss, best_ari