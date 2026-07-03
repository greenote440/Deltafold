"""
Pre-compute pairwise TM-scores for the contrastive training set
(TM-score analysis §5.1 — "Enables Everything Else").

Builds a sparse dict {(pt_basename_a, pt_basename_b): tm_score} for:
  * every within-cluster pair (the signal SupCon erases by collapsing a cluster's
    0.40-0.99 TM range to a single "same" label), and
  * an optional random sample of cross-cluster pairs (useful as informative
    negatives / near-miss pairs for the TM-aux regression loss).

The cache removes the ~0.5s/pair on-the-fly tmtools alignment bottleneck that
forced the TM-aux loss to stay disabled, enabling §5.2 (cached TM-aux loss) and
§5.6 (soft InfoNCE). Keys are .pt file BASENAMES so they line up with the
`paths` list threaded through training (`os.path.basename(path)`).

Run inside the ml_env conda env (tmtools lives there):
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ml_env
    python build_tm_cache.py                       # full within-cluster build
    python build_tm_cache.py --cross-samples 5000  # + sampled cross-cluster pairs
    python build_tm_cache.py --limit-clusters 5 --out /tmp/tm_cache_smoke.pt  # smoke test
"""
import os
import glob
import re
import argparse
import random
from itertools import combinations

import numpy as np
import multiprocessing as mp

import torch
from tqdm import tqdm

# Defaults mirror train.py / evaluate_correlation.py so the cache lines up with training.
PROC_DIR = './data/hoan_processed'
CLUSTER_TSV = './data/cluster.tsv'
CHECKPOINT_DIR = './checkpoints'
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO"


def extract_accession(text):
    """Accession (e.g. YP_010085741) from a filename. Mirrors train.extract_accession."""
    match = re.search(r'([A-Z]{1,2}_[0-9]{5,10})', text)
    return match.group(1) if match else text


def load_struct(pt_path):
    """C-alpha coordinates (float32) and decoded sequence from a processed PCC file.
    Returns (coords, seq) or (None, None) if too small / unreadable."""
    try:
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(pt_path, map_location='cpu')
        ca = data['rank0']['ca_coords'].cpu().numpy().astype(np.float32)
        if ca.shape[0] < 3:
            return None, None
        aa = data['rank0']['aa'].cpu().numpy().argmax(axis=1)
        seq = "".join(AA_ALPHABET[i] if i < len(AA_ALPHABET) else "X" for i in aa)
        return ca, seq
    except Exception:
        return None, None


def build_clusters(proc_dir, cluster_tsv, restrict=None):
    """Group the .pt files actually present in proc_dir by their Foldseek cluster
    representative. Returns (clusters: {rep -> [basename,...]}, basenames: [all]).

    ``restrict`` (optional set of basenames) limits the cache to those files —
    use it to scope the (expensive) TM-align build to the corrected sub-base
    instead of the full 67k set, whose mega-clusters yield millions of pairs."""
    acc_to_cluster = {}
    if os.path.exists(cluster_tsv):
        with open(cluster_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    acc_to_cluster[extract_accession(parts[1])] = extract_accession(parts[0])

    pt_files = sorted(glob.glob(os.path.join(proc_dir, '*.pt')))
    clusters = {}
    basenames = []
    for f in pt_files:
        bn = os.path.basename(f)
        if restrict is not None and bn not in restrict:
            continue
        basenames.append(bn)
        acc = extract_accession(bn)
        rep = acc_to_cluster.get(acc, acc)  # singletons cluster with themselves
        clusters.setdefault(rep, []).append(bn)
    return clusters, basenames


def tm_align_pair(structs, a, b):
    """Symmetric TM-score (max over both normalisations), or None on failure."""
    import tmtools
    ca, sa = structs.get(a, (None, None))
    cb, sb = structs.get(b, (None, None))
    if ca is None or cb is None:
        return None
    try:
        res = tmtools.tm_align(ca, cb, sa, sb)
        return float(max(res.tm_norm_chain1, res.tm_norm_chain2))
    except Exception:
        return None


# --- Parallel alignment -----------------------------------------------------
# TM-align is single-threaded per pair, so a serial loop pins one core and
# leaves the rest idle. The pairs are independent, so fan them out over a process
# pool. Workers hold the (read-only) structure table in a module global set once
# by the initializer, so it isn't re-pickled per task.
_STRUCTS = None


def _init_tm_worker(structs):
    global _STRUCTS
    _STRUCTS = structs


def _align_pair_mp(pair):
    a, b = pair
    return pair, tm_align_pair(_STRUCTS, a, b)


def main():
    ap = argparse.ArgumentParser(description="Pre-compute pairwise TM-score cache (§5.1).")
    ap.add_argument('--proc-dir', default=PROC_DIR)
    ap.add_argument('--cluster-tsv', default=CLUSTER_TSV)
    ap.add_argument('--out', default=os.path.join(CHECKPOINT_DIR, 'tm_score_cache.pt'))
    ap.add_argument('--cross-samples', type=int, default=0,
                    help="Number of random cross-cluster pairs to also align (informative negatives).")
    ap.add_argument('--limit-clusters', type=int, default=None,
                    help="Only process the first N multi-member clusters (smoke test).")
    ap.add_argument('--file-list', nargs='*', default=None,
                    help="One or more manifest files (e.g. the corrected sub-base "
                         "data/subbase_corrected_{train,val}.txt). Restricts the cache to those "
                         "structures so the build is scoped/cheap instead of the full 67k set.")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0,
                    help="Parallel TM-align workers (0 = all CPU cores).")
    ap.add_argument('--max-pairs-per-cluster', type=int, default=0,
                    help="Cap within-cluster pairs sampled per cluster (0 = all C(n,2)). "
                         "Bounds the quadratic blow-up of large clusters; e.g. 30 keeps the "
                         "cache sparse-but-representative for TM-rho/recall eval.")
    args = ap.parse_args()

    try:
        import tmtools  # noqa: F401
    except ImportError:
        raise SystemExit("tmtools not found. Activate ml_env: "
                         "source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ml_env")

    random.seed(args.seed)
    restrict = None
    if args.file_list:
        restrict = set()
        for fl in args.file_list:
            with open(fl) as fh:
                restrict.update(os.path.basename(l.strip()) for l in fh if l.strip())
        print(f"Restricting to {len(restrict)} structures from {len(args.file_list)} manifest(s).")
    clusters, all_basenames = build_clusters(args.proc_dir, args.cluster_tsv, restrict=restrict)
    multi = {rep: mem for rep, mem in clusters.items() if len(mem) >= 2}
    multi_reps = sorted(multi.keys())
    if args.limit_clusters is not None:
        multi_reps = multi_reps[:args.limit_clusters]

    n_within = sum(len(multi[r]) * (len(multi[r]) - 1) // 2 for r in multi_reps)
    print(f"{len(all_basenames)} structures | {len(clusters)} clusters "
          f"({len(multi)} multi-member) | {n_within} within-cluster pairs"
          + (f" | +{args.cross_samples} cross-cluster samples" if args.cross_samples else ""))

    # Pre-load only the structures we need (members of the clusters we process,
    # plus all basenames if cross-sampling).
    needed = set()
    for r in multi_reps:
        needed.update(multi[r])
    if args.cross_samples > 0:
        needed.update(all_basenames)

    structs = {}
    for bn in tqdm(sorted(needed), desc="Loading structures"):
        ca, seq = load_struct(os.path.join(args.proc_dir, bn))
        if ca is not None:
            structs[bn] = (ca, seq)

    # Resume support: reuse any pairs already in the output file so a re-run (or a
    # Ctrl-C'd run) doesn't recompute them.
    cache = {}
    if os.path.exists(args.out):
        try:
            cache = torch.load(args.out, weights_only=False)
            print(f"Resuming: {len(cache)} pairs already cached in {args.out}.")
        except Exception:
            cache = {}

    def _is_cached(a, b):
        return (a, b) in cache or (b, a) in cache

    # Build the full pair list up front, then align in parallel (see _align_pair_mp).
    # Cap within-cluster pairs per cluster so a few huge clusters don't dominate
    # (quadratic): randomly sample up to max-pairs-per-cluster of each cluster's pairs.
    pairs = []
    for rep in multi_reps:
        members = [m for m in multi[rep] if m in structs]
        cps = list(combinations(members, 2))
        if args.max_pairs_per_cluster and len(cps) > args.max_pairs_per_cluster:
            cps = random.sample(cps, args.max_pairs_per_cluster)
        pairs.extend(cps)

    # --- optional cross-cluster sample (informative negatives) ---
    if args.cross_samples > 0 and len(all_basenames) > 1:
        bn_to_rep = {bn: rep for rep, mem in clusters.items() for bn in mem}
        pool_bns = [b for b in all_basenames if b in structs]
        seen, added, attempts, max_attempts = set(), 0, 0, args.cross_samples * 50
        while added < args.cross_samples and attempts < max_attempts:
            attempts += 1
            a, b = random.sample(pool_bns, 2)
            if bn_to_rep.get(a) == bn_to_rep.get(b):
                continue  # same cluster — already a within-cluster pair
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key); pairs.append((a, b)); added += 1
        print(f"Sampled {added} cross-cluster pairs ({attempts} attempts).")

    # Drop pairs already cached (resume), then align the rest in parallel.
    pairs = [(a, b) for (a, b) in pairs if not _is_cached(a, b)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_workers = args.workers or os.cpu_count() or 1
    print(f"Aligning {len(pairs)} pairs across {n_workers} workers"
          + (f" (cap {args.max_pairs_per_cluster}/cluster)" if args.max_pairs_per_cluster else "")
          + "...")
    done = 0
    with mp.Pool(n_workers, initializer=_init_tm_worker, initargs=(structs,)) as pool:
        for pair, tm in tqdm(pool.imap_unordered(_align_pair_mp, pairs, chunksize=32),
                             total=len(pairs), desc="TM-align"):
            if tm is not None:
                cache[pair] = tm
            done += 1
            if done % 5000 == 0:        # periodic checkpoint -> resumable on Ctrl-C
                torch.save(cache, args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)

    if cache:
        vals = np.array(list(cache.values()))
        print(f"\nSaved {len(cache)} pairs -> {args.out}")
        print(f"TM-score stats: min={vals.min():.3f} mean={vals.mean():.3f} "
              f"max={vals.max():.3f} | >0.5: {(vals > 0.5).mean()*100:.1f}% "
              f"| 0.3-0.45: {((vals >= 0.3) & (vals <= 0.45)).mean()*100:.1f}%")
    else:
        print("No pairs computed — check proc-dir / cluster-tsv.")


if __name__ == "__main__":
    main()
