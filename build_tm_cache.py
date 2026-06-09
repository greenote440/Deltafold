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


def build_clusters(proc_dir, cluster_tsv):
    """Group the .pt files actually present in proc_dir by their Foldseek cluster
    representative. Returns (clusters: {rep -> [basename,...]}, basenames: [all])."""
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


def main():
    ap = argparse.ArgumentParser(description="Pre-compute pairwise TM-score cache (§5.1).")
    ap.add_argument('--proc-dir', default=PROC_DIR)
    ap.add_argument('--cluster-tsv', default=CLUSTER_TSV)
    ap.add_argument('--out', default=os.path.join(CHECKPOINT_DIR, 'tm_score_cache.pt'))
    ap.add_argument('--cross-samples', type=int, default=0,
                    help="Number of random cross-cluster pairs to also align (informative negatives).")
    ap.add_argument('--limit-clusters', type=int, default=None,
                    help="Only process the first N multi-member clusters (smoke test).")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    try:
        import tmtools  # noqa: F401
    except ImportError:
        raise SystemExit("tmtools not found. Activate ml_env: "
                         "source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ml_env")

    random.seed(args.seed)
    clusters, all_basenames = build_clusters(args.proc_dir, args.cluster_tsv)
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

    cache = {}

    # --- within-cluster pairs ---
    for rep in tqdm(multi_reps, desc="Within-cluster TM"):
        members = [m for m in multi[rep] if m in structs]
        for a, b in combinations(members, 2):
            tm = tm_align_pair(structs, a, b)
            if tm is not None:
                cache[(a, b)] = tm

    # --- optional cross-cluster sample ---
    if args.cross_samples > 0 and len(all_basenames) > 1:
        bn_to_rep = {bn: rep for rep, mem in clusters.items() for bn in mem}
        pool = [b for b in all_basenames if b in structs]
        added, attempts, max_attempts = 0, 0, args.cross_samples * 20
        pbar = tqdm(total=args.cross_samples, desc="Cross-cluster TM")
        while added < args.cross_samples and attempts < max_attempts:
            attempts += 1
            a, b = random.sample(pool, 2)
            if bn_to_rep.get(a) == bn_to_rep.get(b):
                continue  # same cluster — already covered above
            if (a, b) in cache or (b, a) in cache:
                continue
            tm = tm_align_pair(structs, a, b)
            if tm is not None:
                cache[(a, b)] = tm
                added += 1
                pbar.update(1)
        pbar.close()
        print(f"Added {added} cross-cluster pairs ({attempts} attempts).")

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
