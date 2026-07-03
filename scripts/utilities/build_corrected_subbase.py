"""
Build a prototyping sub-base by selecting whole Nomburg clusters.

Strategy (new, replacing proportional proportional-sampling approach):
  - No deduplication, no pLDDT filtering.
  - Load Nomburg merged clusters (merged_clusters.tax.tsv).
  - Randomly select --n-clusters clusters from the pool.
  - Take ALL members of each selected cluster that have a .pt file
    in --proc-dir (no per-cluster downsampling).
  - Cold cluster-aware split: each selected cluster goes entirely to
    train OR val, so no protein from the same structural cluster can
    appear on both sides.
  - No-leakage is guaranteed by construction (whole clusters), but
    verified explicitly before writing.

Because we take whole clusters, the model sees the full intra-cluster
diversity during training and the full inter-cluster contrast at evaluation
time, both against the same 18k Nomburg ground truth.

Outputs (--out-prefix, default data/subbase_corrected):
  <prefix>_train.txt / <prefix>_val.txt  — .pt path lists
  <prefix>_stats.json                    — provenance + size histograms

Usage
-----
    python scripts/utilities/build_corrected_subbase.py --n-clusters 500
    python scripts/utilities/build_corrected_subbase.py \\
        --n-clusters 800 --min-cluster-size 3 --seed 0
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import deltafold_paths

NOMBURG_TSV = "./code_and_intermediate_data/intermediate_data/merged_clusters.tax.tsv"


def strip_ext(name):
    return re.sub(r'\.(pdb|pt)$', '', os.path.basename(name))


def load_nomburg_clusters(path):
    """Returns {cluster_id: [member_name, ...]} from merged_clusters.tax.tsv.
    Columns (0-indexed): cluster_ID, cluster_rep, subcluster_rep, cluster_member, ...
    First two rows are headers and are skipped.
    """
    clusters = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 4:
                continue
            cluster_id = cols[0]
            member = strip_ext(cols[3])
            clusters[cluster_id].append(member)
    return dict(clusters)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deltafold", action="store_true",
                    help="Use the box data root (/data/pnardi) for --proc-dir/--out-prefix "
                         "defaults; already applied at import. Equivalent to DELTAFOLD_DATA_DIR=/data/pnardi.")
    ap.add_argument("--proc-dir", default=deltafold_paths.PROC_DIR,
                    help="Directory containing featurised .pt files "
                         "(default follows the data root: ./data or /data/pnardi under --deltafold).")
    ap.add_argument("--nomburg-tsv", default=NOMBURG_TSV,
                    help="Path to merged_clusters.tax.tsv from the Nomburg study.")
    ap.add_argument("--out-prefix", default=deltafold_paths.SUBBASE_PREFIX,
                    help="Output path prefix for _train.txt, _val.txt, _stats.json.")
    ap.add_argument("--n-clusters", type=int, default=500,
                    help="Number of Nomburg clusters to select (default 500). "
                         "Each selected cluster is taken whole.")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Fraction of selected proteins assigned to val (default 0.2). "
                         "Clusters are assigned whole, so the actual ratio may differ "
                         "slightly from the target.")
    ap.add_argument("--min-cluster-size", type=int, default=1,
                    help="Only consider clusters with at least this many members in "
                         "proc-dir (default 1, includes singletons). Use 2 to restrict "
                         "to clusters that contribute at least one contrastive pair.")
    ap.add_argument("--max-cluster-size", type=int, default=0,
                    help="Exclude clusters with more than this many members in proc-dir "
                         "(0 = no cap). Useful to avoid one mega-cluster dominating.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1. Index available .pt files (name -> path) ---------------------------------
    pt_files = sorted(glob.glob(os.path.join(args.proc_dir, "*.pt")))
    if not pt_files:
        raise SystemExit(f"No .pt files in {args.proc_dir}")
    name_to_path = {strip_ext(p): p for p in pt_files}
    print(f"Available .pt files: {len(name_to_path)}")

    # 2. Load Nomburg clusters and intersect with available .pt files --------------
    all_clusters = load_nomburg_clusters(args.nomburg_tsv)
    print(f"Nomburg clusters loaded: {len(all_clusters)} "
          f"({sum(len(v) for v in all_clusters.values())} members total)")

    # Keep only members that have a .pt file; drop clusters that become too
    # small/large after the intersection.
    eligible = {}
    for cid, members in all_clusters.items():
        present = [m for m in members if m in name_to_path]
        n = len(present)
        if n < args.min_cluster_size:
            continue
        if args.max_cluster_size > 0 and n > args.max_cluster_size:
            continue
        eligible[cid] = present

    print(f"Eligible clusters (min={args.min_cluster_size}"
          f"{f', max={args.max_cluster_size}' if args.max_cluster_size else ''}"
          f"): {len(eligible)} "
          f"({sum(len(v) for v in eligible.values())} proteins)")

    if args.n_clusters > len(eligible):
        print(f"  Warning: requested {args.n_clusters} clusters but only "
              f"{len(eligible)} are eligible; using all of them.")
        args.n_clusters = len(eligible)

    # 3. Random cluster selection -------------------------------------------------
    selected_ids = rng.sample(sorted(eligible.keys()), args.n_clusters)
    selected = {cid: eligible[cid] for cid in selected_ids}
    total_proteins = sum(len(v) for v in selected.values())
    print(f"Selected {len(selected)} clusters → {total_proteins} proteins")

    # 4. Cold cluster-aware train/val split ---------------------------------------
    #    Shuffle clusters, assign to val until val_ratio of proteins is reserved,
    #    then assign all remaining clusters to train.
    cluster_list = list(selected.items())
    rng.shuffle(cluster_list)

    val_target = total_proteins * args.val_ratio
    val_clusters, train_clusters = [], []
    val_count = 0
    for cid, members in cluster_list:
        if val_count < val_target:
            val_clusters.append((cid, members))
            val_count += len(members)
        else:
            train_clusters.append((cid, members))

    train_files = [name_to_path[m] for _, members in train_clusters for m in members]
    val_files   = [name_to_path[m] for _, members in val_clusters   for m in members]

    # 5. No-leakage verification --------------------------------------------------
    train_cluster_ids = {cid for cid, _ in train_clusters}
    val_cluster_ids   = {cid for cid, _ in val_clusters}
    shared_clusters = train_cluster_ids & val_cluster_ids
    assert not shared_clusters, \
        f"COLD-SPLIT VIOLATION: {len(shared_clusters)} clusters span train/val"

    train_names = {strip_ext(p) for p in train_files}
    val_names   = {strip_ext(p) for p in val_files}
    shared_proteins = train_names & val_names
    assert not shared_proteins, \
        f"PROTEIN-LEAK: {len(shared_proteins)} proteins appear in both train and val"

    print(f"Leak-free verified: 0 shared clusters, 0 shared proteins train↔val.")
    actual_val_ratio = len(val_files) / max(1, len(train_files) + len(val_files))
    print(f"Split: train={len(train_files)} proteins / {len(train_clusters)} clusters | "
          f"val={len(val_files)} proteins / {len(val_clusters)} clusters "
          f"(actual val ratio: {actual_val_ratio:.3f})")

    # 6. Write manifests ----------------------------------------------------------
    def write_list(suffix, items):
        path = f"{args.out_prefix}_{suffix}.txt"
        with open(path, "w") as fh:
            fh.write("\n".join(items) + ("\n" if items else ""))
        return path

    p_train = write_list("train", sorted(train_files))
    p_val   = write_list("val",   sorted(val_files))

    # 7. Stats --------------------------------------------------------------------
    train_sizes = [len(members) for _, members in train_clusters]
    val_sizes   = [len(members) for _, members in val_clusters]
    all_sizes   = train_sizes + val_sizes

    stats = {
        "seed": args.seed,
        "n_clusters_requested": args.n_clusters,
        "n_clusters_selected": len(selected),
        "val_ratio_target": args.val_ratio,
        "val_ratio_actual": round(actual_val_ratio, 4),
        "min_cluster_size": args.min_cluster_size,
        "max_cluster_size": args.max_cluster_size,
        "sampling": "whole-cluster selection from Nomburg merged clusters; no dedup, no pLDDT filter",
        "n_available_pt": len(name_to_path),
        "n_nomburg_clusters_total": len(all_clusters),
        "n_eligible_clusters": len(eligible),
        "n_train_proteins": len(train_files),
        "n_val_proteins": len(val_files),
        "n_train_clusters": len(train_clusters),
        "n_val_clusters": len(val_clusters),
        "cold_split_verified": True,
        "protein_leak_verified": True,
        "cluster_size_hist_selected": dict(sorted(Counter(all_sizes).items())),
        "cluster_size_mean": round(sum(all_sizes) / max(1, len(all_sizes)), 2),
        "cluster_size_median": sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0,
        "cluster_size_max": max(all_sizes) if all_sizes else 0,
    }
    stats_path = f"{args.out_prefix}_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nSub-base written:")
    print(f"  train : {len(train_files):5d} proteins / {len(train_clusters):4d} clusters → {p_train}")
    print(f"  val   : {len(val_files):5d} proteins / {len(val_clusters):4d} clusters → {p_val}")
    print(f"  stats → {stats_path}")
    print(f"  cluster sizes — mean: {stats['cluster_size_mean']:.1f}  "
          f"median: {stats['cluster_size_median']}  max: {stats['cluster_size_max']}")


if __name__ == "__main__":
    main()
