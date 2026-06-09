"""
Phase 3.2 / 3.3 - Compare the model's clusters against the Foldseek clusters.

Computes ARI and NMI between the model clustering (clusters/model_clusters.tsv) and
the Foldseek assignments (data/cluster.tsv), restricted to proteins that are in a
MULTI-MEMBER cluster in both. Then logs:
  - merges: one model cluster spanning >=2 Foldseek clusters (Foldseek may have
            over-split a structural family);
  - splits: one Foldseek cluster spread over >=2 model clusters (a Foldseek cluster
            may hide structurally distinct sub-families);
  - the model singleton ratio vs. the paper's ~2/3 "structural dark matter".

    python compare_clusters.py
    python compare_clusters.py --clusters clusters/model_clusters_agglomerative.tsv
"""
import os
import argparse
from collections import defaultdict

import numpy as np

import cluster_common as cc


def restrict_to_mutual_multi(model, fs_rep):
    """Keep only proteins that are multi-member in BOTH partitions; return aligned
    integer label arrays plus the surviving id list."""
    # model multi-member ids = non-negative
    fs_sizes = defaultdict(int)
    for pid in model:
        if pid in fs_rep:
            fs_sizes[fs_rep[pid]] += 1
    keep = [pid for pid, lab in model.items()
            if lab >= 0 and pid in fs_rep and fs_sizes[fs_rep[pid]] >= 2]
    keep.sort()
    model_lab = np.array([model[p] for p in keep])
    fs_codes = {r: i for i, r in enumerate(sorted({fs_rep[p] for p in keep}))}
    fs_lab = np.array([fs_codes[fs_rep[p]] for p in keep])
    return keep, model_lab, fs_lab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clusters', default=os.path.join(cc.OUT_DIR, 'model_clusters.tsv'))
    ap.add_argument('--top', type=int, default=15, help="How many merge/split cases to print.")
    args = ap.parse_args()

    model = cc.load_cluster_tsv(args.clusters)
    fs_rep = cc.load_foldseek_clusters()
    meta = cc.build_metadata(list(model.keys()))
    n = len(model)
    print(f"model clusters file : {args.clusters}  ({n} proteins)")

    # --- 3.3 singleton ratio ---------------------------------------------------
    singletons = sum(1 for lab in model.values() if lab < 0)
    fs_sizes_all = defaultdict(int)
    for pid in model:
        if pid in fs_rep:
            fs_sizes_all[fs_rep[pid]] += 1
    fs_singletons = sum(1 for pid in model if fs_sizes_all.get(fs_rep.get(pid), 0) <= 1)
    print("\n=== 3.3 Singleton ratio (structural dark matter) ===")
    print(f"  model singletons   : {singletons}/{n}  ({singletons / n:.1%})")
    print(f"  foldseek singletons: {fs_singletons}/{n}  ({fs_singletons / n:.1%})")
    print(f"  paper reference    : ~66% of viral proteins have no detectable homologue")
    print("  (calibration check, not pass/fail: far below => over-merging; far above => under-sensitive)")

    # --- 3.2 ARI / NMI on mutually multi-member proteins -----------------------
    keep, model_lab, fs_lab = restrict_to_mutual_multi(model, fs_rep)
    print("\n=== 3.2 Agreement with Foldseek (mutually multi-member) ===")
    print(f"  proteins compared        : {len(keep)}")
    print(f"  model clusters in subset : {len(set(model_lab))}")
    print(f"  foldseek clusters in sub : {len(set(fs_lab))}")
    if len(keep) >= 2:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(fs_lab, model_lab)
        nmi = normalized_mutual_info_score(fs_lab, model_lab)
        print(f"  ARI (model vs foldseek)  : {ari:+.4f}   (training reached ~0.40)")
        print(f"  NMI (model vs foldseek)  : {nmi:+.4f}")
    else:
        print("  too few mutually-multi-member proteins to score.")

    # --- merge / split discovery ----------------------------------------------
    # Build maps over the FULL model partition (multi-member model clusters only).
    fs_of = {p: fs_rep.get(p) for p in model}
    model_members = defaultdict(list)
    for p, lab in model.items():
        if lab >= 0:
            model_members[lab].append(p)

    # MERGES: model cluster -> set of distinct foldseek reps among its members
    print("\n=== 3.2 Merges (model unites >=2 Foldseek clusters) ===")
    merges = []
    for mlab, members in model_members.items():
        fs_set = {fs_of[p] for p in members if fs_of.get(p)}
        # only count foldseek clusters that are themselves multi-member
        fs_multi = {r for r in fs_set if fs_sizes_all.get(r, 0) >= 2}
        if len(fs_multi) >= 2:
            merges.append((len(members), len(fs_multi), mlab, members, fs_multi))
    merges.sort(reverse=True)
    print(f"  {len(merges)} model clusters merge >=2 multi-member Foldseek clusters")
    for size, nfs, mlab, members, fs_multi in merges[:args.top]:
        fams = sorted({meta[p]['family'] for p in members})
        print(f"   model#{mlab:<5} size={size:<4} unites {nfs} foldseek clusters | families: {', '.join(fams)}")

    # SPLITS: foldseek cluster -> set of distinct model clusters among its members
    print("\n=== 3.2 Splits (model divides one Foldseek cluster) ===")
    fs_members = defaultdict(list)
    for p in model:
        r = fs_of.get(p)
        if r is not None and fs_sizes_all.get(r, 0) >= 2:
            fs_members[r].append(p)
    splits = []
    for r, members in fs_members.items():
        mlabs = {model[p] for p in members if model[p] >= 0}
        if len(mlabs) >= 2:
            splits.append((len(members), len(mlabs), r, members, mlabs))
    splits.sort(reverse=True)
    print(f"  {len(splits)} multi-member Foldseek clusters are split across >=2 model clusters")
    for size, nm, r, members, mlabs in splits[:args.top]:
        fams = sorted({meta[p]['family'] for p in members})
        print(f"   foldseek[{cc.parse_protein_id(r)[0][:30]:<30}] size={size:<4} -> {nm} model clusters | families: {', '.join(fams)}")


if __name__ == '__main__':
    main()
