"""
Phase 3.1 - Unsupervised clustering of the learned embedding space.

Clusters the embeddings using the learned COSINE distance metric, without using the
Foldseek labels as input. HDBSCAN is the primary method (no K, native singletons);
average-linkage agglomerative clustering at a fixed cosine-distance threshold is run
as a cross-check.

SCALE: the original (and calibrated) path builds a dense (N,N) cosine-distance matrix
and runs HDBSCAN(metric='precomputed') + an agglomerative cross-check. That matrix is
~N^2*8 bytes (67k -> ~36GB), infeasible at full scale. So:
  - N <= --max-dense  : exact original path (precomputed cosine HDBSCAN + agglomerative).
                        Reproduces the downsampled decision-gate results bit-for-bit.
  - N >  --max-dense  : HDBSCAN on the EUCLIDEAN metric over the L2-normalised vectors
                        via space-trees (O(N) memory). Embeddings are unit-norm so
                        euclidean is a monotonic transform of cosine -> the MST/hierarchy
                        is identical; only HDBSCAN's stability-based cluster *extraction*
                        (which integrates 1/d) shifts slightly (ARI ~0.96 vs dense cosine).
                        The agglomerative cross-check (also dense) is skipped here.

Output (clusters/model_clusters.tsv): `(protein_id, cluster_id)` where multi-member
clusters get ids 0..k and singletons get a unique NEGATIVE id. The agglomerative
cross-check is written alongside as model_clusters_agglomerative.tsv.

    python cluster_embeddings.py
    python cluster_embeddings.py --min-cluster-size 2 --threshold 0.45
"""
import os
import argparse
import numpy as np

import cluster_common as cc


def relabel(labels):
    """Map a raw label vector to the report convention:
    multi-member clusters -> 0,1,2,... ; every singleton/noise point -> -1,-2,-3,...

    HDBSCAN noise is -1; agglomerative produces genuine size-1 clusters. Both are
    treated as singletons here.
    """
    labels = np.asarray(labels)
    # Count members per (non-noise) label.
    sizes = {}
    for lab in labels:
        if lab < 0:
            continue
        sizes[lab] = sizes.get(lab, 0) + 1
    multi = sorted(l for l, n in sizes.items() if n >= 2)
    remap = {l: i for i, l in enumerate(multi)}
    out = np.empty(len(labels), dtype=np.int64)
    next_singleton = -1
    for idx, lab in enumerate(labels):
        if lab >= 0 and sizes[lab] >= 2:
            out[idx] = remap[lab]
        else:
            out[idx] = next_singleton
            next_singleton -= 1
    return out


def log_stats(name, labels):
    n = len(labels)
    singletons = int((labels < 0).sum())
    multi_ids = sorted(set(int(l) for l in labels if l >= 0))
    sizes = [int((labels == cid).sum()) for cid in multi_ids]
    print(f"\n[{name}]")
    print(f"  proteins                 : {n}")
    print(f"  multi-member clusters    : {len(multi_ids)}")
    print(f"  singletons               : {singletons}  ({singletons / n:.1%})")
    print(f"  proteins in multi-member : {n - singletons}  ({(n - singletons) / n:.1%})")
    if sizes:
        sizes_sorted = sorted(sizes, reverse=True)
        print(f"  cluster size  min/median/max : {min(sizes)} / {int(np.median(sizes))} / {max(sizes)}")
        print(f"  largest 10 clusters      : {sizes_sorted[:10]}")
        # coarse size histogram
        bins = [(2, 2), (3, 5), (6, 10), (11, 25), (26, 1 << 30)]
        print("  size distribution:")
        for lo, hi in bins:
            c = sum(1 for s in sizes if lo <= s <= hi)
            label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1 << 20 else f"{lo}-{hi}")
            print(f"    size {label:<6}: {c} clusters")


def write_tsv(path, ids, labels):
    with open(path, 'w') as f:
        f.write("protein_id\tcluster_id\n")
        for pid, lab in zip(ids, labels):
            f.write(f"{pid}\t{int(lab)}\n")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', default=cc.EMB_FILE)
    ap.add_argument('--out', default=os.path.join(cc.OUT_DIR, 'model_clusters.tsv'))
    ap.add_argument('--min-cluster-size', type=int, default=2,
                    help="HDBSCAN min_cluster_size (>=2; Foldseek clusters can be pairs).")
    ap.add_argument('--min-samples', type=int, default=1,
                    help="HDBSCAN min_samples (lower = less conservative about noise).")
    ap.add_argument('--epsilon', type=float, default=0.0,
                    help="HDBSCAN cluster_selection_epsilon in cosine-distance units "
                         "(merge clusters closer than this; 0 = pure HDBSCAN).")
    ap.add_argument('--threshold', type=float, default=cc.CLOSE_THRESHOLD,
                    help="Cosine-distance cut for the agglomerative cross-check.")
    ap.add_argument('--method', choices=['hdbscan', 'agglomerative'], default='hdbscan',
                    help="Which result becomes the primary model_clusters.tsv.")
    ap.add_argument('--max-dense', type=int, default=15000,
                    help="Above this many proteins, switch HDBSCAN to the scalable euclidean "
                         "space-tree path and skip the agglomerative cross-check; both need a "
                         "dense (N,N) matrix (~N^2*8 bytes) that is infeasible at full scale.")
    ap.add_argument('--hdbscan-algorithm', default='best',
                    help="hdbscan algorithm for the euclidean (large-N) path ('best' picks a "
                         "space-tree; avoid 'generic' -- it materialises the full pairwise matrix).")
    args = ap.parse_args()

    cc.ensure_out_dir(os.path.dirname(args.out) or '.')
    ids, X = cc.load_embeddings(args.emb)
    N = len(ids)
    print(f"{N} embeddings, dim {X.shape[1]}")
    dense_ok = N <= args.max_dense
    import hdbscan

    if dense_ok:
        # --- Exact path: precomputed cosine distances (matches all prior calibration) ---
        print("computing cosine-distance matrix ...")
        D = cc.cosine_distance_matrix(X).astype(np.float64)
        print(f"\nHDBSCAN(min_cluster_size={args.min_cluster_size}, "
              f"min_samples={args.min_samples}, epsilon={args.epsilon}, metric=precomputed)")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.epsilon,
            metric='precomputed',
        )
        hdb_labels = relabel(clusterer.fit_predict(D))
    else:
        # --- Scalable path: euclidean over unit-norm vectors via space-trees (O(N) mem) ---
        # cluster_selection_epsilon is an absolute distance, so convert cosine -> euclidean.
        D = None
        eps_eucl = float(np.sqrt(2.0 * args.epsilon)) if args.epsilon > 0 else 0.0
        print(f"\n[N={N} > --max-dense={args.max_dense}] dense matrix infeasible "
              f"(~{N*N*8/1e9:.0f}GB); using scalable euclidean HDBSCAN.")
        print(f"HDBSCAN(min_cluster_size={args.min_cluster_size}, "
              f"min_samples={args.min_samples}, epsilon={args.epsilon}[cos]->{eps_eucl:.4f}[eucl], "
              f"metric=euclidean, algorithm={args.hdbscan_algorithm})")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=eps_eucl,
            metric='euclidean',
            algorithm=args.hdbscan_algorithm,
            core_dist_n_jobs=-1,
        )
        hdb_labels = relabel(clusterer.fit_predict(np.ascontiguousarray(X)))
    log_stats('HDBSCAN', hdb_labels)

    # --- Cross-check: average-linkage agglomerative (needs the dense matrix) -----
    agg_labels = None
    if dense_ok:
        from sklearn.cluster import AgglomerativeClustering
        print(f"\nAgglomerative(average linkage, cosine threshold={args.threshold})")
        agglo = AgglomerativeClustering(
            n_clusters=None, metric='precomputed', linkage='average',
            distance_threshold=args.threshold,
        )
        agg_labels = relabel(agglo.fit_predict(D))
        log_stats('Agglomerative', agg_labels)
        del D
    else:
        print(f"\n[Agglomerative] SKIPPED: N={N} > --max-dense={args.max_dense}; its dense "
              f"matrix is infeasible. HDBSCAN is the trustworthy partition per the decision-gate analysis.")

    # --- Write outputs ---------------------------------------------------------
    print()
    if args.method == 'agglomerative' and agg_labels is None:
        print("[!] --method agglomerative requested but it was skipped (N too large); "
              "writing HDBSCAN as primary.")
        args.method = 'hdbscan'
    primary = hdb_labels if args.method == 'hdbscan' else agg_labels
    write_tsv(args.out, ids, primary)
    if agg_labels is not None:
        secondary = agg_labels if args.method == 'hdbscan' else hdb_labels
        sec_path = os.path.join(os.path.dirname(args.out) or '.',
                                'model_clusters_agglomerative.tsv'
                                if args.method == 'hdbscan' else 'model_clusters_hdbscan.tsv')
        write_tsv(sec_path, ids, secondary)
    print(f"\nPrimary method: {args.method} -> {args.out}")


if __name__ == '__main__':
    main()
