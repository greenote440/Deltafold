"""
Phase 5.1 - UMAP projection of the 128D embeddings to 2D.

Uses the cosine metric (consistent with the contrastive training objective) and runs
UMAP twice: n_neighbors=15 (local structure) and n_neighbors=50 (global structure).
Both projections are saved into one master TSV (clusters/umap_coords.tsv):

    protein_id, umap_x, umap_y, umap_x_n50, umap_y_n50

umap_x/umap_y are the n_neighbors=15 coordinates (the default the downstream
visualisation and annotation table consume).

    python project_umap.py
    python project_umap.py --neighbors 15 50 --min-dist 0.1 --seed 42
"""
import os
import argparse
import numpy as np

import cluster_common as cc


def run_umap(X, n_neighbors, min_dist, seed):
    import umap
    reducer = umap.UMAP(
        n_components=2, metric='cosine', n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=seed,
    )
    return reducer.fit_transform(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', default=cc.EMB_FILE)
    ap.add_argument('--out', default=os.path.join(cc.OUT_DIR, 'umap_coords.tsv'))
    ap.add_argument('--neighbors', type=int, nargs=2, default=[15, 50],
                    help="The two n_neighbors values (local, global).")
    ap.add_argument('--min-dist', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cc.ensure_out_dir(os.path.dirname(args.out) or '.')
    ids, X = cc.load_embeddings(args.emb)
    n_local, n_global = args.neighbors
    print(f"{len(ids)} embeddings -> UMAP 2D (cosine, min_dist={args.min_dist}, seed={args.seed})")

    print(f"  UMAP n_neighbors={n_local} (local) ...")
    Y_local = run_umap(X, n_local, args.min_dist, args.seed)
    print(f"  UMAP n_neighbors={n_global} (global) ...")
    Y_global = run_umap(X, n_global, args.min_dist, args.seed)

    with open(args.out, 'w') as f:
        f.write("protein_id\tumap_x\tumap_y\tumap_x_n50\tumap_y_n50\n")
        for pid, (x, y), (xg, yg) in zip(ids, Y_local, Y_global):
            f.write(f"{pid}\t{x:.5f}\t{y:.5f}\t{xg:.5f}\t{yg:.5f}\n")
    print(f"wrote {args.out}  (umap_x/umap_y = n_neighbors={n_local}; "
          f"umap_x_n50/umap_y_n50 = n_neighbors={n_global})")


if __name__ == '__main__':
    main()
