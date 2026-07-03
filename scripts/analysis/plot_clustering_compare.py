"""
Compare the random-init feature baseline vs the trained model on all Foldseek
clustering metrics (ARI + Mod 4 directional: homogeneity / completeness /
V-measure / Fowlkes-Mallows / fragmentation / fusion + descriptive n_clusters /
singleton fraction), as a function of HDBSCAN min_cluster_size.

All embeddings are PCA-whitened (de-collapsed) and clustered with
min_samples=None (the sensible default; min_samples=1 over-fragments). One panel
per metric, two lines (baseline vs trained).

Usage:
  python scripts/analysis/plot_clustering_compare.py \
      --baseline data/emb_moco_geom_randinit.pt \
      --trained  data/emb_moco_geom_ep24.pt \
      --out checkpoints/moco_geom/clustering_compare.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epoch_eval as ee          # noqa: E402
import cluster_common as cc      # noqa: E402


def _fs_labels(ids):
    rep = cc.load_foldseek_clusters() if hasattr(cc, 'load_foldseek_clusters') else None
    if rep is None:
        rep = {}
        if os.path.exists(cc.CLUSTER_TSV):
            for line in open(cc.CLUSTER_TSV):
                c = line.rstrip('\n').split('\t')
                if len(c) >= 2:
                    rep[c[1]] = c[0]
    return [rep.get(i, i) for i in ids]


def sweep(emb_path, mcs_list, whiten):
    ids, X = cc.load_embeddings(emb_path)
    fs = _fs_labels(ids)
    out = {}
    for mcs in mcs_list:
        m = ee.evaluate(X, ids, fs, min_cluster_size=mcs, min_samples=None, whiten=whiten)
        out[mcs] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', default='data/emb_moco_geom_randinit.pt')
    ap.add_argument('--trained', default='data/emb_moco_geom_ep24.pt')
    ap.add_argument('--out', default='checkpoints/moco_geom/clustering_compare.png')
    ap.add_argument('--whiten', default='pca', choices=['none', 'center', 'pca'])
    ap.add_argument('--mcs', type=int, nargs='*', default=[2, 3, 5, 8, 12, 20])
    args = ap.parse_args()

    print(f"baseline={args.baseline}  trained={args.trained}  whiten={args.whiten}")
    base = sweep(args.baseline, args.mcs, args.whiten)
    trn = sweep(args.trained, args.mcs, args.whiten)

    panels = [
        ('hdbscan_ari', 'ARI vs Foldseek (higher=better)', True),
        ('homogeneity', 'Homogeneity (cluster purity)', True),
        ('completeness', 'Completeness (Foldseek cluster kept whole)', True),
        ('v_measure', 'V-measure', True),
        ('fowlkes_mallows', 'Fowlkes-Mallows', True),
        ('fragmentation', 'Fragmentation (->1 ideal; Foldseek cl. split)', False),
        ('fusion', 'Fusion (->1 ideal; model cl. merges Foldseek)', False),
        ('n_clusters', '# model clusters', None),
        ('singleton_frac', 'Singleton (noise) fraction', None),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle('Foldseek clustering: random-init baseline vs trained '
                 f'(PCA-whitened, min_samples=None)', fontsize=14)
    mcs = args.mcs
    for a, (key, title, _) in zip(ax.ravel(), panels):
        a.plot(mcs, [base[m].get(key) for m in mcs], 'o-', color='tab:gray', lw=1.6, label='baseline (random feat.)')
        a.plot(mcs, [trn[m].get(key) for m in mcs], 'o-', color='tab:green', lw=1.8, label='trained')
        if key in ('hdbscan_ari',):
            a.plot(mcs, [base[m].get('perm_ari') for m in mcs], '--', color='k', lw=1, label='perm null')
        if key in ('fragmentation', 'fusion'):
            a.axhline(1.0, color='r', ls=':', lw=1, label='ideal=1')
        a.set_title(title, fontsize=10); a.set_xlabel('min_cluster_size'); a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
