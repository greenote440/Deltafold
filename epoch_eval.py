"""
Per-epoch embedding evaluation: extract -> cluster -> ARI / homology, condensed to a
one-line health check you can watch climb during training.

Two entry points:
  * evaluate(embs, ids, foldseek_labels, tm_cache=...) -> metrics dict
      Called in-loop each epoch on the VALIDATION embeddings the training loop already
      computed (no re-extraction). This is what gets logged to checkpoints/epoch_eval.csv.
  * CLI (python epoch_eval.py --emb <file.pt>)
      Standalone DEEP check on a full embeddings file (e.g. data/virome_embeddings.pt
      produced by extract_embeddings.py): clusters the whole set and reports the same
      metrics. Use this periodically; it is too heavy to run every epoch on 67k.

Metrics (all "higher/more-negative = better fold structure"):
  hdbscan_ari   ARI of an HDBSCAN partition of the embeddings vs the Foldseek labels,
                restricted to mutually multi-member clusters (matches compare_clusters.py).
  hdbscan_nmi   NMI on the same subset (high NMI + low ARI = over-segmentation).
  n_clusters    multi-member HDBSCAN clusters found.
  singleton_frac fraction of proteins HDBSCAN leaves as singletons/noise.
  tm_rho        Spearman rho(cosine-distance, TM) over cached pairs present in the set.
                Strongly NEGATIVE = the geometry encodes a continuous structural metric.
  tm_recall     fraction of cross-structure (TM>0.5) cached pairs placed in the close
                cosine band (< CLOSE_THRESHOLD) = homology recovery.
"""
import os
import argparse
import csv
import numpy as np


_TM_CACHE = {}  # path -> {frozenset(id_a,id_b): tm}; module-level so the in-loop call loads once


def _strip(s):
    return s[:-3] if s.endswith('.pt') else s


def load_tm_cache(path='./checkpoints/tm_score_cache.pt'):
    """frozenset({id_a,id_b}) -> TM, cached at module level (loaded at most once per path)."""
    if path in _TM_CACHE:
        return _TM_CACHE[path]
    cache = {}
    if path and os.path.exists(path):
        import torch
        try:
            raw = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location='cpu')
        for (a, b), tm in raw.items():
            cache[frozenset((_strip(a), _strip(b)))] = float(tm)
    _TM_CACHE[path] = cache
    return cache


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _mutual_multimember_ari_nmi(model_labels, fs_labels):
    """ARI/NMI restricted to proteins whose model AND Foldseek clusters are both
    multi-member (singletons make the comparison ill-posed). Mirrors compare_clusters.py."""
    from collections import Counter
    mc, fc = Counter(model_labels), Counter(fs_labels)
    keep = [i for i in range(len(model_labels))
            if mc[model_labels[i]] >= 2 and fc[fs_labels[i]] >= 2]
    if len(keep) < 2:
        return float('nan'), float('nan'), 0
    ml = [model_labels[i] for i in keep]
    fl = [fs_labels[i] for i in keep]
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return (adjusted_rand_score(fl, ml), normalized_mutual_info_score(fl, ml), len(keep))


def _relabel_singletons(labels):
    """HDBSCAN noise (-1) and any size-1 label -> unique negative ids; multi-member -> 0..k."""
    labels = np.asarray(labels)
    from collections import Counter
    sizes = Counter(int(l) for l in labels if l >= 0)
    multi = sorted(l for l, n in sizes.items() if n >= 2)
    remap = {l: i for i, l in enumerate(multi)}
    out = np.empty(len(labels), dtype=np.int64)
    nxt = -1
    for i, l in enumerate(labels):
        l = int(l)
        if l >= 0 and sizes[l] >= 2:
            out[i] = remap[l]
        else:
            out[i] = nxt
            nxt -= 1
    return out


def evaluate(embs, ids, foldseek_labels, tm_cache=None, tm_cache_path='./checkpoints/tm_score_cache.pt',
             min_cluster_size=2, min_samples=1, close_threshold=0.45):
    """Cluster `embs` with HDBSCAN (euclidean over L2-normalised vectors -- the scalable,
    pipeline-consistent metric) and score against `foldseek_labels`; plus TM-correlation.

    embs            (N, D) array of raw model embeddings (need not be normalised).
    ids             list of N canonical protein ids (no .pt), aligned with embs rows.
    foldseek_labels list of N labels (Foldseek cluster rep / id), aligned with embs rows.
    Returns a metrics dict.
    """
    import hdbscan
    X = _l2(np.asarray(embs, dtype=np.float64))
    N = X.shape[0]

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric='euclidean', algorithm='best', core_dist_n_jobs=-1,
    ).fit_predict(np.ascontiguousarray(X))
    labels = _relabel_singletons(labels)
    singleton_frac = float((labels < 0).mean())
    n_clusters = int(len({int(l) for l in labels if l >= 0}))

    ari, nmi, n_eval = _mutual_multimember_ari_nmi(list(labels), list(foldseek_labels))

    # --- TM correlation / homology recall over cached pairs present in this set ---
    if tm_cache is None:
        tm_cache = load_tm_cache(tm_cache_path)
    id2i = {pid: i for i, pid in enumerate(ids)}
    dists, tms = [], []
    for pair, tm in tm_cache.items():
        a, b = tuple(pair)
        ia, ib = id2i.get(a), id2i.get(b)
        if ia is None or ib is None:
            continue
        dists.append(1.0 - float(X[ia] @ X[ib]))
        tms.append(float(tm))
    dists, tms = np.array(dists), np.array(tms)
    tm_rho = float('nan')
    tm_recall = float('nan')
    n_pairs = int(len(dists))
    if n_pairs >= 5:
        from scipy.stats import spearmanr
        tm_rho = float(spearmanr(dists, tms)[0])
        cross = tms > 0.5
        if cross.sum() > 0:
            tm_recall = float((dists[cross] < close_threshold).mean())

    return {
        'n': N, 'n_clusters': n_clusters, 'singleton_frac': round(singleton_frac, 4),
        'hdbscan_ari': round(ari, 4) if ari == ari else float('nan'),
        'hdbscan_nmi': round(nmi, 4) if nmi == nmi else float('nan'),
        'n_eval': n_eval,
        'tm_rho': round(tm_rho, 4) if tm_rho == tm_rho else float('nan'),
        'tm_recall': round(tm_recall, 4) if tm_recall == tm_recall else float('nan'),
        'n_tm_pairs': n_pairs,
    }


_FIELDS = ['epoch', 'n', 'n_clusters', 'singleton_frac', 'hdbscan_ari', 'hdbscan_nmi',
           'n_eval', 'tm_rho', 'tm_recall', 'n_tm_pairs']


def log_row(csv_path, epoch, metrics):
    """Append a metrics row to epoch_eval.csv (writes the header once)."""
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(_FIELDS)
        w.writerow([epoch] + [metrics.get(k, '') for k in _FIELDS[1:]])


def format_line(metrics):
    m = metrics
    return (f"HDBSCAN-ARI={m['hdbscan_ari']} NMI={m['hdbscan_nmi']} "
            f"({m['n_clusters']} clusters, {m['singleton_frac']:.0%} singletons, n_eval={m['n_eval']}) | "
            f"TM-rho={m['tm_rho']} recall@close={m['tm_recall']} ({m['n_tm_pairs']} pairs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', default='./data/virome_embeddings.pt',
                    help="Embeddings .pt produced by extract_embeddings.py (full-dataset deep check).")
    ap.add_argument('--tm-cache', default='./checkpoints/tm_score_cache.pt')
    ap.add_argument('--min-cluster-size', type=int, default=2)
    ap.add_argument('--log', default=None, help="Optional CSV to append the result to.")
    ap.add_argument('--epoch', default='full', help="Epoch label for the logged row.")
    args = ap.parse_args()

    import cluster_common as cc
    ids, X = cc.load_embeddings(args.emb)
    rep_of = cc.load_foldseek_clusters() if hasattr(cc, 'load_foldseek_clusters') else None
    if rep_of is None:
        # fall back to cluster.tsv via load_cluster_tsv-style rep map
        rep_of = {}
        if os.path.exists(cc.CLUSTER_TSV):
            with open(cc.CLUSTER_TSV) as f:
                for line in f:
                    c = line.rstrip('\n').split('\t')
                    if len(c) >= 2:
                        rep_of[c[1]] = c[0]
    fs_labels = [rep_of.get(i, i) for i in ids]
    print(f"{len(ids)} embeddings | evaluating ...")
    m = evaluate(X, ids, fs_labels, tm_cache_path=args.tm_cache, min_cluster_size=args.min_cluster_size)
    print(format_line(m))
    if args.log:
        log_row(args.log, args.epoch, m)
        print(f"  appended to {args.log}")


if __name__ == '__main__':
    main()
