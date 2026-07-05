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


def decollapse(X, mode='none'):
    """De-collapse an embedding set with a FIXED (no-training) linear transform so
    metrics reflect information rather than a random/near-collapse mean direction.
    'center' removes the shared mean direction; 'pca' additionally whitens
    (decorrelate + unit variance). Spearman TM-rho is ~unchanged by this (it is
    rank-based), but recall/ARI stop being collapse-inflated."""
    X = np.asarray(X, dtype=np.float64)
    if mode in ('center', 'pca'):
        Xc = X - X.mean(axis=0, keepdims=True)
        if mode == 'center':
            return Xc
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt.T / (S / np.sqrt(len(Xc)) + 1e-8)
    return X


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


def _health_metrics(X, max_pairs=1000, seed=0):
    """§6.1 collapse/health gate (plan v2) — the BLOCKING metrics.

      effective_rank : Roy–Vetterli effective rank exp(-Σ p_i log p_i) of the
                       embedding singular-value spectrum (p_i = s_i/Σs). ~D = full
                       use of the space; ~1 = (near-)dimensional collapse.
      emb_std        : mean per-dimension std (low -> collapse).
      mean_cos       : mean off-diagonal cosine (high -> collapse).
      uniformity     : Wang & Isola log E[exp(-2‖z_i-z_j‖²)] over random pairs;
                       more negative = better spread (anti-collapse). `[alignunif]`
    X is L2-normalized (N, D)."""
    N = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    emb_std = float(X.std(0).mean())
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, max_pairs, replace=False) if N > max_pairs else np.arange(N)
    Xs = X[idx]
    gram = Xs @ Xs.T
    iu = np.triu_indices(len(Xs), k=1)
    sq = np.maximum(0.0, 2.0 - 2.0 * gram[iu])          # ‖a-b‖² for unit vectors
    uniformity = float(np.log(np.mean(np.exp(-2.0 * sq)) + 1e-12))
    mean_cos = float(gram[iu].mean())
    return {'effective_rank': round(eff_rank, 3), 'emb_std': round(emb_std, 4),
            'mean_cos': round(mean_cos, 4), 'uniformity': round(uniformity, 4)}


def _clustering_metrics(model_labels, fs_labels, n_perm=5, seed=0):
    """ARI/NMI + Mod 4 directional agreement (V-measure / homogeneity / completeness
    / Fowlkes–Mallows, fragmentation/fusion) + Mod 3 permutation-null ARI, all on the
    subset where model AND Foldseek clusters are both multi-member."""
    from collections import Counter, defaultdict
    mc, fc = Counter(model_labels), Counter(fs_labels)
    keep = [i for i in range(len(model_labels))
            if mc[model_labels[i]] >= 2 and fc[fs_labels[i]] >= 2]
    nan = float('nan')
    if len(keep) < 2:
        return {'hdbscan_ari': nan, 'hdbscan_nmi': nan, 'homogeneity': nan,
                'completeness': nan, 'v_measure': nan, 'fowlkes_mallows': nan,
                'fragmentation': nan, 'fusion': nan, 'pair_fpr': nan, 'pair_fnr': nan,
                'perm_ari': nan, 'n_eval': 0}
    ml = [model_labels[i] for i in keep]
    fl = [fs_labels[i] for i in keep]
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_score, completeness_score, v_measure_score,
                                 fowlkes_mallows_score)
    # Mod 4 fragmentation (Foldseek cluster split across model clusters) / fusion (inverse)
    f2m, m2f = defaultdict(set), defaultdict(set)
    for a, b in zip(fl, ml):
        f2m[a].add(b); m2f[b].add(a)
    frag = sum(len(v) for v in f2m.values()) / len(f2m)
    fus = sum(len(v) for v in m2f.values()) / len(m2f)
    # Pair-level false-positive / false-negative rates (§Pair FPR/FNR). A pair is a
    # positive if the two proteins share a reference (Foldseek) fold; the clustering
    # co-clusters them or not. Over the contingency counts n_ck = |C_c ∩ K_k|:
    #   TP = Σ_ck C(n_ck,2);  FP = Σ_k C(n_k,2) - TP;  FN = Σ_c C(n_c,2) - TP;
    #   TN = C(n,2) - TP - FP - FN.
    #   FPR = FP/(FP+TN) = fraction of cross-fold pairs wrongly co-clustered (over-merge);
    #   FNR = FN/(FN+TP) = fraction of same-fold pairs left split (over-split).
    # Computed on the same `keep` subset as the other agreement metrics.
    def _c2(x):
        return x * (x - 1) // 2
    cont = Counter(zip(fl, ml))
    n_c = Counter(fl)
    n_k = Counter(ml)
    tp = sum(_c2(v) for v in cont.values())
    sum_c = sum(_c2(v) for v in n_c.values())          # same-fold (positive) pairs
    sum_k = sum(_c2(v) for v in n_k.values())
    total_pairs = _c2(len(fl))
    fp = sum_k - tp
    fn = sum_c - tp
    tn = total_pairs - tp - fp - fn
    fpr = fp / (fp + tn) if (fp + tn) > 0 else nan     # = fp / (total_pairs - sum_c)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else nan     # = fn / sum_c
    # Mod 3 permutation-null ARI: shuffle the Foldseek labels (preserves marginal
    # cluster sizes) -> calibrated ARI floor. ARI_real - perm_ari = label contribution.
    rng = np.random.default_rng(seed)
    fl_arr = np.array(fl)
    perm = float(np.mean([adjusted_rand_score(rng.permutation(fl_arr), ml) for _ in range(n_perm)]))
    return {
        'hdbscan_ari': round(adjusted_rand_score(fl, ml), 4),
        'hdbscan_nmi': round(normalized_mutual_info_score(fl, ml), 4),
        'homogeneity': round(homogeneity_score(fl, ml), 4),
        'completeness': round(completeness_score(fl, ml), 4),
        'v_measure': round(v_measure_score(fl, ml), 4),
        'fowlkes_mallows': round(fowlkes_mallows_score(fl, ml), 4),
        'fragmentation': round(frag, 3), 'fusion': round(fus, 3),
        'pair_fpr': round(fpr, 4), 'pair_fnr': round(fnr, 4),
        'perm_ari': round(perm, 4), 'n_eval': len(keep),
    }


def evaluate(embs, ids, foldseek_labels, tm_cache=None, tm_cache_path='./checkpoints/tm_score_cache.pt',
             min_cluster_size=2, min_samples=None, close_threshold=0.45, whiten='none'):
    # NOTE: min_samples=None lets HDBSCAN use its default (= min_cluster_size). The
    # old default min_samples=1 over-fragments hard (every point joins a tiny
    # cluster), which crushed ARI and masked real differences — e.g. ep24 ARI went
    # 0.27 (ms=1) -> 0.68 (ms=None), and the training-vs-baseline gain only shows
    # at ms=None. Pass min_samples=1 explicitly only to reproduce old logged values.
    """Cluster `embs` with HDBSCAN (euclidean over L2-normalised vectors -- the scalable,
    pipeline-consistent metric) and score against `foldseek_labels`; plus TM-correlation.

    embs            (N, D) array of raw model embeddings (need not be normalised).
    ids             list of N canonical protein ids (no .pt), aligned with embs rows.
    foldseek_labels list of N labels (Foldseek cluster rep / id), aligned with embs rows.
    Returns a metrics dict.
    """
    import hdbscan
    X = _l2(decollapse(embs, whiten))     # optional fixed de-collapse before metrics
    N = X.shape[0]

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric='euclidean', algorithm='best', core_dist_n_jobs=-1,
    ).fit_predict(np.ascontiguousarray(X))
    labels = _relabel_singletons(labels)
    singleton_frac = float((labels < 0).mean())
    n_clusters = int(len({int(l) for l in labels if l >= 0}))

    # §6.1 health gate (blocking) + Mod 3/4 clustering agreement.
    health = _health_metrics(X)
    clu = _clustering_metrics(list(labels), list(foldseek_labels))
    n_eval = clu['n_eval']

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
    tm_alignment = float('nan')          # §6.1 alignment over structural positives
    n_pairs = int(len(dists))
    if n_pairs >= 5:
        from scipy.stats import spearmanr
        tm_rho = float(spearmanr(dists, tms)[0])
        cross = tms > 0.5
        if cross.sum() > 0:
            tm_recall = float((dists[cross] < close_threshold).mean())
            # alignment = E‖z_i-z_j‖² over high-TM (positive) pairs = 2*(1-cos); lower=better.
            tm_alignment = float(np.mean(2.0 * dists[cross]))

    out = {
        'n': N, 'n_clusters': n_clusters, 'singleton_frac': round(singleton_frac, 4),
        'n_eval': n_eval,
        'tm_rho': round(tm_rho, 4) if tm_rho == tm_rho else float('nan'),
        'tm_recall': round(tm_recall, 4) if tm_recall == tm_recall else float('nan'),
        'tm_alignment': round(tm_alignment, 4) if tm_alignment == tm_alignment else float('nan'),
        'n_tm_pairs': n_pairs,
    }
    out.update(health)    # effective_rank, emb_std, mean_cos, uniformity
    out.update(clu)       # hdbscan_ari/nmi, homogeneity, completeness, v_measure,
                          # fowlkes_mallows, fragmentation, fusion, perm_ari (+ n_eval)
    return out


_FIELDS = ['epoch', 'n', 'n_clusters', 'singleton_frac', 'n_eval',
           'tm_rho', 'tm_recall', 'tm_alignment', 'n_tm_pairs',
           # §6.1 health gate
           'effective_rank', 'emb_std', 'mean_cos', 'uniformity',
           # Mod 3/4 clustering agreement
           'hdbscan_ari', 'hdbscan_nmi', 'homogeneity', 'completeness', 'v_measure',
           'fowlkes_mallows', 'fragmentation', 'fusion', 'pair_fpr', 'pair_fnr', 'perm_ari']


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
    g = lambda k: m.get(k, '?')
    return (f"TM-rho={g('tm_rho')} recall={g('tm_recall')} align={g('tm_alignment')} "
            f"({g('n_tm_pairs')} pairs) | health: eff_rank={g('effective_rank')} "
            f"emb_std={g('emb_std')} mean_cos={g('mean_cos')} unif={g('uniformity')} | "
            f"ARI={g('hdbscan_ari')} (perm {g('perm_ari')}) Vm={g('v_measure')} "
            f"FM={g('fowlkes_mallows')} frag={g('fragmentation')} fus={g('fusion')} "
            f"FPR={g('pair_fpr')} FNR={g('pair_fnr')} "
            f"[{g('n_clusters')} cl, {m.get('singleton_frac', 0):.0%} singl, n_eval={g('n_eval')}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', default='./data/virome_embeddings.pt',
                    help="Embeddings .pt produced by extract_embeddings.py (full-dataset deep check).")
    ap.add_argument('--tm-cache', default='./checkpoints/tm_score_cache.pt')
    ap.add_argument('--min-cluster-size', type=int, default=2)
    ap.add_argument('--log', default=None, help="Optional CSV to append the result to.")
    ap.add_argument('--epoch', default='full', help="Epoch label for the logged row.")
    ap.add_argument('--whiten', choices=['none', 'center', 'pca'], default='none',
                    help="De-collapse the embeddings with a fixed transform before metrics "
                         "(center / PCA-whiten). Use for non-collapsed baselines so recall/ARI "
                         "aren't collapse-inflated. TM-rho is ~unchanged (rank-based).")
    ap.add_argument('--cluster-source', choices=['foldseek', 'nomburg'], default='nomburg',
                    help="Ground-truth cluster labels to score against. 'nomburg' uses the "
                         "18k merged clusters from merged_clusters.tax.tsv (the Nomburg study "
                         "final set). 'foldseek' uses the raw FoldSeek pairwise cluster.tsv.")
    args = ap.parse_args()

    import cluster_common as cc
    ids, X = cc.load_embeddings(args.emb)
    if args.cluster_source == 'nomburg':
        rep_of = cc.load_nomburg_clusters()
        print(f"  cluster source: Nomburg merged ({len(set(rep_of.values()))} clusters)")
    else:
        rep_of = cc.load_foldseek_clusters() if hasattr(cc, 'load_foldseek_clusters') else None
        if rep_of is None:
            rep_of = {}
            if os.path.exists(cc.CLUSTER_TSV):
                with open(cc.CLUSTER_TSV) as f:
                    for line in f:
                        c = line.rstrip('\n').split('\t')
                        if len(c) >= 2:
                            rep_of[c[1]] = c[0]
        print(f"  cluster source: FoldSeek pairwise ({len(set(rep_of.values()))} clusters)")
    fs_labels = [rep_of.get(i, i) for i in ids]
    print(f"{len(ids)} embeddings | evaluating (whiten={args.whiten}) ...")
    m = evaluate(X, ids, fs_labels, tm_cache_path=args.tm_cache,
                 min_cluster_size=args.min_cluster_size, whiten=args.whiten)
    print(format_line(m))
    if args.log:
        log_row(args.log, args.epoch, m)
        print(f"  appended to {args.log}")


if __name__ == '__main__':
    main()
