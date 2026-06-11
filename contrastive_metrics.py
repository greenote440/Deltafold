"""
Per-epoch clustering-quality metrics for the contrastive run.

These are numpy-only reimplementations (sklearn is not always available in ml_env,
and importing it lazily inside the hot validation path is undesirable). They answer
one question during training: "is the embedding space recovering the ground-truth
fold clusters yet?" — tracked as the Adjusted Rand Index between a cosine k-means
clustering of the validation embeddings and the cluster.tsv labels.

Deeper structural metrics (HDBSCAN ARI/NMI, TM-rho, homology recall) live in
epoch_eval.py; this module only covers the lightweight in-loop ARI signal.
"""
import numpy as np


def _adjusted_rand_index(labels_true, labels_pred):
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
