"""Grid-search the real HDBSCAN knobs to MINIMISE the global pair FNR while keeping
pair FPR near zero, on an embedding file vs the Nomburg reference clusters.

HDBSCAN has no single DBSCAN epsilon: it builds the hierarchy over all density levels
and extracts a flat clustering by cluster persistence. The genuine knobs are:
  * min_cluster_size          -- primary granularity (bigger -> fewer, larger clusters)
  * min_samples               -- conservativeness / noise (lower -> less noise)
  * cluster_selection_method  -- 'eom' (persistence, coarser) or 'leaf' (finest)
  * cluster_selection_epsilon -- optional coarsening floor (0 = pure HDBSCAN); secondary

Only the MST depends on min_samples, so we fit once per min_samples and re-extract
every (min_cluster_size, method, epsilon) off the single-linkage tree via the library's
own condense/extract routines -- genuine eom/leaf extraction (no single-linkage chaining).
Metrics are the GLOBAL ones from epoch_eval (noise counts as split), so FNR is real.

Usage (box, repo root):
  .venv/bin/python scripts/analysis/hdbscan_sweep.py --emb /tmp/ep25_all_emb.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import hdbscan
from hdbscan._hdbscan_tree import condense_tree, compute_stability, get_clusters
import cluster_common as cc
from epoch_eval import _relabel_singletons, _clustering_metrics


def _labels_from_slt(single_linkage, mcs, method, eps):
    """Genuine HDBSCAN flat extraction: condense the single-linkage tree at min_cluster_size
    then select clusters by 'eom'/'leaf' persistence (+ optional epsilon floor)."""
    condensed = condense_tree(single_linkage, mcs)
    labels = get_clusters(condensed, compute_stability(condensed),
                          cluster_selection_method=method, allow_single_cluster=False,
                          cluster_selection_epsilon=float(eps))[0]
    return np.asarray(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--fpr-cap", type=float, default=0.005, help="Keep FPR <= this (near 0).")
    ap.add_argument("--min-samples", default="1,2,5")
    ap.add_argument("--min-cluster-size", default="2,5,10,25,50,100", help="Primary knob.")
    ap.add_argument("--method", default="eom,leaf")
    ap.add_argument("--eps", default="0.0", help="cluster_selection_epsilon (secondary; 0 = pure).")
    args = ap.parse_args()

    ids, X = cc.load_embeddings(args.emb)          # L2-normalised
    X = np.ascontiguousarray(X)
    nom = cc.load_nomburg_clusters()
    fl = [nom.get(i, i) for i in ids]
    print(f"{len(ids)} embeddings, {len(set(fl))} reference groups", flush=True)

    mss = [int(x) for x in args.min_samples.split(",")]
    mcss = [int(x) for x in args.min_cluster_size.split(",")]
    methods = args.method.split(",")
    epss = [float(x) for x in args.eps.split(",")]

    rows = []
    for ms in mss:
        print(f"[fit] min_samples={ms} (building single-linkage tree) ...", flush=True)
        cl = hdbscan.HDBSCAN(min_samples=ms, min_cluster_size=2, metric="euclidean",
                             algorithm="best", core_dist_n_jobs=-1,
                             gen_min_span_tree=True).fit(X)
        slt = cl.single_linkage_tree_._linkage          # scipy-format hierarchy
        for mcs in mcss:
            for method in methods:
                for eps in epss:
                    lab = _relabel_singletons(_labels_from_slt(slt, mcs, method, eps))
                    noise = float((lab < 0).mean())
                    m = _clustering_metrics(list(lab), fl)
                    r = dict(ms=ms, mcs=mcs, method=method, eps=eps,
                             fnr=m["pair_fnr"], fpr=m["pair_fpr"], noise=noise,
                             ari=m["hdbscan_ari"], frag=m["fragmentation"],
                             nclu=len({int(x) for x in lab if x >= 0}))
                    rows.append(r)
                    print(f"  ms={ms} mcs={mcs:>3} {method:4} eps={eps:.2f} | "
                          f"FNR={r['fnr']} FPR={r['fpr']} noise={noise:.2f} "
                          f"ARI={r['ari']} frag={r['frag']} nclu={r['nclu']}", flush=True)

    def valid(r):
        return r["fpr"] == r["fpr"] and r["fnr"] == r["fnr"]
    for cap in (0.001, args.fpr_cap, 0.01):
        ok = sorted([r for r in rows if valid(r) and r["fpr"] <= cap], key=lambda r: r["fnr"])
        print(f"\n=== lowest FNR at FPR <= {cap:g} ===", flush=True)
        for r in ok[:6]:
            print(f"  ms={r['ms']} mcs={r['mcs']} {r['method']} eps={r['eps']:.2f} -> "
                  f"FNR={r['fnr']} FPR={r['fpr']} noise={r['noise']:.2f} ARI={r['ari']} "
                  f"frag={r['frag']} nclu={r['nclu']}", flush=True)


if __name__ == "__main__":
    main()
