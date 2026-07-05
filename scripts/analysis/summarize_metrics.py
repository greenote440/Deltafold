"""Summarise a training run's metrics into a Markdown table (protocol §Plan of
experiments), from its ``training_log.jsonl.gz``.

Writes ``metrics.md`` into the run folder: a per-epoch table of the key metrics
plus a detailed, protocol-grouped breakdown of the last epoch (directional
agreement, secondary summaries, health checks) and any logged collapse events.

Usage:
  python scripts/analysis/summarize_metrics.py checkpoints/<run_folder>
  python scripts/analysis/summarize_metrics.py checkpoints/<run_folder> --out metrics.md
"""
import os
import gzip
import json
import glob
import argparse


def _load_epochs(log_path):
    recs = []
    with gzip.open(log_path, "rb") as fz:
        for line in fz:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    epochs = [r for r in recs if r.get("t") == "epoch"]
    collapses = [r for r in recs if r.get("t") == "collapse"]
    return epochs, collapses


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _load_config(run_dir):
    """Best-effort: pull model_config from a checkpoint so the run is self-describing."""
    cks = (glob.glob(os.path.join(run_dir, "*best.pth"))
           or glob.glob(os.path.join(run_dir, "*last.pth"))
           or glob.glob(os.path.join(run_dir, "*epoch*.pth")))
    if not cks:
        return {}
    try:
        import torch
        ck = torch.load(cks[0], map_location="cpu", weights_only=False)
        return ck.get("model_config", {}) or {}
    except Exception as e:
        return {"(config load failed)": str(e)[:80]}


def build_markdown(run_dir, epochs, collapses, config):
    ev_last = epochs[-1].get("eval", {}) if epochs else {}

    def ev(k):
        return _fmt(ev_last.get(k))

    L = []
    A = L.append
    name = os.path.basename(os.path.normpath(run_dir))
    A(f"# Training metrics — `{name}`")
    A("")
    A(f"{len(epochs)} epoch(s) logged. Metric names follow the DeltaFold protocol "
      "(§Plan of experiments). Generated from `training_log.jsonl.gz` by "
      "`scripts/analysis/summarize_metrics.py`.")
    A("")

    # Run config
    if config:
        A("## Run configuration")
        A("")
        A("| field | value |")
        A("|---|---|")
        for k, v in config.items():
            A(f"| {k} | {v} |")
        A("")

    # Per-epoch overview (one row per epoch)
    A("## Per-epoch overview")
    A("")
    A("| ep | train loss | val loss | h | c | V | frag | fus | HDBSCAN ARI | TM-ρ | emb_std | mean_cos | eff_rank |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in epochs:
        e = r.get("eval", {})
        A("| {ep} | {tl} | {vl} | {h} | {c} | {V} | {fr} | {fu} | {ari} | {rho} | {std} | {mc} | {er} |".format(
            ep=r.get("ep"), tl=_fmt(r.get("loss")), vl=_fmt(r.get("vloss")),
            h=_fmt(e.get("homogeneity")), c=_fmt(e.get("completeness")), V=_fmt(e.get("v_measure")),
            fr=_fmt(e.get("fragmentation")), fu=_fmt(e.get("fusion")),
            ari=_fmt(e.get("hdbscan_ari")), rho=_fmt(e.get("tm_rho")),
            std=_fmt(e.get("emb_std")), mc=_fmt(e.get("mean_cos")), er=_fmt(e.get("effective_rank"))))
    A("")

    A(f"## Last epoch (ep {epochs[-1].get('ep')}) — detailed")
    A("")
    A("### Primary — directional agreement (§Homogeneity/Completeness, §Fragmentation/Fusion)")
    A("")
    A("| metric | value | ideal | reads |")
    A("|---|---|---|---|")
    A(f"| Homogeneity h | {ev('homogeneity')} | 1 | learned clusters pure in reference labels |")
    A(f"| Completeness c | {ev('completeness')} | 1 | each reference fold in a single learned cluster |")
    A(f"| V-measure V | {ev('v_measure')} | 1 | harmonic mean of h and c |")
    A(f"| Fragmentation | {ev('fragmentation')} | 1 | mean learned-clusters a fold is split across (over-split) |")
    A(f"| Fusion | {ev('fusion')} | 1 | mean folds merged into one learned cluster (over-merge) |")
    A(f"| Pair FPR | {ev('pair_fpr')} | 0 | cross-fold pairs wrongly co-clustered (— if not logged) |")
    A(f"| Pair FNR | {ev('pair_fnr')} | low | same-fold pairs split apart (— if not logged) |")
    A(f"| selected ε (HDBSCAN) | {ev('selected_epsilon')} | — | tuned to min FNR at FPR≤cap |")
    A("")
    A("### Secondary summaries (§ARI, §TM-ρ)")
    A("")
    A("| metric | value | note |")
    A("|---|---|---|")
    A(f"| HDBSCAN ARI | {ev('hdbscan_ari')} | adjusted Rand vs reference clusters |")
    A(f"| HDBSCAN NMI | {ev('hdbscan_nmi')} | normalized mutual information |")
    A(f"| Fowlkes–Mallows | {ev('fowlkes_mallows')} | pair-level precision/recall geo-mean |")
    A(f"| KMeans ARI (val) | {_fmt(epochs[-1].get('ari'))} | per-epoch quick ARI |")
    A(f"| Permutation ARI | {ev('perm_ari')} | ARI vs shuffled labels (floor; ~0 expected) |")
    A(f"| TM-ρ (Spearman) | {ev('tm_rho')} | rank corr of cosine-dist vs TM (— if no TM cache) |")
    A(f"| TM recall | {ev('tm_recall')} | — |")
    A("")
    A("### Health checks (§Health checks — gating: interpret a checkpoint only if healthy)")
    A("")
    A("| check | value | protocol guideline |")
    A("|---|---|---|")
    A(f"| emb_std (per-dim spread) | {ev('emb_std')} | stay clearly above ~0.02 (collapse threshold) |")
    A(f"| mean off-diag cosine | {ev('mean_cos')} | stay low; 0.9–0.97 ⇒ near-parallel collapse |")
    A(f"| effective rank | {ev('effective_rank')} | number of embedding dims actually used |")
    A(f"| uniformity | {ev('uniformity')} | more negative ⇒ better spread on the sphere |")
    A(f"| n_clusters | {ev('n_clusters')} | learned cluster count |")
    A(f"| singleton fraction | {ev('singleton_frac')} | high ⇒ under-clustered / noisy |")
    A("")

    std = ev_last.get("emb_std")
    mc = ev_last.get("mean_cos")
    healthy = (std is not None and std > 0.02) and (mc is not None and mc < 0.9)
    A(f"**Health at last epoch: {'looks healthy' if healthy else 'CHECK — possible collapse'}** "
      f"(emb_std {'>' if (std and std>0.02) else '≤'} 0.02 floor; "
      f"mean_cos {'below' if (mc and mc<0.9) else 'in/above'} the collapse band).")
    A("")
    if collapses:
        A(f"> ⚠ {len(collapses)} `collapse` event(s) logged during training:")
        for c in collapses:
            A(f">   `{json.dumps(c, separators=(',', ':'))[:300]}`")
        A("")
    if len(epochs) < 3:
        A("> Note: fewer than 3 epochs — the protocol reads the *level and trend* over the "
          "last several epochs (averaged over seeds), so a single point is not a reliable readout.")
        A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Run folder containing training_log.jsonl.gz")
    ap.add_argument("--out", default="metrics.md", help="Output filename inside the run folder.")
    args = ap.parse_args()

    log_path = os.path.join(args.run_dir, "training_log.jsonl.gz")
    if not os.path.exists(log_path):
        raise SystemExit(f"No training_log.jsonl.gz in {args.run_dir}")
    epochs, collapses = _load_epochs(log_path)
    if not epochs:
        raise SystemExit(f"No epoch records in {log_path}")
    config = _load_config(args.run_dir)
    md = build_markdown(args.run_dir, epochs, collapses, config)
    out_path = os.path.join(args.run_dir, args.out)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Wrote {out_path} ({len(epochs)} epoch(s), {len(collapses)} collapse event(s)).")


if __name__ == "__main__":
    main()
