"""Re-evaluate every per-epoch checkpoint in a run folder with the AUTO-TUNED-epsilon
eval, then plot FNR / FPR / fragmentation / fusion vs epoch.

For each ``checkpoint_*epochNNN.pth`` it extracts embeddings for the ``--file-list``
proteins (via extract_embeddings.py as a subprocess — reusing the tested path) and
scores them with ``epoch_eval.evaluate(tune_epsilon=True)``. Needed because a run
logged before the epsilon-tuning landed only has epsilon=0 metrics. Writes a CSV and
a 2x2 PNG into the run folder.

Usage (on the box, from the repo root):
  .venv/bin/python scripts/analysis/eval_all_epochs.py \
      --run-dir checkpoints/<run> --file-list /tmp/val_files.txt --deltafold
"""
import os
import re
import sys
import csv
import glob
import argparse
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # scripts/analysis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # repo root
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cluster_common as cc
import epoch_eval

PANELS = [("pair_fnr", "Pair FNR  (same-fold pairs left split)"),
          ("pair_fpr", "Pair FPR  (cross-fold pairs merged)"),
          ("fragmentation", "Fragmentation  (fold split across clusters)"),
          ("fusion", "Fusion  (folds merged into one cluster)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Folder with checkpoint_*epochNNN.pth.")
    ap.add_argument("--file-list", required=True, help="Validation protein manifest.")
    ap.add_argument("--model", default="topotein")
    ap.add_argument("--fpr-cap", type=float, default=0.01)
    ap.add_argument("--max-residues", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--deltafold", action="store_true", help="Pass --deltafold to extraction.")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    repo = str(Path(__file__).resolve().parent.parent.parent)
    extract = os.path.join(repo, "scripts", "utilities", "extract_embeddings.py")
    out_csv = args.out_csv or os.path.join(args.run_dir, "epoch_metrics_tuned.csv")
    out_png = args.out_png or os.path.join(args.run_dir, "epoch_metrics_tuned.png")
    tmp_emb = os.path.join(tempfile.gettempdir(), "eae_emb.pt")

    cks = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint_*epoch*.pth")),
                 key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
    if not cks:
        raise SystemExit(f"No epoch checkpoints in {args.run_dir}")
    nomburg = cc.load_nomburg_clusters()
    print(f"{len(cks)} checkpoints | {len(set(nomburg.values()))} Nomburg clusters | fpr_cap={args.fpr_cap}")

    rows = []
    for ck in cks:
        ep = int(re.search(r"epoch(\d+)", ck).group(1))
        if os.path.exists(tmp_emb):
            os.remove(tmp_emb)
        cmd = [sys.executable, extract, "--model", args.model, "--emb", ck,
               "--file-list", args.file_list, "--out", tmp_emb,
               "--max-residues", str(args.max_residues), "--batch_size", str(args.batch_size)]
        if args.deltafold:
            cmd.append("--deltafold")
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if not os.path.exists(tmp_emb):
            print(f"  ep{ep}: extract FAILED -> {r.stderr.strip()[-400:]}")
            continue
        ids, X = cc.load_embeddings(tmp_emb)
        fl = [nomburg.get(i, i) for i in ids]
        m = epoch_eval.evaluate(X, ids, fl, tm_cache={}, tune_epsilon=True, fpr_cap=args.fpr_cap)
        row = {"epoch": ep, "pair_fnr": m.get("pair_fnr"), "pair_fpr": m.get("pair_fpr"),
               "fragmentation": m.get("fragmentation"), "fusion": m.get("fusion"),
               "selected_epsilon": m.get("selected_epsilon"), "hdbscan_ari": m.get("hdbscan_ari"),
               "n_eval": m.get("n_eval"), "singleton_frac": m.get("singleton_frac")}
        rows.append(row)
        print(f"  ep{ep:>2}  FNR={row['pair_fnr']}  FPR={row['pair_fpr']}  "
              f"frag={row['fragmentation']}  fus={row['fusion']}  eps={row['selected_epsilon']}")
        sys.stdout.flush()

    keys = ["epoch", "pair_fnr", "pair_fpr", "fragmentation", "fusion",
            "selected_epsilon", "hdbscan_ari", "n_eval", "singleton_frac"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    E = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, (k, title) in zip(ax.flat, PANELS):
        y = [r[k] for r in rows]
        a.plot(E, y, marker="o", ms=3, lw=1.5)
        if k in ("pair_fpr",):
            a.axhline(args.fpr_cap, ls="--", c="r", lw=0.8, label=f"cap {args.fpr_cap}")
            a.legend(fontsize=8)
        if k in ("fragmentation", "fusion"):
            a.axhline(1.0, ls=":", c="grey", lw=0.8)   # ideal = 1
        a.set_title(title, fontsize=10)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(f"Auto-tuned-ε eval vs epoch — {os.path.basename(args.run_dir.rstrip('/'))} "
                 f"(FPR≤{args.fpr_cap:g}, n≈{rows[-1]['n_eval'] if rows else 0} eval)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_csv} and {out_png} ({len(rows)} epochs)")


if __name__ == "__main__":
    main()
