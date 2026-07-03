"""
Summarize the overnight ablation sweep (plan_experimentation_v2 §5) for quick
morning analysis. Reads every run under checkpoints/sweep/<run>/ and reports, per
run: the config, the per-epoch metrics (val loss / ARI / TM-rho / TM-recall), the
collapse-health envelope, and the per-batch loss trajectory — then a cross-run
leaderboard sorted by best val TM-rho against the random-init feature baseline.

Each run directory is expected to contain:
  config.json                 — what the run was (written by run_overnight_sweep.sh)
  training_log.jsonl.gz       — 'epoch', 'step', 'collapse', 'restart' records
  checkpoint_*_epoch*.pth     — one checkpoint per epoch

Usage:
  python scripts/analysis/summarize_sweep.py [checkpoints/sweep]
  python scripts/analysis/summarize_sweep.py --csv checkpoints/sweep/summary.csv
"""
import argparse
import csv
import glob
import gzip
import json
import os

# The random-init / feature baseline measured this session (val Spearman rho on
# the corrected sub-base). A trained config only adds value if it beats this.
RANDOM_BASELINE_RHO = -0.48


def read_log(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with gzip.open(path, 'rb') as fz:
        for line in fz:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def summarize_run(rundir):
    name = os.path.basename(rundir.rstrip('/'))
    cfg = {}
    cfgp = os.path.join(rundir, 'config.json')
    if os.path.exists(cfgp):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            pass
    recs = read_log(os.path.join(rundir, 'training_log.jsonl.gz'))
    epochs = [r for r in recs if r.get('t') == 'epoch']
    steps = [r for r in recs if r.get('t') == 'step']
    collapses = [r for r in recs if r.get('t') == 'collapse']
    ckpts = sorted(glob.glob(os.path.join(rundir, 'checkpoint_*_epoch*.pth')))

    def epoch_rho(r):
        return (r.get('eval') or {}).get('tm_rho')

    rhos = [(r['ep'], epoch_rho(r)) for r in epochs if epoch_rho(r) is not None]
    best_rho_ep, best_rho = min(rhos, key=lambda t: t[1]) if rhos else (None, None)

    return {
        'name': name, 'cfg': cfg, 'epochs': epochs, 'steps': steps,
        'collapses': collapses, 'n_ckpt': len(ckpts),
        'best_rho': best_rho, 'best_rho_ep': best_rho_ep,
    }


def print_run(s):
    print(f"\n[{s['name']}]  flags: {s['cfg'].get('flags', '?')}")
    print(f"  {len(s['epochs'])} epochs · {s['n_ckpt']} checkpoints"
          + (f" · COLLAPSE EVENTS: {len(s['collapses'])}" if s['collapses'] else ""))
    print(f"  {'ep':>3} {'vloss':>8} {'ari':>7} {'tm_rho':>8} {'tm_recall':>10} {'emb_std':>8} {'mean_cos':>8}")
    for r in s['epochs']:
        ev = r.get('eval') or {}
        # nearest step-health to this epoch end (last step record of the epoch)
        ep_steps = [st for st in s['steps'] if st.get('ep') == r['ep']]
        std = ep_steps[-1].get('std') if ep_steps else None
        cos = ep_steps[-1].get('cos') if ep_steps else None
        print(f"  {r['ep']:>3} {_fmt(r.get('vloss')):>8} {_fmt(r.get('ari')):>7} "
              f"{_fmt(ev.get('tm_rho')):>8} {_fmt(ev.get('tm_recall')):>10} "
              f"{_fmt(std):>8} {_fmt(cos):>8}")
    if s['steps']:
        losses = [st['loss'] for st in s['steps'] if isinstance(st.get('loss'), (int, float))]
        if losses:
            print(f"  per-batch loss: first={losses[0]:.4f} min={min(losses):.4f} "
                  f"last={losses[-1]:.4f}  ({len(losses)} step records)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_root', nargs='?', default='checkpoints/sweep')
    ap.add_argument('--csv', default=None, help="Also write a per-epoch CSV across all runs.")
    args = ap.parse_args()

    rundirs = sorted(d for d in glob.glob(os.path.join(args.sweep_root, '*')) if os.path.isdir(d))
    if not rundirs:
        print(f"No run directories under {args.sweep_root}")
        return

    summaries = [summarize_run(d) for d in rundirs]
    for s in summaries:
        print_run(s)

    # Cross-run leaderboard by best val TM-rho (most negative = best).
    print("\n==================== LEADERBOARD (best val TM-rho) ====================")
    print(f"  random-init / feature baseline rho = {RANDOM_BASELINE_RHO:+.3f}  (must beat this)")
    ranked = sorted([s for s in summaries if s['best_rho'] is not None], key=lambda s: s['best_rho'])
    for s in ranked:
        beats = "  <-- beats baseline" if s['best_rho'] < RANDOM_BASELINE_RHO else ""
        flag = "  [COLLAPSE]" if s['collapses'] else ""
        print(f"  {s['name']:<14} best_rho={s['best_rho']:+.4f} @ep{s['best_rho_ep']}{beats}{flag}")
    no_rho = [s['name'] for s in summaries if s['best_rho'] is None]
    if no_rho:
        print(f"  (no TM-rho logged — was --tm-cache passed?): {', '.join(no_rho)}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run', 'epoch', 'vloss', 'ari', 'tm_rho', 'tm_recall'])
            for s in summaries:
                for r in s['epochs']:
                    ev = r.get('eval') or {}
                    w.writerow([s['name'], r['ep'], r.get('vloss'), r.get('ari'),
                                ev.get('tm_rho'), ev.get('tm_recall')])
        print(f"\nPer-epoch CSV -> {args.csv}")


if __name__ == '__main__':
    main()
