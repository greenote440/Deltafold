"""
Compile the overnight ablation sweep (plan_experimentation_v2 §5) into comparison
charts + a compact markdown summary, so conclusions can be drawn without re-reading
the raw gzipped logs.

Reads every run under checkpoints/sweep/<run>/training_log.jsonl.gz and writes:
  checkpoints/sweep/sweep_report.png   — 8-panel comparison figure (all runs overlaid)
  checkpoints/sweep/sweep_summary.md   — small per-run table (best/final metrics, health)
  checkpoints/sweep/sweep_summary.csv  — same, machine-readable

Panels: TM-rho/epoch (+random baseline), TM-recall/epoch, val-loss/epoch,
ARI/epoch, per-batch train loss, collapse-health (emb_std, mean_cos), and a
best-TM-rho leaderboard bar.

Usage:  python scripts/analysis/plot_sweep.py [checkpoints/sweep]
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')                      # file output, no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_sweep import read_log, RANDOM_BASELINE_RHO  # noqa: E402

COLLAPSE_STD = 0.02      # emb_std below this  -> collapse warning
COLLAPSE_COS = 0.90      # mean_cos above this -> collapse warning


def collect(sweep_root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(sweep_root, '*'))):
        if not os.path.isdir(d) or os.path.basename(d).startswith('.'):
            continue
        recs = read_log(os.path.join(d, 'training_log.jsonl.gz'))
        epochs = [r for r in recs if r.get('t') == 'epoch']
        steps = [r for r in recs if r.get('t') == 'step']
        if not epochs and not steps:
            continue
        n_ckpt = len(glob.glob(os.path.join(d, 'checkpoint_*_epoch*.pth')))
        runs[os.path.basename(d)] = {'epochs': epochs, 'steps': steps, 'n_ckpt': n_ckpt}
    return runs


def _series(epochs, key, sub=None):
    """(x_epochs, y) dropping None y's. sub='eval' digs into the eval dict."""
    xs, ys = [], []
    for r in epochs:
        v = (r.get('eval') or {}).get(key) if sub == 'eval' else r.get(key)
        if v is not None:
            xs.append(r['ep']); ys.append(v)
    return xs, ys


def make_figure(runs, out_png):
    names = list(runs)
    cmap = plt.get_cmap('tab20')
    colors = {n: cmap(i % 20) for i, n in enumerate(names)}

    fig, ax = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Overnight ablation sweep — plan v2 §5', fontsize=15, y=0.995)

    def plot_epoch(a, key, sub, title, ylabel):
        for n in names:
            x, y = _series(runs[n]['epochs'], key, sub)
            if x:
                a.plot(x, y, marker='o', ms=3, lw=1.3, color=colors[n], label=n)
        a.set_title(title); a.set_xlabel('epoch'); a.set_ylabel(ylabel); a.grid(alpha=0.3)

    plot_epoch(ax[0, 0], 'tm_rho', 'eval', 'TM-rho vs epoch (lower=better)', 'TM-rho')
    ax[0, 0].axhline(RANDOM_BASELINE_RHO, color='k', ls='--', lw=1,
                     label=f'random baseline {RANDOM_BASELINE_RHO}')
    plot_epoch(ax[0, 1], 'tm_recall', 'eval', 'TM-recall@close vs epoch (higher=better)', 'recall')
    plot_epoch(ax[1, 0], 'vloss', None, 'Validation loss vs epoch', 'val loss')
    plot_epoch(ax[1, 1], 'ari', None, 'Cluster ARI vs epoch (vs Foldseek; sanity only)', 'ARI')

    # per-batch training loss (logged step records), x = sequential record index
    for n in names:
        steps = runs[n]['steps']
        ys = [s['loss'] for s in steps if isinstance(s.get('loss'), (int, float))]
        if ys:
            ax[2, 0].plot(range(len(ys)), ys, lw=1, color=colors[n], label=n)
    ax[2, 0].set_title('Per-batch training loss'); ax[2, 0].set_xlabel('logged step #')
    ax[2, 0].set_ylabel('loss'); ax[2, 0].grid(alpha=0.3)
    if any(runs[n]['steps'] for n in names):
        ax[2, 0].set_yscale('log')

    # collapse health: emb_std and mean_cos over logged steps
    for n in names:
        steps = runs[n]['steps']
        stds = [(i, s['std']) for i, s in enumerate(steps) if isinstance(s.get('std'), (int, float))]
        coss = [(i, s['cos']) for i, s in enumerate(steps) if isinstance(s.get('cos'), (int, float))]
        if stds:
            ax[2, 1].plot(*zip(*stds), lw=1, color=colors[n], label=n)
        if coss:
            ax[3, 0].plot(*zip(*coss), lw=1, color=colors[n], label=n)
    ax[2, 1].axhline(COLLAPSE_STD, color='r', ls='--', lw=1, label=f'collapse <{COLLAPSE_STD}')
    ax[2, 1].set_title('Embedding std (collapse if low)'); ax[2, 1].set_xlabel('logged step #')
    ax[2, 1].set_ylabel('emb_std'); ax[2, 1].grid(alpha=0.3)
    ax[3, 0].axhline(COLLAPSE_COS, color='r', ls='--', lw=1, label=f'collapse >{COLLAPSE_COS}')
    ax[3, 0].set_title('Mean off-diagonal cosine (collapse if high)'); ax[3, 0].set_xlabel('logged step #')
    ax[3, 0].set_ylabel('mean_cos'); ax[3, 0].grid(alpha=0.3)

    # leaderboard: best (most negative) TM-rho per run
    best = []
    for n in names:
        _, ys = _series(runs[n]['epochs'], 'tm_rho', 'eval')
        if ys:
            best.append((n, min(ys)))
    best.sort(key=lambda t: t[1], reverse=True)   # worst at top, best at bottom of barh
    if best:
        bn, bv = zip(*best)
        bar_colors = ['tab:green' if v < RANDOM_BASELINE_RHO else 'tab:gray' for v in bv]
        ax[3, 1].barh(bn, bv, color=bar_colors)
        ax[3, 1].axvline(RANDOM_BASELINE_RHO, color='k', ls='--', lw=1)
        ax[3, 1].set_title('Best TM-rho per run (green = beats random baseline)')
        ax[3, 1].set_xlabel('best TM-rho')
    ax[3, 1].grid(alpha=0.3, axis='x')

    # single shared legend (runs) below everything
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f"Wrote {out_png}")


def make_summary(runs, out_md, out_csv):
    rows = []
    for n, d in runs.items():
        eps = d['epochs']
        _, rhos = _series(eps, 'tm_rho', 'eval')
        _, recs = _series(eps, 'tm_recall', 'eval')
        _, vls = _series(eps, 'vloss', None)
        _, aris = _series(eps, 'ari', None)
        stds = [s['std'] for s in d['steps'] if isinstance(s.get('std'), (int, float))]
        coss = [s['cos'] for s in d['steps'] if isinstance(s.get('cos'), (int, float))]
        best_rho = min(rhos) if rhos else None
        best_ep = eps[rhos.index(best_rho)]['ep'] if rhos else None
        collapse = (min(stds) < COLLAPSE_STD if stds else False) or (max(coss) > COLLAPSE_COS if coss else False)
        rows.append({
            'run': n, 'n_ep': len(eps), 'n_ckpt': d['n_ckpt'],
            'best_rho': best_rho, 'best_ep': best_ep,
            'final_rho': rhos[-1] if rhos else None,
            'best_recall': max(recs) if recs else None,
            'final_vloss': vls[-1] if vls else None,
            'final_ari': aris[-1] if aris else None,
            'min_std': min(stds) if stds else None,
            'max_cos': max(coss) if coss else None,
            'collapse': 'YES' if collapse else '',
            'beats_baseline': 'YES' if (best_rho is not None and best_rho < RANDOM_BASELINE_RHO) else '',
        })
    rows.sort(key=lambda r: (r['best_rho'] is None, r['best_rho'] if r['best_rho'] is not None else 0))

    cols = ['run', 'n_ep', 'n_ckpt', 'best_rho', 'best_ep', 'final_rho', 'best_recall',
            'final_vloss', 'final_ari', 'min_std', 'max_cos', 'collapse', 'beats_baseline']

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ('' if v is None else str(v))

    with open(out_md, 'w') as f:
        f.write(f"# Sweep summary (random-init baseline TM-rho = {RANDOM_BASELINE_RHO})\n\n")
        f.write("Sorted by best val TM-rho (most negative = best). "
                "`beats_baseline`=YES means trained embedding adds value over the features.\n\n")
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in rows:
            f.write('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |\n')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_md} and {out_csv}")


def plot_single_run(rundir, out_png=None):
    """Full per-epoch metric panel for ONE run dir (its training_log.jsonl.gz),
    including the §6.1 health gate + Mod 3/4 metrics now logged in the eval dict."""
    recs = read_log(os.path.join(rundir, 'training_log.jsonl.gz'))
    epochs = [r for r in recs if r.get('t') == 'epoch']
    steps = [r for r in recs if r.get('t') == 'step']
    if not epochs:
        print(f"No epoch records in {rundir}")
        return
    name = os.path.basename(rundir.rstrip('/'))
    ep = [r['ep'] for r in epochs]

    def top(k):    # series from the top-level epoch record
        return [r.get(k) for r in epochs]

    def ev(k):     # series from the nested eval dict
        return [(r.get('eval') or {}).get(k) for r in epochs]

    def line(a, series_specs, title, ylabel, hline=None):
        for ys, lab in series_specs:
            xs = [e for e, y in zip(ep, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            if xs:
                a.plot(xs, yv, marker='o', ms=3, lw=1.4, label=lab)
        if hline is not None:
            a.axhline(hline, color='k', ls='--', lw=1, label=f'baseline {hline}')
        a.set_title(title); a.set_xlabel('epoch'); a.set_ylabel(ylabel)
        a.grid(alpha=0.3)
        if len(series_specs) > 1 or hline is not None:
            a.legend(fontsize=7)

    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle(f'MoCo run [{name}] — per-epoch metrics (plan §6)', fontsize=14)
    line(ax[0, 0], [(ev('tm_rho'), 'tm_rho')], 'TM-rho (primary, lower=better)', 'rho',
         hline=RANDOM_BASELINE_RHO)
    line(ax[0, 1], [(ev('hdbscan_ari'), 'ARI'), (ev('perm_ari'), 'perm-null')],
         'ARI vs permutation null (Mod 3)', 'ARI')
    line(ax[0, 2], [(ev('effective_rank'), 'eff_rank')],
         'Effective rank (§6.1 collapse gate; higher=healthier)', 'eff_rank')
    line(ax[1, 0], [(ev('mean_cos'), 'mean_cos'), (ev('emb_std'), 'emb_std')],
         'Collapse health (mean_cos low / emb_std high = good)', 'value')
    line(ax[1, 1], [(ev('uniformity'), 'uniformity')],
         'Uniformity (§6.1; more negative = better spread)', 'uniformity')
    line(ax[1, 2], [(ev('tm_alignment'), 'alignment'), (ev('tm_recall'), 'recall')],
         'TM alignment (positives) & recall@close', 'value')
    line(ax[2, 0], [(ev('v_measure'), 'V-measure'), (ev('homogeneity'), 'homogeneity'),
                    (ev('completeness'), 'completeness'), (ev('fowlkes_mallows'), 'Fowlkes-Mallows')],
         'Directional clustering vs Foldseek (Mod 4)', 'score')
    line(ax[2, 1], [(ev('fragmentation'), 'fragmentation'), (ev('fusion'), 'fusion')],
         'Fragmentation / fusion (Mod 4; ->1 = aligned)', 'avg #clusters')
    line(ax[2, 2], [(top('loss'), 'train loss'), (top('vloss'), 'val loss')],
         'Loss', 'loss')

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = out_png or os.path.join(rundir, 'metrics.png')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f"Wrote {out_png}  ({len(epochs)} epochs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_root', nargs='?', default='checkpoints/sweep')
    ap.add_argument('--run', default=None,
                    help="Plot a SINGLE run directory (its full per-epoch metric panel) "
                         "instead of a multi-run sweep comparison.")
    args = ap.parse_args()
    if args.run:
        plot_single_run(args.run)
        return
    runs = collect(args.sweep_root)
    if not runs:
        print(f"No runs with logs under {args.sweep_root}")
        return
    print(f"{len(runs)} runs: {', '.join(runs)}")
    make_figure(runs, os.path.join(args.sweep_root, 'sweep_report.png'))
    make_summary(runs, os.path.join(args.sweep_root, 'sweep_summary.md'),
                 os.path.join(args.sweep_root, 'sweep_summary.csv'))


if __name__ == '__main__':
    main()
