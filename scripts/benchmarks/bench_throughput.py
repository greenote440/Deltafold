"""Throughput profiler for the asymmetric contrastive training step.

Replicates the EXACT live train_contrastive step (asymmetric model, hard-neg
mining, supervised SupCon + optional TM-aux, bucket padding, governor reclaim
cadence) and logs EVERY per-batch metric to a JSONL file so an optimization
session has the full picture, not just stdout medians.

Why these metrics (for optimizing THROUGHPUT specifically):

  data_wait   time the loop is BLOCKED on next(it). If this >> compute, the GPU is
              starved and the fix is the data pipeline (workers / prefetch / collate
              / transform cost), NOT the model. THIS is the metric bench_step_full
              omits and the single most important throughput signal.
  fwd/bwd     model compute. The theoretical floor; everything else is overhead.
  supcon/tmaux/item/opt/clip/collapse/reclaim
              per-phase overhead. item() and any .item()/sync force a CPU<->GPU
              stall; reclaim (gc+empty_cache) can be surprisingly expensive on MPS.
  B,N,E,S     batch composition (proteins, residues, edges, SSEs). Throughput is
              residues/sec, so steps/sec alone is misleading when batch size varies.
  *_pad,waste padding overhead from pad_to_buckets — wasted compute on dummy
              nodes/SSEs/proteins. High waste => tune bucket sizes.
  fp_gb       physical footprint per batch — catches the cross-epoch leak/drift that
              eventually triggers swap (the real throughput cliff).

Run (inside ml_env):
  python bench_throughput.py --steps 100 --out checkpoints/bench_throughput.jsonl

The per-batch JSONL + the printed summary together tell you where to spend effort.
"""
import argparse
import gc
import json
import os
import sys
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from train import (get_split, PCCDataset, to_device, extract_accession,
                   DEVICE, PROC_DIR, CLUSTER_TSV, CHECKPOINT_DIR)
from contrastive_engine import (StructuralAugmentations, supervised_ntxent_loss,
                                soft_supcon_loss, build_tm_matrix,
                                tm_score_aux_loss_cached, collapse_metrics)
from contrastive_data import (contrastive_collate, extract_batch_keys,
                              HardNegativeBatchSampler, worker_init_fn, pad_to_buckets)
from contrastive_memory import _phys_footprint_gb
from asymmetric_topotein import AsymmetricTopoNet


def sync():
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Per-batch throughput profiler (asymmetric contrastive)")
    ap.add_argument('--steps', type=int, default=100, help="Timed batches (after warmup).")
    ap.add_argument('--warmup', type=int, default=3, help="Warmup batches (excluded from stats).")
    ap.add_argument('--out', type=str, default=os.path.join(CHECKPOINT_DIR, 'bench_throughput.jsonl'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--batch_size', type=int, default=64, help="Protein count cap per batch.")
    ap.add_argument('--budget', type=int, default=4000, help="max_residues per batch (pre-2-view).")
    ap.add_argument('--split', type=str, default='phylo', choices=['cluster', 'phylo'])
    ap.add_argument('--cleanup-every', dest='cleanup_every', type=int, default=10)
    ap.add_argument('--tm-aux-weight', dest='tm_aux_weight', type=float, default=0.1,
                    help="0 disables the TM-aux loss (and TM-cache load unless --soft-supcon).")
    ap.add_argument('--soft-supcon', dest='soft_supcon', action='store_true',
                    help="Use TM-weighted soft SupCon instead of binary SupCon.")
    ap.add_argument('--tm-cache', dest='tm_cache', type=str,
                    default=os.path.join(CHECKPOINT_DIR, 'tm_score_cache.pt'))
    ap.add_argument('--no-pad-buckets', dest='pad_buckets', action='store_false')
    args = ap.parse_args()

    print("=" * 72)
    print(f"Throughput profiler | device={DEVICE} workers={args.workers} "
          f"split={args.split} budget={args.budget} cap={args.batch_size}")
    print(f"  loss={'soft-supcon' if args.soft_supcon else 'supcon'} "
          f"tm_aux={args.tm_aux_weight} pad_buckets={args.pad_buckets} "
          f"cleanup_every={args.cleanup_every}")
    print(f"  logging per-batch metrics -> {args.out}")
    print("=" * 72)

    # --- data pipeline (identical to the live train loader) ---
    train_files, _ = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split=args.split)
    ds = PCCDataset(train_files, transform=StructuralAugmentations(jitter_sigma=0.3, use_crop=False))
    keys = extract_batch_keys(train_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt'))
    sampler = HardNegativeBatchSampler(keys, args.batch_size, seed=42, max_residues=args.budget)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=contrastive_collate,
                        num_workers=args.workers, prefetch_factor=2,
                        worker_init_fn=worker_init_fn, pin_memory=DEVICE.type == 'cuda')

    # --- labels + optional TM cache ---
    acc_to_cluster = {}
    with open(CLUSTER_TSV) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                acc_to_cluster[extract_accession(p[1])] = extract_accession(p[0])

    tm_cache = None
    if args.soft_supcon or args.tm_aux_weight > 0.0:
        if os.path.exists(args.tm_cache):
            tm_cache = torch.load(args.tm_cache, weights_only=False)
            print(f"Loaded TM cache: {len(tm_cache)} pairs")
        else:
            print(f"[!] TM cache {args.tm_cache} missing; TM-aux/soft-supcon disabled for this run.")
            args.tm_aux_weight = 0.0
            args.soft_supcon = False

    # --- model (live asymmetric arch: §5.3/5.4/5.7 fixes ON, no-PE/no-residue) ---
    model = AsymmetricTopoNet(scalar_dim=128, use_positional_encoding=False,
                              use_residue_features=False, use_3di_features=True,
                              edge_attn_softmax=True, dist_bias_gamma=0.1,
                              detach_h3=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.train()

    # phases we time per batch
    phases = ['data_wait', 'fwd', 'loss', 'tmaux', 'collapse', 'item', 'bwd', 'opt', 'reclaim']
    agg = {k: [] for k in phases + ['compute', 'wall', 'res_per_s']}
    fh = open(args.out, 'w')

    it = iter(loader)
    done = 0
    total = args.steps + args.warmup
    print(f"\nrunning {total} batches ({args.warmup} warmup)...\n")
    for step in range(total):
        # --- data wait: time blocked fetching the next batch (GPU-starvation signal) ---
        t = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            print("loader exhausted early"); break
        t_data = time.perf_counter() - t
        if batch is None:
            continue
        done += 1
        batch_data, paths = batch

        feats = to_device(batch_data, DEVICE)
        # batch composition (real, pre-pad)
        N = feats['rank0']['aa'].shape[0]
        K = feats['rank1']['source'].shape[1]
        E = N * K
        S = feats['rank2_features'].shape[0]
        B = feats['rank3']['protein_size'].shape[0]

        if args.pad_buckets:
            model_in, real_B = pad_to_buckets(feats)
            N_pad = model_in['rank0']['aa'].shape[0]
            S_pad = model_in['rank2_features'].shape[0]
            B_pad = model_in['rank3']['protein_size'].shape[0]
            waste = round(1.0 - N / max(N_pad, 1), 4)
        else:
            model_in, real_B = feats, B
            N_pad, S_pad, B_pad, waste = N, S, B, 0.0

        t = time.perf_counter(); z = model(model_in); z = z[:real_B]; sync(); t_fwd = time.perf_counter() - t

        # --- loss (supervised or soft SupCon) ---
        t = time.perf_counter()
        cluster_ids = [acc_to_cluster.get(extract_accession(os.path.basename(p)), p) for p in paths]
        label_map = {cid: i for i, cid in enumerate(set(cluster_ids))}
        labels = torch.tensor([label_map[c] for c in cluster_ids], device=DEVICE)
        if args.soft_supcon and tm_cache is not None:
            tm_matrix = build_tm_matrix(paths, tm_cache, z.device)
            loss = soft_supcon_loss(z, labels, tm_matrix, temperature=0.1)
        else:
            loss = supervised_ntxent_loss(z, labels, temperature=0.1, hard_neg_beta=0.0)
        sync(); t_loss = time.perf_counter() - t

        # --- TM-aux ---
        t = time.perf_counter()
        if args.tm_aux_weight > 0.0 and tm_cache is not None:
            tml = tm_score_aux_loss_cached(z, paths, tm_cache)
            if tml is not None:
                loss = loss + args.tm_aux_weight * tml
        sync(); t_tm = time.perf_counter() - t

        # --- collapse metric (every 10 steps, as in the live loop) ---
        t = time.perf_counter()
        if done % 10 == 0:
            collapse_metrics(z)
        sync(); t_coll = time.perf_counter() - t

        t = time.perf_counter(); lv = loss.item(); t_item = time.perf_counter() - t

        t = time.perf_counter(); loss.backward(); sync(); t_bwd = time.perf_counter() - t

        t = time.perf_counter()
        params = [p for p in model.parameters() if p.grad is not None]
        if DEVICE.type == 'mps':
            torch.nn.utils.clip_grad_value_(params, clip_value=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step(); opt.zero_grad(); sync(); t_opt = time.perf_counter() - t

        t = time.perf_counter()
        if DEVICE.type == 'mps' and args.cleanup_every > 0 and (done % args.cleanup_every == 0):
            gc.collect(); torch.mps.empty_cache(); gc.collect()
        t_reclm = time.perf_counter() - t

        compute = t_fwd + t_loss + t_tm + t_coll + t_item + t_bwd + t_opt + t_reclm
        wall = t_data + compute
        res_per_s = (2 * N) / wall if wall > 0 else 0.0   # 2*N: both augmented views

        rec = {
            'step': done, 'loss': round(lv, 4),
            'data_wait': round(t_data, 4), 'fwd': round(t_fwd, 4), 'loss_t': round(t_loss, 4),
            'tmaux': round(t_tm, 4), 'collapse': round(t_coll, 4), 'item': round(t_item, 4),
            'bwd': round(t_bwd, 4), 'opt': round(t_opt, 4), 'reclaim': round(t_reclm, 4),
            'compute': round(compute, 4), 'wall': round(wall, 4),
            'B': B, 'N': N, 'E': E, 'S': S, 'K': K,
            'N_pad': N_pad, 'S_pad': S_pad, 'B_pad': B_pad, 'pad_waste': waste,
            'res_per_s': round(res_per_s, 1),
            'fp_gb': round(_phys_footprint_gb(), 2),
        }
        fh.write(json.dumps(rec, separators=(',', ':')) + '\n'); fh.flush()

        if step >= args.warmup:
            timed = {'data_wait': t_data, 'fwd': t_fwd, 'loss': t_loss, 'tmaux': t_tm,
                     'collapse': t_coll, 'item': t_item, 'bwd': t_bwd, 'opt': t_opt,
                     'reclaim': t_reclm, 'compute': compute, 'wall': wall, 'res_per_s': res_per_s}
            for k, v in timed.items():
                agg[k].append(v)

        if step % 20 == 0 or step == total - 1:
            print(f"step {step:>4} | wall {wall:.2f}s (data {t_data:.2f} fwd {t_fwd:.2f} bwd {t_bwd:.2f}) "
                  f"| B={B} N={N} pad_waste={waste:.0%} | {_phys_footprint_gb():.1f}GB")

        del feats, model_in, z, loss, batch

    fh.close()

    # ---------------- summary ----------------
    def med(k): return statistics.median(agg[k]) if agg[k] else 0.0
    def p90(k):
        if not agg[k]: return 0.0
        s = sorted(agg[k]); return s[min(len(s) - 1, int(0.9 * len(s)))]

    wall_med = med('wall')
    print("\n" + "=" * 72)
    print(f"SUMMARY over {len(agg['wall'])} timed batches  (median | p90, seconds/step)")
    print("=" * 72)
    for k in ['data_wait', 'fwd', 'loss', 'tmaux', 'collapse', 'item', 'bwd', 'opt', 'reclaim', 'compute', 'wall']:
        share = (med(k) / wall_med * 100) if wall_med else 0.0
        print(f"  {k:>10}: {med(k):.3f} | {p90(k):.3f}   ({share:4.1f}% of wall)")

    data_med, compute_med = med('data_wait'), med('compute')
    print("-" * 72)
    print(f"  throughput : {med('res_per_s'):.0f} residues/s (median) | "
          f"{1.0/wall_med if wall_med else 0:.2f} steps/s")
    print(f"  est. epoch : {wall_med * len(sampler) / 60:.0f} min  ({len(sampler)} batches/epoch)")
    # GPU-starvation diagnosis: with workers, data_wait should be ~0 if compute hides it.
    if data_med > 0.25 * compute_med:
        print(f"  [!] DATA-BOUND: data_wait ({data_med:.2f}s) is large vs compute ({compute_med:.2f}s) "
              f"-> optimize the data pipeline (workers/prefetch/collate). See bench_data.py.")
    else:
        print(f"  [ok] compute-bound: data_wait ({data_med:.2f}s) hidden behind compute ({compute_med:.2f}s) "
              f"-> optimize the model/loss/step.")
    fp = agg.get('res_per_s')
    print(f"  full log   : {args.out}  (one JSON record per batch)")
    print("=" * 72)


if __name__ == '__main__':
    main()
