"""
Training-step benchmark
=======================
Measures where time goes in a single contrastive training step (data wait /
to_device / forward / backward) and the sustained per-step rate, under the SAME
data + model config as train.py. Use it to validate speed fixes (residue budget,
cleanup throttling) before committing to a 15-epoch run.

It is self-contained and guarded with `if __name__ == '__main__'` (required for
num_workers>0 on macOS spawn). Defaults mirror the recommended run:
    asymmetric, --no-positional --no-residue --no-3di, unsupervised InfoNCE,
    --hard-neg-beta 1.0, --hard-neg-mining, --split phylo, --max-residues 10000

Examples:
    # Default (the recommended fast config)
    python benchmark_step.py

    # Reproduce the OLD slow behavior: mining but NO residue cap
    python benchmark_step.py --max-residues 0

    # Sweep the budget
    python benchmark_step.py --max-residues 8000 --steps 25

    # Plain random batches (no length bucketing), fixed count
    python benchmark_step.py --no-mining
"""
import argparse
import os
import time
import statistics
import torch
from torch.utils.data import DataLoader

from train import get_split, PCCDataset, to_device, DEVICE, PROC_DIR, CLUSTER_TSV, CHECKPOINT_DIR
from contrastive_engine import StructuralAugmentations, NTXentLoss
from asymmetric_topotein import AsymmetricTopoNet
from topotein import Topotein
from train_contrastive import (contrastive_collate, extract_batch_keys,
                               HardNegativeBatchSampler, worker_init_fn, pad_to_buckets)


def sync():
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Per-step training benchmark")
    ap.add_argument('--model', default='asymmetric', choices=['asymmetric', 'topotein'])
    ap.add_argument('--batch_size', type=int, default=32, help="max proteins per batch (count cap)")
    ap.add_argument('--steps', type=int, default=20, help="steps to time (excludes 2 warmups)")
    ap.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                    help="DataLoader workers (default = train.py's cpu//2)")
    ap.add_argument('--split', default='phylo', choices=['phylo', 'cluster'])
    ap.add_argument('--no-mining', dest='mining', action='store_false',
                    help="disable hard-negative length-bucketed batching (use plain random batches)")
    ap.add_argument('--max-residues', dest='max_residues', type=int, default=4000,
                    help="ORIGINAL residue budget per batch (forward sees ~1.75x after 2 views); "
                         "0 = no cap (reproduces the old slow behavior)")
    ap.add_argument('--hard-neg-beta', dest='beta', type=float, default=1.0)
    ap.add_argument('--use-positional', dest='no_pe', action='store_false', default=True)
    ap.add_argument('--use-residue', dest='no_res', action='store_false', default=True)
    ap.add_argument('--use-3di', dest='no_3di', action='store_false', default=True)
    ap.add_argument('--max-seconds', dest='max_seconds', type=float, default=180,
                    help="wall-clock safety cap")
    ap.add_argument('--no-pad', dest='pad', action='store_false',
                    help="disable bucket-padding (reproduce the per-step kernel-cache bloat)")
    args = ap.parse_args()

    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 4) // 2)
    WARMUP = 2

    print("=" * 70)
    print("Deltafold step benchmark")
    print(f"  device={DEVICE}  cpu={os.cpu_count()}  num_workers={num_workers}")
    print(f"  model={args.model}  batch_size={args.batch_size}  split={args.split}")
    print(f"  mining={args.mining}  max_residues={args.max_residues if args.max_residues > 0 else 'NO CAP'}")
    print(f"  hard_neg_beta={args.beta}  PE_off={args.no_pe} residue_off={args.no_res} 3di_off={args.no_3di}")
    print("=" * 70)

    train_files, _ = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split=args.split)
    ds = PCCDataset(train_files, transform=StructuralAugmentations())

    if args.mining:
        keys = extract_batch_keys(train_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt'))
        budget = args.max_residues if args.max_residues > 0 else 10 ** 12  # 0 => effectively no cap
        sampler = HardNegativeBatchSampler(keys, args.batch_size, seed=42, max_residues=budget)
        sampler.set_epoch(0)
        batches_per_epoch = len(sampler)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=contrastive_collate,
                            num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None,
                            worker_init_fn=worker_init_fn if num_workers > 0 else None)
    else:
        batches_per_epoch = (len(train_files) + args.batch_size - 1) // args.batch_size
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=contrastive_collate,
                            num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None,
                            worker_init_fn=worker_init_fn if num_workers > 0 else None)

    model_kw = dict(use_positional_encoding=not args.no_pe,
                    use_residue_features=not args.no_res,
                    use_3di_features=not args.no_3di)
    model = (AsymmetricTopoNet if args.model == 'asymmetric' else Topotein)(scalar_dim=128, **model_kw).to(DEVICE)
    crit = NTXentLoss(temperature=0.1, hard_neg_beta=args.beta).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    print(f"~{batches_per_epoch} batches/epoch\n")
    print(f"{'step':>4} | {'prot':>4} | {'residues':>8} | {'edges':>7} | {'data':>6} | {'toDev':>6} | {'fwd':>6} | {'bwd':>6} | {'TOTAL':>6}")
    print("-" * 78)

    times = []
    start = time.perf_counter()
    step = 0
    it = iter(loader)
    while True:
        td = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t_data = time.perf_counter() - td
        if batch is None:
            continue

        t = time.perf_counter(); feats = to_device(batch[0], DEVICE); sync(); t_dev = time.perf_counter() - t
        if args.pad:
            feats, real_B = pad_to_buckets(feats)
        else:
            real_B = feats['rank3']['protein_size'].shape[0]
        N = feats['rank0']['aa'].shape[0]
        E = feats['rank1']['source'].numel()
        P = real_B

        t = time.perf_counter(); z = model(feats)[:real_B]; sync(); t_fwd = time.perf_counter() - t
        t = time.perf_counter()
        B = z.size(0) // 2
        loss = crit(z[:B], z[B:]); loss.backward(); opt.step(); opt.zero_grad(); sync()
        t_bwd = time.perf_counter() - t

        total = t_data + t_dev + t_fwd + t_bwd
        tag = "  <- warmup/compile" if step < WARMUP else ""
        print(f"{step:>4} | {P:>4} | {N:>8} | {E:>7} | {t_data:6.2f} | {t_dev:6.2f} | {t_fwd:6.2f} | {t_bwd:6.2f} | {total:6.2f}{tag}")
        if step >= WARMUP:
            times.append(total)
        step += 1
        if step >= args.steps + WARMUP or (time.perf_counter() - start) > args.max_seconds:
            break

    print("-" * 78)
    if times:
        med = statistics.median(times)
        print(f"sustained (excl {WARMUP} warmups): median {med:.2f}s/step | min {min(times):.2f} | max {max(times):.2f}")
        print(f"projected epoch time: {med * batches_per_epoch / 60:.1f} min  ({batches_per_epoch} batches x {med:.2f}s)")
    else:
        print("no timed steps recorded (increase --steps or --max-seconds)")
    print("done")


if __name__ == '__main__':
    main()
