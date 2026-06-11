"""GPU-free data-pipeline benchmark.

Measures how fast the EXACT training data pipeline (phylo split + hard-neg
length-bucketed sampler @ budget 4000 / cap 64 + StructuralAugmentations 2-view
transform + contrastive_collate) can PRODUCE batches, with no model and no MPS.
This isolates whether data loading is the per-step bottleneck, without contending
with the live training run for the GPU.

Reports: per-batch wall time with workers, the single-thread collate+transform
cost (num_workers=0), and the resulting proteins/residues per batch.
"""
import os, sys, time, statistics, argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from train import get_split, PCCDataset, PROC_DIR, CLUSTER_TSV, CHECKPOINT_DIR
from contrastive_engine import StructuralAugmentations
from train_contrastive import (contrastive_collate, extract_batch_keys,
                               HardNegativeBatchSampler, worker_init_fn)


def run(num_workers, steps, batch_size, budget):
    train_files, _ = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split='phylo')
    ds = PCCDataset(train_files, transform=StructuralAugmentations())
    keys = extract_batch_keys(train_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt'))
    sampler = HardNegativeBatchSampler(keys, batch_size, seed=42, max_residues=budget)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=contrastive_collate,
                        num_workers=num_workers,
                        prefetch_factor=2 if num_workers > 0 else None,
                        worker_init_fn=worker_init_fn if num_workers > 0 else None)

    print(f"\n[num_workers={num_workers}] timing {steps} batches (after 2 warmups)...")
    times, prots, ress = [], [], []
    it = iter(loader)
    for i in range(steps + 2):
        t = time.perf_counter()
        b = next(it)
        dt = time.perf_counter() - t
        if b is None:
            continue
        feats = b[0]
        P = feats['rank3']['protein_size'].shape[0]
        N = feats['rank0']['aa'].shape[0]
        if i >= 2:
            times.append(dt); prots.append(P); ress.append(N)
    med = statistics.median(times)
    print(f"  per-batch wait: median {med*1000:6.1f} ms | min {min(times)*1000:.0f} | max {max(times)*1000:.0f}")
    print(f"  batch shape   : ~{int(statistics.median(prots))} proteins (x2 views) | "
          f"~{int(statistics.median(ress))} residues in-forward")
    return med


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--budget', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    print("=" * 64)
    print(f"Data-pipeline benchmark (no model/GPU) budget={args.budget} cap={args.batch_size}")
    print(f"  host cpu={os.cpu_count()} (NOTE: live training is using ~5 workers now)")
    print("=" * 64)
    # single-thread cost first: pure per-batch collate+transform on this thread
    t0 = run(0, max(6, args.steps // 3), args.batch_size, args.budget)
    # then with workers, to see the effective overlapped rate
    tw = run(args.workers, args.steps, args.batch_size, args.budget)
    print("\n" + "=" * 64)
    print(f"single-thread per-batch : {t0*1000:.0f} ms  (CPU work to build one batch)")
    print(f"workers={args.workers} per-batch  : {tw*1000:.0f} ms  (what the train loop waits on if compute<this)")
    print("=" * 64)
