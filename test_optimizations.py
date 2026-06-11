#!/usr/bin/env python
"""Smoke test: verify optimized train_contrastive runs without errors."""
import os, sys, time
import torch
from train import get_split, PCCDataset, PROC_DIR, CLUSTER_TSV, CHECKPOINT_DIR, DEVICE
from contrastive_engine import StructuralAugmentations, NTXentLoss
from asymmetric_topotein import AsymmetricTopoNet
from train_contrastive import (
    contrastive_collate, extract_batch_keys, HardNegativeBatchSampler,
    worker_init_fn, pad_to_buckets, to_device
)
from torch.utils.data import DataLoader
import torch.optim as optim


def main():
    print("="*70)
    print("OPTIMIZATION TEST: train_contrastive with unified train/val")
    print("="*70)

    train_files, val_files = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split='phylo')
    print(f"\nDataset: {len(train_files)} train, {len(val_files)} val")

    # Build loaders using the new unified approach
    train_aug = StructuralAugmentations(jitter_sigma=0.3, use_crop=False)
    train_ds = PCCDataset(train_files, transform=train_aug)
    keys = extract_batch_keys(train_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt'))
    train_sampler = HardNegativeBatchSampler(keys, 64, seed=42, max_residues=4000)
    train_sampler.set_epoch(0)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=contrastive_collate,
                              num_workers=2, prefetch_factor=2, worker_init_fn=worker_init_fn)

    val_aug = StructuralAugmentations(jitter_sigma=0.0, drop_ratio_range=(0.0,0.0), mask_ratio=0.0, use_crop=False)
    val_ds = PCCDataset(val_files, transform=val_aug)
    val_keys = extract_batch_keys(val_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache_val.pt'))
    val_sampler = HardNegativeBatchSampler(val_keys, 64, seed=0, max_residues=4000,
                                           length_jitter=0.0, ratio_jitter=0.0)
    val_sampler.set_epoch(0)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=contrastive_collate,
                            num_workers=2, prefetch_factor=2, worker_init_fn=worker_init_fn)

    print(f"Train batches: {len(train_sampler)}")
    print(f"Val batches:   {len(val_sampler)} (was ~217 with old unbounded sampler)")

    model = AsymmetricTopoNet(scalar_dim=128, use_positional_encoding=False,
                              use_residue_features=False, use_3di_features=True,
                              edge_attn_softmax=True, dist_bias_gamma=0.1, detach_h3=True).to(DEVICE)
    criterion = NTXentLoss(temperature=0.1).to(DEVICE)

    try:
        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, fused=True)
        print(f"Optimizer: AdamW (fused) ✓")
    except (TypeError, RuntimeError):
        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        print(f"Optimizer: AdamW (eager) [fused unavailable]")

    print(f"\nTiming test (20 train steps + 1 val pass)...")
    model.train()
    t0_train = time.perf_counter()
    step_times = []
    for i, batch in enumerate(train_loader):
        if i >= 20: break
        if batch is None: continue
        feats = to_device(batch[0], DEVICE)
        model_in, real_B = pad_to_buckets(feats)
        t_step = time.perf_counter()
        with torch.enable_grad():
            z = model(model_in)[:real_B]
            B = z.size(0) // 2
            loss = criterion(z[:B], z[B:])
            loss.backward()
        opt.step()
        opt.zero_grad()
        step_times.append(time.perf_counter() - t_step)
        if DEVICE.type == 'mps': torch.mps.synchronize()

    train_elapsed = time.perf_counter() - t0_train
    print(f"  Training: {len(step_times)} steps in {train_elapsed:.1f}s "
          f"({sum(step_times)/len(step_times)*1000:.0f}ms/step median)")

    model.eval()
    t0_val = time.perf_counter()
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            feats = to_device(batch[0], DEVICE)
            model_in, real_B = pad_to_buckets(feats)
            z = model(model_in)[:real_B]
            B = z.size(0) // 2
            criterion(z[:B], z[B:]).item()
            val_steps += 1
            if val_steps >= 30: break
        if DEVICE.type == 'mps': torch.mps.synchronize()

    val_elapsed = time.perf_counter() - t0_val
    print(f"  Validation: {val_steps} steps in {val_elapsed:.1f}s "
          f"({val_elapsed/val_steps*1000:.0f}ms/step)")

    print("\n" + "="*70)
    print("✓ All systems operational. Ready for full training.")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  python train.py --model asymmetric --no-positional --no-residue \\")
    print(f"    --hard-neg-mining --split phylo \\")
    print(f"    --tm-cache ./checkpoints/tm_score_cache.pt --tm-aux-weight 0.1 \\")
    print(f"    --epochs 50 --batch_size 64")


if __name__ == '__main__':
    main()
