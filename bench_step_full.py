"""Faithful full-step benchmark.

Replicates the EXACT train_contrastive step under the live config (asymmetric,
no-PE/no-residue, 3di ON, supervised SupCon, tm-aux 0.1, mining budget 4000,
cap 64, cleanup_every 10) and times each component separately so we can see what
makes the real step ~2.5s when the bare model fwd+bwd is only ~0.7s.

Components timed: fwd | supcon-loss | tm-aux | loss.item() | bwd | clip+opt |
collapse(10) | reclaim(10).
"""
import os, sys, time, gc, statistics, random
import torch
from torch.utils.data import DataLoader

from train import (get_split, PCCDataset, to_device, extract_accession,
                   DEVICE, PROC_DIR, CLUSTER_TSV, CHECKPOINT_DIR)
from train_contrastive import _phys_footprint_gb
from contrastive_engine import StructuralAugmentations, NTXentLoss
from asymmetric_topotein import AsymmetricTopoNet
from train_contrastive import (contrastive_collate, extract_batch_keys,
                               HardNegativeBatchSampler, worker_init_fn, pad_to_buckets,
                               supervised_ntxent_loss, tm_score_aux_loss_cached,
                               collapse_metrics)

CLEANUP_EVERY = 10
TM_AUX_W = 0.1
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def sync():
    if DEVICE.type == 'mps': torch.mps.synchronize()
    elif DEVICE.type == 'cuda': torch.cuda.synchronize()


def main():
    num_workers = 5
    train_files, _ = get_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42, split='phylo')
    ds = PCCDataset(train_files, transform=StructuralAugmentations(jitter_sigma=0.3, use_crop=False))
    keys = extract_batch_keys(train_files, os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt'))
    sampler = HardNegativeBatchSampler(keys, 64, seed=42, max_residues=4000)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=contrastive_collate,
                        num_workers=num_workers, prefetch_factor=2, worker_init_fn=worker_init_fn)

    acc_to_cluster = {}
    with open(CLUSTER_TSV) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                acc_to_cluster[extract_accession(p[1])] = extract_accession(p[0])
    tm_cache = torch.load(os.path.join(CHECKPOINT_DIR, 'tm_score_cache.pt'), weights_only=False)

    model = AsymmetricTopoNet(scalar_dim=128, use_positional_encoding=False,
                              use_residue_features=False, use_3di_features=True).to(DEVICE)
    crit = NTXentLoss(temperature=0.1, hard_neg_beta=0.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    print(f"device={DEVICE} workers={num_workers} | replicating real step, {STEPS} timed (2 warmup)")
    hdr = f"{'st':>3} {'fwd':>6} {'supcon':>6} {'tmaux':>6} {'item':>6} {'bwd':>6} {'opt':>6} {'coll':>6} {'reclm':>6} | {'TOT':>6}"
    print(hdr); print('-' * len(hdr))

    agg = {k: [] for k in ['fwd', 'supcon', 'tmaux', 'item', 'bwd', 'opt', 'coll', 'reclm', 'tot']}
    it = iter(loader)
    done = 0
    for step in range(STEPS + 2):
        try: batch = next(it)
        except StopIteration: break
        if batch is None: continue
        done += 1
        batch_data, paths = batch
        feats = to_device(batch_data, DEVICE)
        model_in, real_B = pad_to_buckets(feats)

        t = time.perf_counter(); z = model(model_in); z = z[:real_B]; sync(); t_fwd = time.perf_counter() - t

        t = time.perf_counter()
        cluster_ids = [acc_to_cluster.get(extract_accession(os.path.basename(p)), p) for p in paths]
        label_map = {cid: i for i, cid in enumerate(set(cluster_ids))}
        labels = torch.tensor([label_map[c] for c in cluster_ids], device=DEVICE)
        loss = supervised_ntxent_loss(z, labels, temperature=0.1, hard_neg_beta=0.0)
        sync(); t_sup = time.perf_counter() - t

        t = time.perf_counter()
        tml = tm_score_aux_loss_cached(z, paths, tm_cache)
        if tml is not None: loss = loss + TM_AUX_W * tml
        sync(); t_tm = time.perf_counter() - t

        t = time.perf_counter()
        if done % 10 == 0:
            collapse_metrics(z)
        sync(); t_coll = time.perf_counter() - t

        t = time.perf_counter(); lv = loss.item(); t_item = time.perf_counter() - t

        t = time.perf_counter(); loss.backward(); sync(); t_bwd = time.perf_counter() - t

        t = time.perf_counter()
        params = [p for p in model.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_value_(params, clip_value=1.0)
        opt.step(); opt.zero_grad(); sync(); t_opt = time.perf_counter() - t

        t = time.perf_counter()
        if DEVICE.type == 'mps' and CLEANUP_EVERY > 0 and (done % CLEANUP_EVERY == 0):
            gc.collect(); torch.mps.empty_cache(); gc.collect()
        t_reclm = time.perf_counter() - t

        tot = t_fwd + t_sup + t_tm + t_item + t_bwd + t_opt + t_coll + t_reclm
        if step >= 2:
            for k, v in zip(['fwd','supcon','tmaux','item','bwd','opt','coll','reclm','tot'],
                            [t_fwd,t_sup,t_tm,t_item,t_bwd,t_opt,t_coll,t_reclm,tot]):
                agg[k].append(v)
        # rolling window: median of last 25 totals + current footprint -> catches drift
        if step >= 2 and (step % 25 == 0 or step == STEPS + 1):
            win = agg['tot'][-25:]
            print(f"step {step:>4} | last-25 median {statistics.median(win):.2f}s/step "
                  f"(fwd {statistics.median(agg['fwd'][-25:]):.2f} bwd {statistics.median(agg['bwd'][-25:]):.2f}) "
                  f"| footprint {_phys_footprint_gb():.1f}GB")
        del feats, model_in, z, loss, batch
        if step >= STEPS + 1:
            break

    print('-' * len(hdr))
    print("MEDIAN per-component (s/step):")
    for k in ['fwd','supcon','tmaux','item','bwd','opt','coll','reclm','tot']:
        print(f"  {k:>7}: {statistics.median(agg[k]):.3f}")
    print(f"\nprojected epoch: {statistics.median(agg['tot'])*4345/60:.0f} min")


if __name__ == '__main__':
    main()
