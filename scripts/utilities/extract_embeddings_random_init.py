"""
Extract embeddings from a FRESH, randomly-initialized model (no checkpoint loaded)
— the epoch-0 / random-weights control for plan_experimentation_v2 (témoin:
"embedding aléatoire / modèle non entraîné"). Mirrors extract_embeddings.py
except it skips the checkpoint *weight* load.

For a fair baseline the random model must have the IDENTICAL architecture to the
trained run (same feature flags, same edge_attn_softmax/dist_bias_gamma/detach_h3,
same dist_encoding/rbf_dim — these change the forward pass and even the layer
dims). Pass ``--config-from <trained_checkpoint.pth>`` to copy that checkpoint's
``model_config`` and build the same model with random weights. Scope to the
corrected sub-base with ``--file-list`` so the metrics line up with the trained
eval (epoch_eval.py / the tm_score_cache).

Example
-------
    python scripts/utilities/extract_embeddings_random_init.py \
        --config-from checkpoints/checkpoint_contrastive_asymmetric_epoch001.pth \
        --file-list data/subbase_corrected_train.txt data/subbase_corrected_val.txt \
        --max-residues 6000 \
        --out data/emb_random_init_corrected.pt
    # then:
    python scripts/analysis/epoch_eval.py --emb data/emb_random_init_corrected.pt \
        --tm-cache checkpoints/tm_score_cache.pt --log checkpoints/epoch_eval.csv --epoch 0
"""
import sys
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from extract_embeddings import (
    ExtractionDataset, collate_with_names, LengthBudgetSampler, DEVICE, PROC_DIR,
    _downsampled_files,
)
from train_contrastive import pad_to_buckets, extract_batch_keys, free_memory, worker_init_fn
from train import to_device

# Fallback config when --config-from is not given. NOTE: this may NOT match your
# trained run — prefer --config-from for an apples-to-apples baseline.
DEFAULT_MODEL_CONFIG = {
    'use_positional_encoding': True, 'use_residue_features': True,
    'use_3di_features': True, 'edge_attn_softmax': True,
    'dist_bias_gamma': 0.1, 'detach_h3': True,
    'dist_encoding': 'rbf', 'rbf_dim': 16,
}
OUTPUT_FILE = './data/virome_embeddings_random_init.pt'
CHECKPOINT_DIR = './checkpoints'


def _resolve_files(file_list, downsampled):
    if file_list:
        files = []
        for fl in file_list:
            with open(fl) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        files.append(ln if os.path.isabs(ln) or os.path.exists(ln)
                                     else os.path.join(PROC_DIR, os.path.basename(ln)))
        print(f"File-list mode: {len(files)} proteins from {len(file_list)} manifest(s)")
        return files
    if downsampled:
        return _downsampled_files()
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a randomly initialised model (no checkpoint weights).")
    parser.add_argument('--model', choices=['asymmetric', 'topotein'], default='asymmetric')
    parser.add_argument('--config-from', dest='config_from', type=str, default=None,
                        help="Trained checkpoint .pth to copy `model_config` from, so the random "
                             "model has the IDENTICAL architecture (recommended).")
    parser.add_argument('--file-list', nargs='*', default=None,
                        help="Manifest file(s) to restrict extraction to (e.g. the corrected "
                             "sub-base data/subbase_corrected_{train,val}.txt).")
    parser.add_argument('--downsampled', action='store_true',
                        help="Use the legacy deterministic downsampled split (superseded by --file-list).")
    parser.add_argument('--max-residues', dest='max_residues', type=int, default=8000,
                        help="Cap residues per (forward-only) batch; lower if MPS OOMs.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Resolve the architecture config: copy from a trained checkpoint if given.
    if args.config_from:
        ckpt = torch.load(args.config_from, map_location='cpu', weights_only=False)
        model_config = dict(ckpt.get('model_config') or {})
        print(f"Model config copied from {args.config_from}: {model_config}")
        if not model_config:
            print("[!] checkpoint had no model_config; falling back to DEFAULT_MODEL_CONFIG")
            model_config = dict(DEFAULT_MODEL_CONFIG)
    else:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        print(f"[!] No --config-from; using DEFAULT_MODEL_CONFIG (may not match your trained run): {model_config}")

    files = _resolve_files(args.file_list, args.downsampled)
    dataset = ExtractionDataset(PROC_DIR, files=files)
    out_file = args.out or OUTPUT_FILE
    if len(dataset) == 0:
        print(f"No .pt files found in {PROC_DIR}.")
        return

    keys_cache = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
    sizes = [k[0] for k in extract_batch_keys(dataset.files, keys_cache)]
    sampler = LengthBudgetSampler(sizes, batch_size=64, max_residues=args.max_residues)
    print(f"{len(sampler)} batches over {len(dataset)} proteins (max_residues={args.max_residues})")

    num_workers = max(1, (os.cpu_count() or 4) // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_with_names,
                            num_workers=num_workers, prefetch_factor=1, persistent_workers=False,
                            pin_memory=DEVICE.type == 'cuda', worker_init_fn=worker_init_fn)

    if args.model == 'asymmetric':
        from asymmetric_topotein import AsymmetricTopoNet
        model = AsymmetricTopoNet(scalar_dim=128, **model_config).to(DEVICE)
    else:
        from topotein import Topotein
        model = Topotein(scalar_dim=128, **model_config).to(DEVICE)
    model.eval()

    embeddings_dict = {}
    with torch.no_grad():
        for step, (names, batch) in enumerate(tqdm(dataloader, desc="Extracting (random init)")):
            if batch is None:
                continue
            features = to_device(batch, DEVICE)
            model_in, real_B = pad_to_buckets(features)
            out = model(model_in)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            out = out[:real_B]
            for name, emb in zip(names, out):
                embeddings_dict[name] = emb.cpu().numpy()
            del features, model_in, out, batch
            if DEVICE.type == 'mps' and (step + 1) % 50 == 0:
                import gc
                gc.collect()
                torch.mps.empty_cache()

    free_memory()
    print(f"\nExtracted embeddings for {len(embeddings_dict)} proteins.")
    torch.save(embeddings_dict, out_file)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
