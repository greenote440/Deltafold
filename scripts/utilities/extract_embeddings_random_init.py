"""
One-off: extract embeddings from a FRESH, randomly-initialized model (no checkpoint
loaded) so we have a true epoch-0 baseline for Table 1 of deltafold_protocol.tex.
Mirrors extract_embeddings.py exactly except it skips the checkpoint load.
"""
import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from extract_embeddings import (
    ExtractionDataset, collate_with_names, LengthBudgetSampler, DEVICE, PROC_DIR,
    _downsampled_files,
)
from train_contrastive import pad_to_buckets, extract_batch_keys, free_memory, worker_init_fn
from train import to_device
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)

MODEL_CONFIG = {
    'use_positional_encoding': False, 'use_residue_features': False,
    'use_3di_features': True, 'edge_attn_softmax': True,
    'dist_bias_gamma': 0.1, 'detach_h3': True,
}
OUTPUT_FILE = './data/virome_embeddings_random_init.pt'
CHECKPOINT_DIR = './checkpoints'


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a randomly initialised model (no checkpoint).")
    parser.add_argument('--downsampled', action='store_true', help="Only extract the ~3647 proteins from the downsampled training split (same deterministic set as --downsampled in extract_embeddings.py).")
    parser.add_argument('--out', type=str, default=None, help="Output path for the embeddings .pt file (default: data/virome_embeddings_random_init.pt).")
    args = parser.parse_args()

    files = _downsampled_files() if args.downsampled else None
    dataset = ExtractionDataset(PROC_DIR, files=files)
    out_file = args.out or OUTPUT_FILE
    if args.downsampled:
        print(f"Downsampled mode: {len(dataset)} proteins")

    keys_cache = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
    sizes = [k[0] for k in extract_batch_keys(dataset.files, keys_cache)]
    sampler = LengthBudgetSampler(sizes, batch_size=64, max_residues=24000)
    print(f"{len(sampler)} batches over {len(dataset)} proteins")

    num_workers = max(1, (os.cpu_count() or 4) // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_with_names,
                            num_workers=num_workers, prefetch_factor=1, persistent_workers=False,
                            pin_memory=DEVICE.type == 'cuda', worker_init_fn=worker_init_fn)

    from asymmetric_topotein import AsymmetricTopoNet
    model = AsymmetricTopoNet(scalar_dim=128, **MODEL_CONFIG).to(DEVICE)
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
