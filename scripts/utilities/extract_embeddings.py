"""
Embedding Extraction Script
Extracts global topological embeddings from the trained Topotein model
for downstream structural clustering.

Memory/throughput optimizations (mirrored from the train/val loops in
train_contrastive.py, see memory `training-perf`):
  1. Residue-budgeted batching (LengthBudgetSampler) keeps every step's compute
     small and roughly equal, so the longest proteins never blow a batch past the
     16GB MPS memory-swap cliff (the cause of 100-200s "hung" steps).
  2. pad_to_buckets() rounds node/edge/SSE/protein counts up to bucket multiples so
     the model sees only a handful of distinct tensor shapes; this bounds the MPS
     kernel cache and stops each step from getting slower as the run proceeds.
  3. Throttled gc.collect()+empty_cache() inside the loop stops the MPS allocator
     pool from creeping upward across the full dataset and spilling into swap.
Extraction is forward-only (1 view, no backward), so it runs comfortably at a
HIGHER residue budget than training (which sees ~1.75x residues after 2 aug views
plus backward activations).
"""
import os
import sys
import glob
import gc
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from topotein import Topotein
from train import custom_collate, to_device
from train_contrastive import pad_to_buckets, extract_batch_keys, free_memory, worker_init_fn

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
PROC_DIR = './data/hoan_processed'
CHECKPOINT_DIR = './checkpoints'
OUTPUT_FILE = './data/virome_embeddings.pt'

class ExtractionDataset(Dataset):
    """
    Dataset that returns both the protein ID (filename) and the PCC features.
    """
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name = os.path.basename(path).replace('.pt', '')
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(path, map_location='cpu')
        return name, data

def collate_with_names(batch):
    """
    Separates names from data, batches the complex data using the custom collate,
    and returns both.
    """
    valid_batch = [b for b in batch if b[1] is not None and b[1]['rank1']['source'].shape[1] == 16]
    if not valid_batch:
        return None, None
    names = [b[0] for b in valid_batch]
    data = [b[1] for b in valid_batch]
    batched_data = custom_collate(data)
    return names, batched_data


class LengthBudgetSampler(Sampler):
    """Residue-budgeted batch sampler for extraction (the inference analogue of
    HardNegativeBatchSampler). Order is irrelevant for extraction (outputs are keyed
    by protein name), so we simply sort by length and greedily pack proteins into a
    batch until either the residue budget or the count cap is hit.

    Two wins at once: (a) per-step compute stays small and bounded -> no MPS swap
    cliff; (b) length-sorting groups similarly sized proteins, which both minimizes
    bucket-padding waste and keeps the set of distinct (bucketed) shapes tiny.
    """
    def __init__(self, sizes, batch_size, max_residues=8000):
        self.batch_size = max(1, batch_size)
        self.max_residues = max(1, max_residues)
        order = sorted(range(len(sizes)), key=lambda i: sizes[i])
        batches, cur, cur_res = [], [], 0
        for idx in order:
            r = int(sizes[idx])
            if cur and (cur_res + r > self.max_residues or len(cur) >= self.batch_size):
                batches.append(cur)
                cur, cur_res = [], 0
            cur.append(idx)
            cur_res += r
        if cur:
            batches.append(cur)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def extract_embeddings(model_type='topotein', task='contrastive', batch_size=32,
                       max_residues=8000, cleanup_every=50, pad_buckets=True):
    dataset = ExtractionDataset(PROC_DIR)
    if len(dataset) == 0:
        print(f"No .pt files found in {PROC_DIR}.")
        return

    # Residue-budgeted batches (reuses training's cached per-file metadata).
    keys_cache = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
    sizes = [k[0] for k in extract_batch_keys(dataset.files, keys_cache)]
    sampler = LengthBudgetSampler(sizes, batch_size, max_residues=max_residues)
    print(f"Residue budget {max_residues}/batch (forward-only), count cap {batch_size}: "
          f"{len(sampler)} batches over {len(dataset)} proteins | pad_buckets={pad_buckets}")

    # prefetch_factor=1: variable-size batches make large prefetch buffers a memory
    # liability on MPS unified memory (the same reasoning as the training loader).
    num_workers = max(1, (os.cpu_count() or 4) // 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_with_names,
                            num_workers=num_workers, prefetch_factor=1, persistent_workers=False,
                            pin_memory=DEVICE.type == 'cuda', worker_init_fn=worker_init_fn)

    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{task}_{model_type}_best.pth')

    # Fallback for legacy checkpoint names if the new best_ckpt_path doesn't exist
    best_ckpt = None
    if not os.path.exists(best_ckpt_path):
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_ep*.pth'))
        if not checkpoint_files:
            print(f"No checkpoints found in {CHECKPOINT_DIR}. Train the model first.")
            return

        best_loss = float('inf')
        for f in checkpoint_files:
            ckpt = torch.load(f, map_location='cpu')
            if ckpt.get('loss', float('inf')) < best_loss:
                best_loss = ckpt['loss']
                best_ckpt = f
    else:
        best_ckpt = best_ckpt_path

    print(f"Loading checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)

    # Restore the shortcut-mitigation config the model was trained with, so that
    # disabled input channels (PE / residue / 3Di) stay disabled at inference.
    model_config = checkpoint.get('model_config', {})
    if model_config:
        print(f"Model config from checkpoint: {model_config}")

    if model_type == 'asymmetric':
        from asymmetric_topotein import AsymmetricTopoNet
        model = AsymmetricTopoNet(scalar_dim=128, **model_config).to(DEVICE)
    else:
        model = Topotein(scalar_dim=128, **model_config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    embeddings_dict = {}

    with torch.no_grad():
        for step, (names, batch) in enumerate(tqdm(dataloader, desc="Extracting Embeddings")):
            if batch is None:
                continue
            features = to_device(batch, DEVICE)

            # Bucket-pad so the model sees only a handful of distinct shapes; slice
            # the dummy-protein rows off the readout before saving.
            if pad_buckets:
                model_in, real_B = pad_to_buckets(features)
            else:
                model_in, real_B = features, features['rank3']['protein_size'].shape[0]

            # Pass through the network (defaults to return_nodes=False, returning global states)
            out = model(model_in)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            out = out[:real_B]

            for name, emb in zip(names, out):
                # .cpu() immediately so the device tensor is freed each step rather
                # than retained in the dict (keeps the MPS pool from growing).
                embeddings_dict[name] = emb.cpu().numpy()

            del features, model_in, out, batch

            # Throttled reclaim — same throttling rationale as the train/val loops:
            # empty_cache() every step would force costly reallocation, but never
            # doing it lets the MPS allocator pool creep into swap over a long run.
            if DEVICE.type == 'mps' and cleanup_every > 0 and (step + 1) % cleanup_every == 0:
                gc.collect()
                torch.mps.empty_cache()

    free_memory()
    print(f"\nExtracted embeddings for {len(embeddings_dict)} proteins.")
    torch.save(embeddings_dict, OUTPUT_FILE)
    print(f"Embeddings successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deltafold Embedding Extraction")
    parser.add_argument('--model', type=str, choices=['topotein', 'asymmetric'], default='topotein')
    parser.add_argument('--task', type=str, choices=['mtm', 'contrastive'], default='contrastive')
    parser.add_argument('--batch_size', type=int, default=32, help="Hard upper bound on proteins per batch (the residue budget usually binds first).")
    parser.add_argument('--max-residues', dest='max_residues', type=int, default=8000, help="Cap residues per batch. Forward-only, so this maps ~directly to in-forward residues; safe above training's 4000 (which sees ~1.75x after 2 views + backward). Keeps batches off the MPS swap cliff.")
    parser.add_argument('--cleanup-every', dest='cleanup_every', type=int, default=50, help="Run MPS gc/empty_cache every N steps (0 disables). Throttled because doing it every step forces costly pool reallocation on MPS.")
    parser.add_argument('--no-pad-buckets', dest='pad_buckets', action='store_false', help="Disable bucket-padding of batch shapes (only for debugging; padding avoids per-step slowdown from MPS kernel-cache bloat).")

    args = parser.parse_args()

    extract_embeddings(model_type=args.model, task=args.task, batch_size=args.batch_size,
                       max_residues=args.max_residues, cleanup_every=args.cleanup_every,
                       pad_buckets=args.pad_buckets)
