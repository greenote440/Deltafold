"""
Deltafold Training Entry Point
Modular training script supporting Topotein/Asymmetric models and MTM/Contrastive tasks.
"""
import os
import glob
import re
import random
import argparse
from collections import defaultdict
import torch
import cProfile
import pstats
import torch.profiler
from torch.utils.data import Dataset, DataLoader
from contrastive_engine import StructuralAugmentations

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
PROC_DIR = './data/hoan_processed'
CHECKPOINT_DIR = './checkpoints'
CLUSTER_TSV = './data/cluster.tsv'

class PCCDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        try:
            try:
                data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(self.files[idx], map_location='cpu')
        except Exception as e:
            # A corrupt/partially-written .pt (e.g. lifter still writing, or an interrupted
            # save) must not kill the whole run. custom_collate / contrastive_collate already
            # drop None entries, so skip this sample.
            print(f"[PCCDataset] skipping unreadable {os.path.basename(self.files[idx])}: {e}")
            return None, self.files[idx]

        if self.transform:
            data = self.transform(data)
        return data, self.files[idx]

def custom_collate(batch):
    # Handle (data, path) tuples from PCCDataset
    if isinstance(batch[0], tuple):
        paths = [b[1] for b in batch]
        batch = [b[0] for b in batch]
    else:
        paths = None

    batched = {
        'rank0': {}, 'rank1': {}, 'rank2': [], 'rank3': {},
        'batch_idx_0': [], 'batch_idx_2': [], 'sse_map_0': [],
        'rank2_features': [], 'nodes_per_sse': []
    }
    if paths is not None:
        batched['paths'] = paths
    
    node_offset = 0
    sse_offset = 0
    for i, data in enumerate(batch):
        if data is None:
            continue
        # Protect against shape mismatches in batching
        if data['rank1']['source'].shape[1] != 16:
            continue
            
        r0 = data['rank0']
        n_nodes = r0['aa'].shape[0]
        if n_nodes == 0:
            continue
            
        for k, v in r0.items():
            if k not in batched['rank0']:
                batched['rank0'][k] = []
            batched['rank0'][k].append(v)
            
        batched['batch_idx_0'].append(torch.full((n_nodes,), i, dtype=torch.long))
            
        r1 = data['rank1']
        for k, v in r1.items():
            if k not in batched['rank1']:
                batched['rank1'][k] = []
            if k in ['source', 'target']:
                batched['rank1'][k].append(v + node_offset)
            else:
                batched['rank1'][k].append(v)
        
        sse_map = torch.zeros(n_nodes, dtype=torch.long)
        r2 = data['rank2']
        for j, sse in enumerate(r2):
            feat = torch.cat([sse['type'], sse['shape_descriptors'], sse['eigenvalues']], dim=-1)
            batched['rank2_features'].append(feat)
            batched['nodes_per_sse'].append(float(sse['end_idx'] - sse['start_idx'] + 1))
            
            # Map nodes in this SSE range to the global SSE index
            sse_map[sse['start_idx']:sse['end_idx']+1] = sse_offset + j
            
        batched['batch_idx_2'].append(torch.full((len(r2),), i, dtype=torch.long))
        batched['sse_map_0'].append(sse_map)
            
        r3 = data['rank3']
        for k, v in r3.items():
            if k not in batched['rank3']:
                batched['rank3'][k] = []
            batched['rank3'][k].append(v)
        
        node_offset += n_nodes
        sse_offset += len(r2)
        
    if node_offset == 0:
        return None
        
    # Concat lists of tensors
    for k in batched['rank0']:
        if isinstance(batched['rank0'][k][0], torch.Tensor):
            batched['rank0'][k] = torch.cat(batched['rank0'][k], dim=0)
            
    for k in batched['rank1']:
        if isinstance(batched['rank1'][k][0], torch.Tensor):
            batched['rank1'][k] = torch.cat(batched['rank1'][k], dim=0)
            
    for k in batched['rank3']:
        first_elem = batched['rank3'][k][0]
        if isinstance(first_elem, torch.Tensor):
            batched['rank3'][k] = torch.stack(batched['rank3'][k])
        elif isinstance(first_elem, (int, float)):
            batched['rank3'][k] = torch.tensor(batched['rank3'][k], dtype=torch.float32)
            
    batched['batch_idx_0'] = torch.cat(batched['batch_idx_0'])
    batched['batch_idx_2'] = torch.cat(batched['batch_idx_2'])
    batched['sse_map_0'] = torch.cat(batched['sse_map_0'])
    
    if batched['rank2_features']:
        batched['rank2_features'] = torch.stack(batched['rank2_features'])
        batched['nodes_per_sse'] = torch.tensor(batched['nodes_per_sse']).float().unsqueeze(1)
    else:
        batched['rank2_features'] = torch.empty(0, 12)
        batched['nodes_per_sse'] = torch.empty(0, 1)
    
    return batched

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj

def extract_accession(text):
    """Extracts the accession number (e.g., YP_010085741) from a string."""
    match = re.search(r'([A-Z]{1,2}_[0-9]{5,10})', text)
    return match.group(1) if match else text

def get_cluster_aware_split(data_dir, cluster_tsv_path, split_ratio=0.8, seed=42):
    """Splits data such that members of the same cluster stay in the same split."""
    acc_to_cluster = {}
    if os.path.exists(cluster_tsv_path):
        with open(cluster_tsv_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep = extract_accession(parts[0])
                    member = extract_accession(parts[1])
                    acc_to_cluster[member] = rep
    
    pt_files = glob.glob(os.path.join(data_dir, '*.pt'))
    cluster_groups = defaultdict(list)
    for f in pt_files:
        acc = extract_accession(os.path.basename(f))
        cluster_id = acc_to_cluster.get(acc, acc) # Singletons use their own accession
        cluster_groups[cluster_id].append(f)
    
    cluster_ids = sorted(list(cluster_groups.keys()))
    random.seed(seed)
    random.shuffle(cluster_ids)
    
    split_idx = int(len(cluster_ids) * split_ratio)
    train_ids = cluster_ids[:split_idx]
    val_ids = cluster_ids[split_idx:]
    
    train_files = [f for cid in train_ids for f in cluster_groups[cid]]
    val_files = [f for cid in val_ids for f in cluster_groups[cid]]
    
    return train_files, val_files

def parse_taxon(filename, level='taxid'):
    """Extracts a taxonomic id from a processed filename of the form
    '{gene}__{accession}__{species}__{taxid}.pt'. Falls back to the basename."""
    base = os.path.basename(filename)
    base = base[:-3] if base.endswith('.pt') else base
    parts = base.split('__')
    if len(parts) >= 4:
        return parts[-1] if level == 'taxid' else '__'.join(parts[2:-1])
    return base


def get_phylogenetic_split(data_dir, split_ratio=0.8, seed=42, level='taxid'):
    """Splits data by viral taxonomy (taxid or species) so that no taxon spans
    train and validation (report 7, "phylogenetic train/validation/test splits").
    This is the correct test of generalizable structure vs virome-specific stats."""
    pt_files = glob.glob(os.path.join(data_dir, '*.pt'))
    taxon_groups = defaultdict(list)
    for f in pt_files:
        taxon_groups[parse_taxon(f, level=level)].append(f)

    taxa = sorted(taxon_groups.keys())
    random.seed(seed)
    random.shuffle(taxa)

    split_idx = int(len(taxa) * split_ratio)
    train_taxa, val_taxa = taxa[:split_idx], taxa[split_idx:]
    train_files = [f for t in train_taxa for f in taxon_groups[t]]
    val_files = [f for t in val_taxa for f in taxon_groups[t]]
    print(f"Phylogenetic split by {level}: {len(train_taxa)} train / {len(val_taxa)} val taxa "
          f"({len(train_files)}/{len(val_files)} proteins).")
    return train_files, val_files


def get_split(data_dir, cluster_tsv_path, split_ratio=0.8, seed=42, split='cluster'):
    """Dispatches to the requested split strategy."""
    if split == 'phylo':
        return get_phylogenetic_split(data_dir, split_ratio=split_ratio, seed=seed, level='taxid')
    return get_cluster_aware_split(data_dir, cluster_tsv_path, split_ratio=split_ratio, seed=seed)


def run_profiler(model_type, task, batch_size, dataset_size=None):
    """Initializes model and dataloader to run a short profiling session."""
    print(f"\n--- Running Profiler for model='{model_type}', task='{task}' ---")
    
    if model_type == 'asymmetric':
        from asymmetric_topotein import AsymmetricTopoNet
        model = AsymmetricTopoNet(scalar_dim=128).to(DEVICE)
    else:
        from topotein import Topotein
        model = Topotein(scalar_dim=128).to(DEVICE)
    
    model.train()
    
    if task == 'contrastive':
        from train_contrastive import contrastive_collate
        from contrastive_engine import NTXentLoss
        aug = StructuralAugmentations()
        criterion = NTXentLoss().to(DEVICE)
        train_files, _ = get_cluster_aware_split(PROC_DIR, CLUSTER_TSV)
        if dataset_size:
            print(f"Profiling with a limited dataset size of {dataset_size}.")
            train_files = train_files[:dataset_size]
        # Intentionally use 0 workers to expose the data loading bottleneck
        dataset = PCCDataset(train_files, transform=aug)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate, num_workers=0)
        
        print("Profiling one batch preparation and forward pass...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], 
            record_shapes=True, 
            with_stack=True,
            profile_memory=True
        ) as prof:
            with torch.profiler.record_function("data_preparation"):
                batch = next(iter(loader))

            if batch:
                with torch.profiler.record_function("model_forward"):
                    features = to_device(batch, DEVICE)
                    z = model(features)
                    B = z.size(0) // 2
                    loss = criterion(z[:B], z[B:])

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    trace_file = f"trace_{model_type}_{task}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to {trace_file}. Open in chrome://tracing to view.")

def main():
    parser = argparse.ArgumentParser(description="Deltafold Modular Training")
    parser.add_argument('--model', type=str, choices=['topotein', 'asymmetric'], default='topotein')
    parser.add_argument('--task', type=str, choices=['mtm', 'contrastive'], default='contrastive')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--dataset-size', dest='dataset_size', type=int, default=None, help="Limit the size of the training/validation set for quick checks.")
    parser.add_argument('--profile', action='store_true', help="Run a short profiling session and exit.")
    parser.add_argument('--profile_train', action='store_true', help="Profile a few steps of the training loop and save the trace.")
    parser.add_argument('--cprofile', action='store_true', help="Run the entire execution through cProfile to find Python-level bottlenecks.")
    # --- Shortcut mitigations (report 7) ---
    parser.add_argument('--no-positional', dest='no_positional', action='store_true', help="Drop sequence-index positional encoding (fixes sequential-attention shortcut, report 3.2).")
    parser.add_argument('--no-residue', dest='no_residue', action='store_true', help="Drop residue-type (aa) node features (composition shortcut, report 2.2).")
    parser.add_argument('--no-3di', dest='no_3di', action='store_true', help="Drop the Foldseek 3Di node features (benchmark shortcut, report 5).")
    parser.add_argument('--hard-neg-beta', dest='hard_neg_beta', type=float, default=0.0, help="Hard-negative reweighting strength for InfoNCE (0=off; report 7).")
    parser.add_argument('--hard-neg-mining', dest='hard_neg_mining', action='store_true', help="Build batches of length/SSE-matched proteins so negatives are superficially similar but topologically distinct (report 7 #1).")
    parser.add_argument('--cleanup-every', dest='cleanup_every', type=int, default=10, help="Run MPS gc/empty_cache every N steps (default 50; 0 disables per-step cleanup). Lower = less peak RAM but much slower on MPS.")
    parser.add_argument('--max-residues', dest='max_residues', type=int, default=4000, help="With --hard-neg-mining, cap ORIGINAL residues per batch (forward sees ~1.75x after 2 views). Default 4000 keeps batches off the MPS memory-swap cliff. Lower = lighter/faster steps.")
    parser.add_argument('--no-pad-buckets', dest='pad_buckets', action='store_false', help="Disable bucket-padding of batch shapes. Padding bounds the number of distinct MPS kernels (avoids per-epoch slowdown from kernel-cache bloat); only disable for debugging.")
    parser.add_argument('--mem-soft-gb', dest='mem_soft_gb', type=float, default=11.0, help="Soft RSS cap (GB): above this, force an off-schedule gc+empty_cache every step. 0 disables the governor's pressure logic.")
    parser.add_argument('--mem-hard-gb', dest='mem_hard_gb', type=float, default=14.0, help="Hard RSS cap (GB): above this (after a reclaim attempt), cold-restart DataLoader workers and shrink the residue budget. Keeps headroom below the 16GB swap cliff. 0 disables.")
    parser.add_argument('--min-residues', dest='min_residues', type=int, default=2000, help="Floor for the dynamic residue-budget shrink under memory pressure.")
    parser.add_argument('--split', type=str, choices=['cluster', 'phylo'], default='cluster', help="Train/val split: cluster-aware or phylogenetic by taxid (report 7).")
    parser.add_argument('--tm-aux-weight', dest='tm_aux_weight', type=float, default=0.0, help="Weight of the TM-score regression auxiliary loss (0=off; report 7/3.4).")
    parser.add_argument('--unsupervised', dest='unsupervised', action='store_true', help="Use plain InfoNCE instead of cluster-label SupCon (avoids taxonomic label leakage, report 4.1).")
    # --- TM-score analysis fixes (tm_score_analysis.md §5) ---
    parser.add_argument('--tm-cache', dest='tm_cache', type=str, default=None,
                        help="Path to the pre-computed pairwise TM-score cache from build_tm_cache.py "
                             "(§5.1). Enables cached --tm-aux-weight (§5.2) and --soft-supcon (§5.6).")
    parser.add_argument('--soft-supcon', dest='soft_supcon', action='store_true',
                        help="Use TM-weighted soft supervised contrastive loss instead of binary SupCon "
                             "(§5.6). Requires --tm-cache. Optimises rho at some cost to ARI.")
    parser.add_argument('--crop-aug', dest='crop_aug', action='store_true',
                        help="Re-enable destructive SSE crop augmentation (off by default per §5.5).")
    parser.add_argument('--jitter-sigma', dest='jitter_sigma', type=float, default=0.3,
                        help="Coordinate-jitter sigma in Angstrom (§5.5; default 0.3 ~ AF2 uncertainty).")
    parser.add_argument('--no-softmax-edge', dest='no_softmax_edge', action='store_true',
                        help="Disable softmax edge attention; revert to the saturating sigmoid gate (§5.3).")
    parser.add_argument('--dist-bias-gamma', dest='dist_bias_gamma', type=float, default=0.1,
                        help="Geometric attention-bias strength -gamma*distance (§5.4; 0 disables). asymmetric only.")
    parser.add_argument('--no-detach-h3', dest='no_detach_h3', action='store_true',
                        help="Disable detaching the per-layer h3 global update; keep the shortcut gradient (§5.7).")
    args = parser.parse_args()
    print(f"Dataset size: {args.dataset_size if args.dataset_size else 'Full'}")
    
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.profile:
        run_profiler(model_type=args.model, task=args.task, batch_size=args.batch_size, dataset_size=args.dataset_size)
    elif args.task == 'mtm':
        from train_mtm import train_mtm
        train_mtm(model_type=args.model, epochs=args.epochs, batch_size=args.batch_size, mask_ratio=0.25, lr=args.lr, accum_steps=args.accum_steps, dataset_size=args.dataset_size, profile_train=args.profile_train)
    else:
        from train_contrastive import train_contrastive
        train_contrastive(
            model_type=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            accum_steps=args.accum_steps, dataset_size=args.dataset_size, profile_train=args.profile_train,
            use_positional_encoding=not args.no_positional,
            use_residue_features=not args.no_residue,
            use_3di_features=not args.no_3di,
            hard_neg_beta=args.hard_neg_beta,
            split=args.split,
            tm_aux_weight=args.tm_aux_weight,
            supervised=not args.unsupervised,
            hard_neg_mining=args.hard_neg_mining,
            cleanup_every=args.cleanup_every,
            max_residues=args.max_residues,
            pad_buckets=args.pad_buckets,
            tm_cache_path=args.tm_cache,
            soft_supcon=args.soft_supcon,
            use_crop=args.crop_aug,
            jitter_sigma=args.jitter_sigma,
            edge_attn_softmax=not args.no_softmax_edge,
            dist_bias_gamma=args.dist_bias_gamma,
            detach_h3=not args.no_detach_h3,
            mem_soft_gb=args.mem_soft_gb,
            mem_hard_gb=args.mem_hard_gb,
            min_residues=args.min_residues,
        )

    if args.cprofile:
        profiler.disable()
        print("\n" + "="*30 + " CPROFILE RESULTS (Top 30 by Cumulative Time) " + "="*30)
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

if __name__ == "__main__":
    main()