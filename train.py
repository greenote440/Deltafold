"""
Deltafold Training Entry Point
Modular training script supporting Topotein/Asymmetric models and MTM/Contrastive tasks.
"""
import os
import sys
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


def _deltafold_requested():
    """True when this run targets the CUDA `deltafold` box (1x L40S, Xeon, 1TB RAM).
    Detected from argv/env at import time so DEVICE is resolved to CUDA *before*
    anything binds it — the `--deltafold` flag itself is parsed later in main()."""
    return ('--deltafold' in sys.argv) or (os.environ.get('DELTAFOLD_DEVICE') == 'cuda')


def _resolve_device():
    """Pick the compute device. `--deltafold` (or DELTAFOLD_DEVICE=cuda) forces the
    CUDA path instead of the mps/cpu autodetect and pins a *single* GPU (the box has
    two L40S but only one is usable), honouring a pre-set CUDA_VISIBLE_DEVICES."""
    if _deltafold_requested():
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')   # one GPU only
        if not torch.cuda.is_available():
            raise SystemExit("[--deltafold] no CUDA GPU visible to torch "
                             "(run this on the deltafold box, not macOS/MPS).")
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available()
                        else ('cuda' if torch.cuda.is_available() else 'cpu'))


DEVICE = _resolve_device()
PROC_DIR = './data/hoan_processed'
# Overridable per-run (e.g. by the overnight ablation sweep) so each config's
# checkpoints / training_log / epoch_eval land in their own directory and don't
# collide. Resolved at import time from the env, before train_contrastive binds it.
CHECKPOINT_DIR = os.environ.get('DELTAFOLD_CKPT_DIR', './checkpoints')
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


def get_corrected_split(prefix='./data/subbase_corrected'):
    """Loads the corrected prototyping sub-base (plan_experimentation_v2 §4) from
    the manifests written by ``scripts/utilities/build_corrected_subbase.py``.

    This is the bias-corrected replacement for the old downsampler
    (``subdataset_files_refined.txt``): the manifests already encode a
    representative size distribution (singletons kept), a cold cluster-aware
    train/val split, and exact-sequence dedup, so no further sampling happens
    here. Keyword positive-controls live in ``<prefix>_controls.txt`` and are
    deliberately NOT loaded into train/val."""
    train_path, val_path = f"{prefix}_train.txt", f"{prefix}_val.txt"
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(
            f"Corrected sub-base manifests not found ({train_path} / {val_path}). "
            f"Run: python scripts/utilities/build_corrected_subbase.py")

    def _read(p):
        with open(p) as f:
            return [ln.strip() for ln in f if ln.strip()]

    train_files, val_files = _read(train_path), _read(val_path)
    print(f"Corrected sub-base (§4): {len(train_files)} train / {len(val_files)} val "
          f"proteins (cold cluster split, deduped, size-distribution preserved).")
    return train_files, val_files


def get_split(data_dir, cluster_tsv_path, split_ratio=0.8, seed=42, split='cluster'):
    """Dispatches to the requested split strategy."""
    if split == 'phylo':
        return get_phylogenetic_split(data_dir, split_ratio=split_ratio, seed=seed, level='taxid')
    if split == 'corrected':
        return get_corrected_split()
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

def _cli_has(*flags):
    """Whether the user passed one of these flag spellings on the command line
    (so the --deltafold preset never clobbers an explicit choice)."""
    return any(a == f or a.startswith(f + '=') for a in sys.argv for f in flags)


def _configure_cuda_perf():
    """Turn on the Ada/L40S throughput levers: TF32 for fp32 matmuls (big speedup,
    negligible precision loss vs the equivariance tolerances), and cudnn autotuning
    (safe because bucket-padding keeps the set of tensor shapes small)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


# --deltafold hardware preset for the L40S box (1x NVIDIA L40S 46GB, Xeon 96T,
# ~981GiB RAM): trade the MPS/16GB-Mac safety limits for CUDA throughput. Each
# entry maps an args attribute to (preset value, CLI spellings that mean "user set
# it"); a knob is only overridden when the user did NOT pass it explicitly.
_DELTAFOLD_PRESET = {
    # Big VRAM -> pack far more residues per step (MPS default was 1200/1800).
    'max_residues':   (8000,  ['--max-residues']),
    'min_residues':   (6000,  ['--min-residues']),
    # More proteins per batch => more in-batch InfoNCE negatives (count cap; the
    # residue budget usually binds first). Raise --batch_size further if VRAM allows.
    'batch_size':     (128,   ['--batch_size']),
    # The RSS governor + per-step empty_cache exist to protect a 16GB unified-memory
    # Mac; on a 1TB CUDA box they only cost throughput (and the 14GB hard cap would
    # falsely trip cold-restarts). Disable them.
    'cleanup_every':  (0,     ['--cleanup-every']),
    'mem_soft_gb':    (0.0,   ['--mem-soft-gb']),
    'mem_hard_gb':    (0.0,   ['--mem-hard-gb']),
    'no_budget_adapt': (True, ['--no-budget-adapt']),
    # 96 hardware threads + 1TB RAM: parallel .pt loading with pinned host buffers.
    'num_workers':    (16,    ['--num-workers']),
}


def _apply_deltafold_preset(args):
    applied = {}
    for attr, (val, flags) in _DELTAFOLD_PRESET.items():
        if not _cli_has(*flags):
            setattr(args, attr, val)
            applied[attr] = val
    return applied


def main():
    parser = argparse.ArgumentParser(description="Deltafold Modular Training")
    parser.add_argument('--model', type=str, choices=['topotein', 'asymmetric', 'equivariant'], default='equivariant',
                        help="topotein=vendored TCPNet; asymmetric=invariant AsymmetricTopoNet; "
                             "equivariant=EquivariantTopoNet (SE(3)/O(3), plan §F; use --scalarize/--vector-dim).")
    # --- topotein/tcpnet architecture cost knobs (None = use the config default) ---
    # These change the model shape, so a checkpoint trained with them must be
    # extracted with the same values; they're saved in the checkpoint's model_config.
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=None, help="[topotein] GCP message-passing depth (config default 6). Compute is ~linear in this; 4 or 3 give the biggest cheap speedup.")
    parser.add_argument('--emb-dim', dest='emb_dim', type=int, default=None, help="[topotein] Hidden width (config default 128). All sub-dims derive from it; keep a multiple of 128 (e.g. 128/256) or instantiation errors on the vector-dim divisibility constraint.")
    parser.add_argument('--knn', dest='knn', type=int, default=None, help="[topotein] k nearest-neighbour edges per node (config default 16). Edge-attention cost scales with edges, so knn 8 ~halves the per-layer edge work.")
    parser.add_argument('--num-message-layers', dest='num_message_layers', type=int, default=None, help="[topotein] GCP sub-layers inside each message step (config default 4).")
    parser.add_argument('--num-feedforward-layers', dest='num_feedforward_layers', type=int, default=None, help="[topotein] feed-forward sub-layers per block (config default 2).")
    parser.add_argument('--no-typecheck', dest='disable_typecheck', action='store_true', help="[topotein] Disable the workshop's per-call jaxtyping/beartype shape checks (sets JAXTYPING_DISABLE=1). A few %% faster; only do this once the model is known-good.")
    parser.add_argument('--task', type=str, choices=['mtm', 'contrastive'], default='contrastive')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--dataset-size', dest='dataset_size', type=int, default=None, help="Limit the size of the training/validation set for quick checks.")
    parser.add_argument('--downsampled', action='store_true', help="Train on the downsampled sub-dataset: the same ~3647 proteins (train+val) as the original downsampled run, reconstructed from the deterministic cluster-aware split (seed=42). Matches extract_embeddings.py --downsampled. Composable with --dataset-size for an even smaller quick check.")
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
    parser.add_argument('--max-residues', dest='max_residues', type=int, default=3000, help="Cap ORIGINAL residues per batch (forward sees ~1.75x after 2 views). Batches are packed by residue budget, not protein count, so a few large proteins can't blow up memory. Default is model-aware: ~1800 for the equivariant topotein/tcpnet model (heavy edge/SSE attention on MPS), ~3500 for asymmetric. Lower = lighter/faster steps.")
    parser.add_argument('--no-pad-buckets', dest='pad_buckets', action='store_false', help="Disable bucket-padding of batch shapes. Padding bounds the number of distinct MPS kernels (avoids per-epoch slowdown from kernel-cache bloat); only disable for debugging.")
    parser.add_argument('--mem-soft-gb', dest='mem_soft_gb', type=float, default=11.0, help="Soft RSS cap (GB): above this, force an off-schedule gc+empty_cache every step. 0 disables the governor's pressure logic.")
    parser.add_argument('--mem-hard-gb', dest='mem_hard_gb', type=float, default=14.0, help="Hard RSS cap (GB): above this (after a reclaim attempt), cold-restart DataLoader workers and shrink the residue budget. Keeps headroom below the 16GB swap cliff. 0 disables.")
    parser.add_argument('--min-residues', dest='min_residues', type=int, default=3000, help="Floor for the dynamic residue-budget shrink under memory pressure. Default is model-aware (~1200 for topotein/tcpnet, ~3500 for asymmetric).")
    parser.add_argument('--no-budget-adapt', dest='no_budget_adapt', action='store_true', help="Disable dynamic residue-budget shrink under memory pressure (keeps cold restarts).")
    parser.add_argument('--split', type=str, choices=['cluster', 'phylo', 'corrected'], default='phylo', help="Train/val split: cluster-aware, phylogenetic by taxid (report 7), or 'corrected' = the bias-corrected prototyping sub-base from scripts/utilities/build_corrected_subbase.py (plan v2 §4; reads data/subbase_corrected_{train,val}.txt). Use without --downsampled.")
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
    parser.add_argument('--dist-encoding', dest='dist_encoding', type=str,
                        choices=['sinusoidal', 'rbf'], default='rbf',
                        help="Rank-1 distance encoding (§1, Mod 1): 'sinusoidal' (original) or 'rbf' "
                             "(Gaussian radial bases, calibrated to the contact scale). asymmetric only.")
    parser.add_argument('--rbf-dim', dest='rbf_dim', type=int, default=16,
                        help="Number of Gaussian RBF channels when --dist-encoding rbf (default 16; try 32).")
    parser.add_argument('--temperature', dest='temperature', type=float, default=0.1,
                        help="InfoNCE temperature tau (§5 axis B sweep; MoCo plan default 0.2).")
    parser.add_argument('--objective', type=str, choices=['infonce', 'moco'], default='infonce',
                        help="Contrastive objective (plan_impl_3): 'infonce' (jitter views) or "
                             "'moco' (substructure positives + momentum queue + projection head).")
    parser.add_argument('--moco-k', dest='moco_k', type=int, default=8192, help="MoCo negative-queue length.")
    parser.add_argument('--moco-m', dest='moco_m', type=float, default=0.99, help="MoCo key-encoder EMA momentum.")
    parser.add_argument('--sub-f-lo', dest='sub_f_lo', type=float, default=0.4, help="Substructure size fraction, low.")
    parser.add_argument('--sub-f-hi', dest='sub_f_hi', type=float, default=0.7, help="Substructure size fraction, high.")
    parser.add_argument('--sub-mode', dest='sub_mode', type=str, choices=['contiguous', 'ball'],
                        default='contiguous', help="Substructure sampling mode.")
    parser.add_argument('--scalarize', dest='scalarize', type=str, choices=['frame', 'norm'], default='frame',
                        help="[equivariant] 'frame'=GCP SE(3)+chiral (default), 'norm'=GVP O(3) (plan §F ablation).")
    parser.add_argument('--vector-dim', dest='vector_dim', type=int, default=16,
                        help="[equivariant/topotein] vector channels per cell (default 16; use 8 for memory, esp. with MoCo).")
    parser.add_argument('--deltafold', action='store_true',
                        help="Run on the CUDA `deltafold` box (1x NVIDIA L40S 46GB, Xeon 96T, ~1TB RAM): "
                             "force the CUDA device (pins one GPU via CUDA_VISIBLE_DEVICES=0), enable "
                             "TF32/cudnn autotune, and apply a high-throughput preset (larger residue "
                             "budget + batch, more DataLoader workers, RSS memory-governor disabled). "
                             "Any of those knobs passed explicitly still wins.")
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                        help="DataLoader worker processes for the train loader (default: cpu_count//2; "
                             "the --deltafold preset uses 16). Val always uses 0.")
    parser.add_argument('--tensor-diagram', dest='tensor_diagram', type=str, default='default',
                        help="[topotein] Message-passing order/channels (protocol §Modular ordering): "
                             "'default' (Topotein 4-step), 'residue_hub' (no inter-SSE outer-edge channel), "
                             "'no_rank3' (drop the protein cell), 'reordered', or a custom comma-separated "
                             "step order e.g. 'edge,node,sse,protein'.")
    parser.add_argument('--readout', dest='readout', type=str, choices=['node', 'protein'], default='node',
                        help="[topotein] Graph readout (protocol §Readout): 'node' pooling (default) or "
                             "the rank-3 'protein' cell.")
    args = parser.parse_args()

    if args.deltafold:
        _configure_cuda_perf()
        applied = _apply_deltafold_preset(args)
        props = torch.cuda.get_device_properties(0)
        print(f"[--deltafold] CUDA on {props.name} ({props.total_memory / 1e9:.0f} GB, "
              f"visible GPU(s)={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}); TF32+cudnn.benchmark on.")
        print(f"[--deltafold] preset overrides (explicit flags win): {applied}")

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
            dist_encoding=args.dist_encoding,
            rbf_dim=args.rbf_dim,
            temperature=args.temperature,
            objective=args.objective,
            moco_k=args.moco_k,
            moco_m=args.moco_m,
            sub_f_lo=args.sub_f_lo,
            sub_f_hi=args.sub_f_hi,
            sub_mode=args.sub_mode,
            scalarize=args.scalarize,
            vector_dim=args.vector_dim,
            tensor_diagram=args.tensor_diagram,
            readout=args.readout,
            num_workers=args.num_workers,
            mem_soft_gb=args.mem_soft_gb,
            mem_hard_gb=args.mem_hard_gb,
            min_residues=args.min_residues,
            no_budget_adapt=args.no_budget_adapt,
            downsampled=args.downsampled,
            num_layers=args.num_layers,
            emb_dim=args.emb_dim,
            knn=args.knn,
            num_message_layers=args.num_message_layers,
            num_feedforward_layers=args.num_feedforward_layers,
            disable_typecheck=args.disable_typecheck,
        )

    if args.cprofile:
        profiler.disable()
        print("\n" + "="*30 + " CPROFILE RESULTS (Top 30 by Cumulative Time) " + "="*30)
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(30)

if __name__ == "__main__":
    main()