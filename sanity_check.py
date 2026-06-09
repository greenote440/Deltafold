import os
import glob
import random
import torch
import numpy as np
import pprint
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

PROC_DIR = './data/hoan_processed'
EMBEDDINGS_FILE = './data/virome_embeddings.pt'

def run_sanity_check():
    pt_files = glob.glob(os.path.join(PROC_DIR, '*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in '{PROC_DIR}'! Have you run the lifter script yet?")
        return
        
    print(f"Found {len(pt_files)} files. Starting structural sanity check...")
    
    valid_count = 0
    corrupted = []
    issues = []
    
    avg_sizes = []
    
    global_sse_counts = {'H': 0, 'E': 0, 'C': 0, 'Unknown': 0}
    global_3di_valid = 0
    global_3di_unknown = 0
    
    for file in tqdm(pt_files, desc="Checking Tensors"):
        try:
            # Safely load the dictionary of tensors (handles both older and newer PyTorch versions)
            try:
                data = torch.load(file, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(file, map_location='cpu')
            
            # 1. Structural Validation
            for rank in ['rank0', 'rank1', 'rank2', 'rank3']:
                if rank not in data:
                    raise ValueError(f"Missing '{rank}' dictionary branch")
            
            N = data['rank3'].get('protein_size', 0)
            if N < 16:
                raise ValueError(f"Protein size {N} is suspiciously small (<16 residues).")
            
            avg_sizes.append(N)
            
            ca_coords = data['rank0'].get('ca_coords')
            if ca_coords is None or ca_coords.shape[0] != N:
                raise ValueError(f"Rank 0 spatial size ({ca_coords.shape[0] if ca_coords is not None else 'None'}) != Rank 3 logic size ({N})")
            
            # Check 3Di Alphabet (Rank 0)
            r0_3di = data['rank0'].get('3di')
            if r0_3di is not None:
                # FOLDSEEK_ALPHABET has 20 chars + 'X'. 'X' is index 20.
                unknowns = int(r0_3di[:, 20].sum().item())
                valid = N - unknowns
                global_3di_valid += valid
                global_3di_unknown += unknowns

            # Check Interaction Graph (Edges)
            rank1 = data['rank1']
            if isinstance(rank1, dict) and 'source' in rank1:
                # New vectorized format (dict of tensors)
                edges = rank1['source']
                if edges.shape[1] not in [15, 16]:
                    raise ValueError(f"Interaction graph k-NN size violation. Neighbors found per node: {edges.shape[1]}")
            elif isinstance(rank1, list):
                # Old format (list of dicts)
                if len(rank1) == 0:
                    raise ValueError("Rank 1 interaction graph is empty.")
                if 'source' not in rank1[0]:
                    raise ValueError("Rank 1 list elements missing 'source' key.")
            else:
                raise ValueError("Rank 1 missing 'source' key or unrecognized format.")
            
            # Check Secondary Structure Elements (Rank 2)
            r2 = data['rank2']
            if not isinstance(r2, list):
                raise ValueError("Rank 2 is not a list format.")
            if len(r2) == 0:
                raise ValueError("Rank 2 SSE list is empty (no secondary structure).")
                
            expected_start = 0
            for sse in r2:
                if sse['start_idx'] != expected_start:
                    raise ValueError(f"SSE segment mismatch: expected start {expected_start}, got {sse['start_idx']}")
                expected_start = sse['end_idx'] + 1
                
                t = sse['type']
                if t[0] == 1: global_sse_counts['H'] += 1
                elif t[1] == 1: global_sse_counts['E'] += 1
                elif t[2] == 1: global_sse_counts['C'] += 1
                else: global_sse_counts['Unknown'] += 1
                
            if expected_start != N:
                raise ValueError(f"SSEs do not cover the whole protein. End at {expected_start-1}, size {N}")

            # 2. Mathematical Integrity Validation (No NaNs or Infs from MPS Covariance)
            shape_desc = data['rank3'].get('global_shape_descriptors')
            if shape_desc is None:
                raise ValueError("Missing 'global_shape_descriptors' in rank3.")
            elif isinstance(shape_desc, torch.Tensor) and (torch.isnan(shape_desc).any() or torch.isinf(shape_desc).any()):
                raise ValueError("NaN or Inf eigenvalues found in Global Shape Descriptors.")
            elif not isinstance(shape_desc, torch.Tensor):
                import numpy as np
                if np.isnan(shape_desc).any() or np.isinf(shape_desc).any():
                    raise ValueError("NaN or Inf eigenvalues found in Global Shape Descriptors.")
            
            valid_count += 1
            
        except Exception as e:
            corrupted.append(file)
            issues.append(str(e))
    
    print("\n" + "="*40 + "\n--- Sanity Check Summary ---\n" + "="*40)
    print(f"Total Files Checked: {len(pt_files)}")
    print(f"Structurally Valid:  {valid_count}")
    print(f"Corrupted/Failed:    {len(corrupted)}")
    
    if avg_sizes:
        print(f"Average Protein Size: {sum(avg_sizes)/len(avg_sizes):.2f} residues")
        
    print("\n--- Detailed Structural Distribution ---")
    print(f"3Di Structural Tokens:    {global_3di_valid} Valid / {global_3di_unknown} Unknown (Masked/Failed)")
    total_sse = sum(global_sse_counts.values())
    if total_sse > 0:
        print(f"Total SSE Segments:       {total_sse}")
        print(f" -> Helices (H):          {global_sse_counts['H']} ({(global_sse_counts['H']/total_sse)*100:.1f}%)")
        print(f" -> Sheets (E):           {global_sse_counts['E']} ({(global_sse_counts['E']/total_sse)*100:.1f}%)")
        print(f" -> Coils (C):            {global_sse_counts['C']} ({(global_sse_counts['C']/total_sse)*100:.1f}%)")
        
    if corrupted:
        print("\n[!] Sample Issues:")
        for i in range(min(10, len(corrupted))):
            print(f" -> {os.path.basename(corrupted[i])}: {issues[i]}")

    if pt_files:
        sample_file = random.choice(pt_files)
        print(f"\nDumping random sample ({os.path.basename(sample_file)}) to sample_pcc.txt...")
        try:
            try:
                data = torch.load(sample_file, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(sample_file, map_location='cpu')
            
            # Prevent PyTorch from truncating large tensors in the string output
            torch.set_printoptions(profile="full")
            with open('sample_pcc.txt', 'w') as f:
                f.write(pprint.pformat(data))
            print("Successfully saved entire PCC to sample_pcc.txt")
        except Exception as e:
            print(f"Failed to dump sample: {e}")

def check_length_bias(max_proteins=200):
    """
    Checks whether embedding distances are spuriously correlated with protein
    length differences. A Spearman rho above ~0.3 suggests the encoder is
    encoding sequence length rather than structure.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"\n[Length Bias Check] Embeddings file '{EMBEDDINGS_FILE}' not found. Skipping.")
        return

    try:
        embeddings_dict = torch.load(EMBEDDINGS_FILE, map_location='cpu', weights_only=False)
    except TypeError:
        embeddings_dict = torch.load(EMBEDDINGS_FILE, map_location='cpu')

    # Load protein lengths from processed .pt files
    protein_ids = list(embeddings_dict.keys())
    if len(protein_ids) > max_proteins:
        print(f"\n[Length Bias Check] Subsampling {max_proteins} proteins from {len(protein_ids)}...")
        protein_ids = random.sample(protein_ids, max_proteins)

    lengths = {}
    for pid in tqdm(protein_ids, desc="[Length Bias] Loading sizes"):
        pt_path = os.path.join(PROC_DIR, f"{pid}.pt")
        if not os.path.exists(pt_path):
            continue
        try:
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(pt_path, map_location='cpu')
            lengths[pid] = int(data['rank3'].get('protein_size', 0))
        except Exception:
            pass

    # Only keep IDs that have both an embedding and a known length
    valid_ids = [pid for pid in protein_ids if pid in lengths and lengths[pid] > 0]
    if len(valid_ids) < 2:
        print("[Length Bias Check] Not enough valid proteins to evaluate. Skipping.")
        return

    emb_dists = []
    len_diffs = []

    for pid1, pid2 in tqdm(list(combinations(valid_ids, 2)), desc="[Length Bias] Pairwise distances"):
        emb1 = embeddings_dict[pid1]
        emb2 = embeddings_dict[pid2]
        # Convert to numpy float64 for scipy
        v1 = emb1.cpu().numpy() if torch.is_tensor(emb1) else emb1
        v2 = emb2.cpu().numpy() if torch.is_tensor(emb2) else emb2
        v1 = v1.astype(np.float64).ravel()
        v2 = v2.astype(np.float64).ravel()

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        emb_dists.append(cosine(v1, v2))
        len_diffs.append(abs(lengths[pid1] - lengths[pid2]))

    if len(emb_dists) < 10:
        print("[Length Bias Check] Too few valid pairs to compute correlation. Skipping.")
        return

    rho, p_val = spearmanr(emb_dists, len_diffs)

    print("\n" + "="*40)
    print("--- Length Bias Diagnosis ---")
    print("="*40)
    print(f"Proteins evaluated:  {len(valid_ids)}")
    print(f"Pairs evaluated:     {len(emb_dists)}")
    print(f"Spearman rho:        {rho:.4f}")
    print(f"p-value:             {p_val:.4e}")

    if abs(rho) > 0.3:
        print(f"\n⚠️  SUSPICIOUS: |rho| = {abs(rho):.4f} > 0.3")
        print("    Embeddings appear to encode protein length, not structure.")
        print("    Consider length-normalizing inputs or adding a length-decorrelation loss.")
    else:
        print(f"\n✅  OK: |rho| = {abs(rho):.4f} <= 0.3  (no strong length bias detected)")


if __name__ == '__main__':
    #run_sanity_check()
    check_length_bias()