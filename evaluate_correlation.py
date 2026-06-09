"""
Evaluate Continuous Distance Correlation
Validates the PoC by checking if latent embedding distances correlate 
negatively with empirical TM-scores (structural similarity).
"""
import os
import random
import torch
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from tqdm import tqdm

try:
    import tmtools
except ImportError:
    raise ImportError("Please install tmtools first: pip install tmtools")

EMBEDDINGS_FILE = './data/virome_embeddings.pt'
PROC_DIR = './data/hoan_processed'

# Same alphabet used in TopoteinLifter
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO"

def get_struct_data(pdb_id):
    """
    Extracts the C-alpha coordinates and amino acid sequence 
    directly from the processed PCC files.
    """
    pt_path = os.path.join(PROC_DIR, f"{pdb_id}.pt")
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(pt_path, map_location='cpu')
        
    # Get C-alpha coordinates as a numpy array
    ca_coords = data['rank0']['ca_coords'].cpu().numpy().astype(np.float32)
    
    # Decode the one-hot amino acid vectors back to a string sequence
    aa_one_hot = data['rank0']['aa'].cpu().numpy()
    seq = "".join([AA_ALPHABET[np.argmax(x)] for x in aa_one_hot])
    
    return ca_coords, seq

def evaluate_correlation(max_proteins=30):
    """
    Computes the pairwise latent distance and TM-score for a sample of the dataset,
    then evaluates the Spearman correlation.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Run extract_embeddings.py first.")
        return

    try:
        embeddings_dict = torch.load(EMBEDDINGS_FILE, map_location='cpu', weights_only=False)
    except TypeError:
        embeddings_dict = torch.load(EMBEDDINGS_FILE, map_location='cpu')
    protein_ids = list(embeddings_dict.keys())
    
    # Subsample if the dataset is too large, to keep the all-vs-all O(N^2) TM-align tractable
    if len(protein_ids) > max_proteins:
        print(f"Subsampling {max_proteins} proteins from a pool of {len(protein_ids)}...")
        protein_ids = random.sample(protein_ids, max_proteins)
        
    num_pairs = len(protein_ids) * (len(protein_ids) - 1) // 2
    print(f"Evaluating {len(protein_ids)} proteins ({num_pairs} pairs)...")

    # Preload all structures into memory
    structs = {}
    for pid in tqdm(protein_ids, desc="Loading structures"):
        try:
            structs[pid] = get_struct_data(pid)
        except Exception as e:
            print(f"Error loading {pid}: {e}")
            
    # Ensure we only keep IDs that successfully loaded
    protein_ids = [pid for pid in protein_ids if pid in structs]
    
    emb_dists = []
    tm_scores = []
    
    pairs = list(combinations(protein_ids, 2))
    for pid1, pid2 in tqdm(pairs, desc="Calculating alignments"):
        # 1. Latent Cosine Distance
        emb1 = embeddings_dict[pid1]
        emb2 = embeddings_dict[pid2]
        dist = cosine(emb1, emb2)
        
        # 2. TM-score
        coords1, seq1 = structs[pid1]
        coords2, seq2 = structs[pid2]
        
        try:
            # tm_align returns an object containing scores normalized by both chains
            res = tmtools.tm_align(coords1, coords2, seq1, seq2)
            # Take the maximum of the two normalized TM-scores to create a symmetric measure
            tm = max(res.tm_norm_chain1, res.tm_norm_chain2)
        except Exception:
            # Skip pairs that tmtools fails to align (e.g., highly unusual topologies or length mismatches)
            continue
            
        emb_dists.append(dist)
        tm_scores.append(tm)

    if not emb_dists:
        print("No valid pairs evaluated.")
        return

    # 3. Correlation Analysis
    rho, p_val = spearmanr(emb_dists, tm_scores)
    print("\n" + "="*40 + "\n--- Continuous Distance Correlation ---\n" + "="*40)
    print(f"Valid Pairs Evaluated: {len(emb_dists)}")
    print(f"Spearman Correlation (\u03C1): {rho:.4f}")
    print(f"p-value: {p_val:.4e}")
    
    if rho < -0.3:
        print("\n\u2705 Success: A strong negative correlation indicates the embeddings successfully map structural similarity (high TM-score = low latent distance)!")
    else:
        print("\n\u26A0\uFE0F Weak Correlation: The network might require more epochs, a larger training batch, or hyperparameter tuning.")

if __name__ == "__main__":
    evaluate_correlation()