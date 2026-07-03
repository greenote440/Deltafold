"""
DEPRECATED downsampler — kept for provenance only.

``refine_subdataset_by_clustering`` below produced the biased prototyping base
``data/subdataset_files_refined.txt`` (3 647 proteins) whose four biases are
catalogued in plan_experimentation_v2 §4: singletons removed (multi-member only),
size distribution flattened ("max 2 per cluster"), hand-picked folds (the
``target_keywords`` list), and selection⇄evaluation entanglement (those keyword
proteins are the downstream test cases).

Use ``scripts/utilities/build_corrected_subbase.py`` instead — it corrects all
four (proportional sampling, singletons kept, keyword-blind selection with
separate controls, cold cluster split + dedup) and writes
``data/subbase_corrected_{train,val,controls}.txt``, consumed in training via
``--split corrected`` (see ``train.get_corrected_split``).
"""
import re
import json
import tqdm
import zipfile
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import os


BASE_DIR = './data'
os.makedirs(BASE_DIR, exist_ok=True)
PROC_DIR = os.path.join(BASE_DIR, 'hoan_processed')
os.makedirs(PROC_DIR, exist_ok=True)
RAW_DIR = os.path.join(BASE_DIR, 'hoan_raw_pdb')
os.makedirs(RAW_DIR, exist_ok=True)

def get_plddt_from_pdb_content(content):
    """Extracts B-factors (pLDDT) from ATOM lines and returns the mean."""
    plddts = []
    for line in content.splitlines():
        if line.startswith("ATOM"):
            try:
                # B-factor is in columns 61-66
                plddt = float(line[60:66].strip())
                plddts.append(plddt)
            except (ValueError, IndexError):
                continue
    return np.mean(plddts) if plddts else 0.0

def create_stratified_subdataset(zip_path, plddt_threshold=70.0):
    print("Starting stratification process...")
    target_keywords = ['LigT', 'UL43', 'ENT4', 'I3L', 'SSB', 'phosphodiesterase', 'OB-fold']
    selected_files = []
    cluster_counts = defaultdict(int)

    with zipfile.ZipFile(zip_path, 'r') as archive:
        all_pdbs = [m for m in archive.namelist() if m.endswith('.pdb')]
        print(f"Scanning {len(all_pdbs)} structures...")

        for member_name in tqdm(all_pdbs, desc="Filtering"):
            # 1. Extract cluster info from path
            # Example path: viral_structures/family_name/protein_name.pdb
            parts = member_name.split('/')
            if len(parts) < 2: continue
            cluster_id = parts[-2]

            # Skip singletons by first pass or specific logic (here we track counts)
            # Optimization: In this dataset, singletons are often in 'undefined_family' or specific folders
            if cluster_id == 'undefined_family': continue

            try:
                with archive.open(member_name) as f:
                    content = f.read().decode('utf-8')

                # 2. Filter by pLDDT
                avg_plddt = get_plddt_from_pdb_content(content)
                if avg_plddt < plddt_threshold:
                    continue

                # 3. Check for specific target proteins (Phase 4 test cases)
                is_target = any(key.lower() in member_name.lower() for key in target_keywords)

                selected_files.append(member_name)
                cluster_counts[cluster_id] += 1

            except Exception as e:
                continue

    # 4. Final Filter: Keep only multi-member clusters (count > 1)
    final_selection = [f for f in selected_files if cluster_counts[f.split('/')[-2]] > 1]

    print(f"\nStratification Results:")
    print(f"- Initial count: {len(all_pdbs)}")
    print(f"- High-confidence multi-member count: {len(final_selection)}")

    # Save list to local storage
    subdataset_path = os.path.join(BASE_DIR, 'subdataset_files.txt')
    with open(subdataset_path, 'w') as f:
        for item in final_selection:
            f.write(f"{item}\n")

    print(f"Subdataset list saved to: {subdataset_path}")
    return final_selection

# Run the filter
zip_path = os.path.join(RAW_DIR, "virome_pdbs.zip")
#sub_files = create_stratified_subdataset(zip_path)

import os
import re
import random
from collections import defaultdict
from tqdm.auto import tqdm

def extract_accession(text):
    """Extracts the accession number (e.g., YP_010085741) from a string."""
    match = re.search(r'([A-Z]{1,2}_[0-9]{5,10})', text)
    return match.group(1) if match else text

def refine_subdataset_by_clustering(cluster_tsv_path, input_list_path, stride=1):
    """
    Refined Downsampling Logic:
    1. Filters for Multi-Member Clusters only (Internal variance).
    2. Implements Cluster Striding (Even sampling of structural universe).
    3. Samples max 2 proteins per selected cluster.
    """
    if not os.path.exists(cluster_tsv_path):
        print(f"Error: {cluster_tsv_path} not found.")
        return []

    # 1. Map Accession -> Full Path
    with open(input_list_path, 'r') as f:
        available_files = [line.strip() for line in f if line.strip()]

    acc_to_path = {extract_accession(os.path.basename(p)): p for p in available_files}

    # 2. Build Cluster Map
    cluster_map = defaultdict(list)
    print("Loading and filtering clusters...")
    with open(cluster_tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep = extract_accession(parts[0])
                member = extract_accession(parts[1])
                cluster_map[rep].append(member)

    # 3. Apply Multi-Member and Striding logic
    # Filter for clusters with > 1 member
    multi_member_reps = [rep for rep, members in cluster_map.items() if len(members) > 1]
    multi_member_reps.sort() # Ensure deterministic striding

    # Cluster Striding: select every N-th cluster
    strided_reps = multi_member_reps[::stride]

    target_keywords = ['LigT', 'UL43', 'ENT4', 'I3L', 'SSB']
    refined_selection = []
    processed_files = set()

    # Priority: Target Proteins
    print("Keeping target proteins...")
    for fn in available_files:
        if any(key.lower() in fn.lower() for key in target_keywords):
            refined_selection.append(fn)
            processed_files.add(fn)

    # Sampling: Strided Clusters
    print(f"Sampling {len(strided_reps)} strided clusters (Stride={stride})...")
    for rep_id in tqdm(strided_reps):
        members = cluster_map[rep_id]
        count = 0
        for member_id in members:
            matching_path = acc_to_path.get(member_id)
            if matching_path and matching_path not in processed_files:
                refined_selection.append(matching_path)
                processed_files.add(matching_path)
                count += 1
                if count >= 2: break

    print(f"\nDownsampling Results:")
    print(f"- Initial Clusters: {len(cluster_map)}")
    print(f"- Multi-member Clusters: {len(multi_member_reps)}")
    print(f"- Strided Clusters Selected: {len(strided_reps)}")
    print(f"- Final Protein Count: {len(refined_selection)}")

    final_path = os.path.join(BASE_DIR, 'subdataset_files_refined.txt')
    with open(final_path, 'w') as f:
        for item in refined_selection: f.write(f"{item}\n")

    global SUBDATASET_LIST_PATH
    SUBDATASET_LIST_PATH = final_path
    return refined_selection

# Run the strided cluster refinement
cluster_tsv = os.path.join(BASE_DIR, 'cluster.tsv')
input_list = os.path.join(BASE_DIR, 'subdataset_files.txt')
# Using stride=3 to reduce cluster count while maintaining coverage
efined_files = refine_subdataset_by_clustering(cluster_tsv, input_list, stride=1)