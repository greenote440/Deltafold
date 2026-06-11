import os
import argparse
import zipfile
import logging
import gc
import tempfile
import subprocess
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import torch
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BASE_DIR = Path('./data')
RAW_ZIP = BASE_DIR / 'hoan_raw_pdb' / 'virome_pdbs.zip'
REFINED_LIST = BASE_DIR / 'subdataset_files_refined.txt'
PROC_DIR = BASE_DIR / 'hoan_processed'
PROC_DIR.mkdir(parents=True, exist_ok=True)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO" # 22 + Unknown (X) = 23
FOLDSEEK_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX" # 20 standard + X = 21
POS_ENC_DIM = 16  # sine/cosine positional-encoding width (must match the rank0 embedding)

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

class TopoteinLifter:
    def __init__(self, lift_residue=False, lift_positional=False):
        # Defaults match the training config we aim for (--no-residue --no-positional):
        # residue identity and positional encoding are stored as zero placeholders unless
        # explicitly requested. The model zeros these features anyway under those flags,
        # so a zeroed placeholder is bit-for-bit equivalent at train time while keeping the
        # rank0 feature width fixed at 23+21+4+16=64 (no loader/model change needed).
        self.lift_residue = lift_residue
        self.lift_positional = lift_positional

    def dict_to_cpu(self, obj):
        """Recursively moves tensors back to CPU to prevent MPS serialization warnings."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number):
                return torch.from_numpy(obj).float().cpu()
            return obj
        elif isinstance(obj, dict):
            return {k: self.dict_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], dict):
                return [self.dict_to_cpu(v) for v in obj]
            try:
                arr = np.array(obj)
                if np.issubdtype(arr.dtype, np.number):
                    return torch.from_numpy(arr).float().cpu()
            except Exception:
                pass
            return [self.dict_to_cpu(v) for v in obj]
        return obj

    def _one_hot(self, char, alphabet):
        """Creates a one-hot encoded vector for a given character and alphabet."""
        char = char.upper() if char else 'X'
        idx = alphabet.find(char)
        if idx == -1:
            idx = len(alphabet) - 1 if 'X' in alphabet else 0
        vec = np.zeros(len(alphabet) + (1 if 'X' not in alphabet else 0))
        vec[idx] = 1.0
        return vec

    def get_positional_encoding(self, positions, d_model=POS_ENC_DIM):
        """Generates sine/cosine positional encodings natively on PyTorch MPS."""
        if isinstance(positions, torch.Tensor):
            positions = positions.view(-1, 1).float()
        else:
            positions = torch.tensor(positions, device=DEVICE, dtype=torch.float32).view(-1, 1)
            
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(positions.shape[0], d_model, device=DEVICE)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe

    def get_pca_features(self, coords):
        """Computes Eigenvalues, Eigenvectors, and Shape Descriptors purely on MPS."""
        if coords.shape[0] < 3:
            return torch.zeros(3, device=DEVICE), torch.eye(3, device=DEVICE), torch.zeros(5, device=DEVICE)
        
        com = coords.mean(dim=0)
        centered = coords - com
        cov_matrix = (centered.T @ centered) / (coords.shape[0] - 1)
        
        # torch.linalg.eigh is not yet implemented natively for MPS.
        # Explicitly compute this step on the CPU and move back to DEVICE.
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.cpu())
        eigenvalues, eigenvectors = eigenvalues.to(DEVICE), eigenvectors.to(DEVICE)
        
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        e1, e2, e3 = eigenvalues + 1e-8
        
        linearity = (e1 - e2) / e1
        planarity = 2 * (e2 - e3) / (e1 + e2)
        scattering = 3 * e3 / (e1 + e2 + e3)
        omnivariance = torch.sign(e1 * e2 * e3) * torch.abs(e1 * e2 * e3).pow(1.0/3.0)
        anisotropy = (e1 - e3) / e1
        
        shape_descriptors = torch.stack([linearity, planarity, scattering, omnivariance, anisotropy])
        return eigenvalues, eigenvectors, shape_descriptors

    def run_foldseek(self, pdb_file_path, tmp_dir):
        """Runs actual foldseek binary to extract the 3Di structural alphabet, with error catching."""
        import shutil  # Ensure shutil is imported
        
        db_path = os.path.join(tmp_dir, 'db')
        fasta_path = os.path.join(tmp_dir, 'db_ss.fasta')
        
        # Pass the current environment so spawned processes can find foldseek
        env = os.environ.copy()
        
        try:
            # 1. Create DB
            res = subprocess.run(["foldseek", "createdb", pdb_file_path, db_path], 
                           capture_output=True, text=True, env=env)
            if res.returncode != 0:
                logging.error(f"Foldseek createdb failed: {res.stderr}")
                return ""
            
            # --- THE FIX: Explicitly clone the amino acid headers for the 3Di database ---
            # This prevents the "needs header information" error on macOS/temp file systems
            for ext in ["_h", "_h.index", "_h.dbtype"]:
                src = db_path + ext
                dst = db_path + "_ss" + ext
                if os.path.exists(src):
                    if os.path.lexists(dst):
                        os.remove(dst)
                    shutil.copy(src, dst)
            # -----------------------------------------------------------------------------
            
            # 2. Convert to FASTA
            if os.path.exists(db_path + "_ss"):
                res = subprocess.run(["foldseek", "convert2fasta", db_path + "_ss", fasta_path], 
                               capture_output=True, text=True, env=env)
                if res.returncode != 0:
                    logging.error(f"Foldseek convert2fasta failed: {res.stderr}")
                    return ""
                    
                # 3. Safely parse the FASTA (handling multi-line wrapping)
                if os.path.exists(fasta_path):
                    with open(fasta_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            # Join all sequence lines, ignoring the >header
                            seq = "".join([l.strip() for l in lines[1:] if not l.startswith('>')])
                            return seq
        except FileNotFoundError:
            logging.error("Foldseek binary not found. Is it in your PATH?")
        except Exception as e:
            logging.error(f"Foldseek execution error: {e}")
            
        return ""

    def lift_rank0_residues(self, chain, foldseek_seq, phi_array, psi_array):
        """
        Rank 0: Residue Level
        - Always lifted: 3Di one-hot, Phi/Psi, Ca coords.
        - aa one-hot (residue identity) and sine/cosine positional encoding are lifted
          only when self.lift_residue / self.lift_positional are set; otherwise they are
          stored as zero placeholders of the correct shape (N,23) / (N,16). The shapes are
          preserved so downstream code (rank0 embedding width, r0['aa'].shape[0] as the
          residue count, torch.zeros_like(r0['aa'])) is unchanged.
        """
        features = {'aa': [], '3di': [], 'phi_psi': [], 'ca_coords': []}
        valid_residues = []

        for i, res_atoms in enumerate(struc.residue_iter(chain)):
            ca_mask = res_atoms.atom_name == "CA"
            if not np.any(ca_mask):
                continue

            ca_atom = res_atoms[ca_mask][0]
            res_name = ca_atom.res_name

            aa_char = AA_3_TO_1.get(res_name, 'X')

            di_char = foldseek_seq[i] if i < len(foldseek_seq) else 'X'

            # Biotite pads the ends with NaNs, convert them to 0.0
            phi = phi_array[i] if (i < len(phi_array) and not np.isnan(phi_array[i])) else 0.0
            psi = psi_array[i] if (i < len(psi_array) and not np.isnan(psi_array[i])) else 0.0

            if self.lift_residue:
                features['aa'].append(self._one_hot(aa_char, AA_ALPHABET))
            features['3di'].append(self._one_hot(di_char, FOLDSEEK_ALPHABET))
            features['phi_psi'].append([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)])
            features['ca_coords'].append(ca_atom.coord)

            valid_residues.append(res_name)

        n = len(valid_residues)
        features['ca_coords'] = torch.tensor(np.array(features['ca_coords']), dtype=torch.float32, device=DEVICE)
        features['3di'] = torch.tensor(np.array(features['3di']), dtype=torch.float32, device=DEVICE)
        features['phi_psi'] = torch.tensor(np.array(features['phi_psi']), dtype=torch.float32, device=DEVICE)

        # aa: real one-hot, or a zeroed (N, 23) placeholder ('X' adds a catch-all column).
        aa_dim = len(AA_ALPHABET) + 1
        if self.lift_residue:
            features['aa'] = torch.tensor(np.array(features['aa']), dtype=torch.float32, device=DEVICE)
        else:
            features['aa'] = torch.zeros((n, aa_dim), dtype=torch.float32, device=DEVICE)

        # positional encoding: real sine/cosine, or a zeroed (N, 16) placeholder.
        if self.lift_positional:
            features['positional_encoding'] = self.get_positional_encoding(torch.arange(n, device=DEVICE))
        else:
            features['positional_encoding'] = torch.zeros((n, POS_ENC_DIM), dtype=torch.float32, device=DEVICE)

        return features, valid_residues

    def lift_rank1_edges(self, ca_coords, k=16):
        """
        Rank 1: Interactions
        Vectorized heavily for MPS: pairwise distances and topk lookups.
        """
        n_res = ca_coords.shape[0]
        k_actual = min(k + 1, n_res)
        
        dist_matrix = torch.cdist(ca_coords, ca_coords)
        distances, indices = torch.topk(dist_matrix, k=k_actual, largest=False, dim=1)
        
        # Exclude self-loop (the closest neighbor is always the residue itself, d=0)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        sources = torch.arange(n_res, device=DEVICE).view(-1, 1).expand(-1, k_actual - 1)
        vectors = ca_coords[indices] - ca_coords[sources]
        
        dist_flat = distances.flatten()
        dist_enc = self.get_positional_encoding(dist_flat).view(n_res, k_actual - 1, -1)
        
        return {
            'source': sources,
            'target': indices,
            'distance': distances,
            'distance_encoding': dist_enc,
            'vector': vectors
        }

    def lift_rank2_secondary_structures(self, ca_coords, dssp_seq):
        """
        Rank 2: Secondary Structures
        - Contiguous DSSP segmentation, Eigenvalues, Shape Descriptors, Frame Vectors
        """
        n_res = ca_coords.shape[0]
        sse_list = []
        
        if not dssp_seq or n_res == 0:
            return sse_list
            
        # Group contiguous states
        current_ss = dssp_seq[0]
        start_idx = 0
        
        segments = []
        for i in range(min(n_res, len(dssp_seq))):
            if dssp_seq[i] != current_ss:
                segments.append((current_ss, start_idx, i-1))
                current_ss = dssp_seq[i]
                start_idx = i
        segments.append((current_ss, start_idx, min(n_res, len(dssp_seq))-1))

        for ss_type, s_idx, e_idx in segments:
            sse_coords = ca_coords[s_idx:e_idx+1]
            
            eigenvalues, eigenvectors, shape_desc = self.get_pca_features(sse_coords)
            com = sse_coords.mean(dim=0) if len(sse_coords) > 0 else torch.zeros(3, device=DEVICE)
            
            vec_start = sse_coords[0] - com if len(sse_coords) > 0 else torch.zeros(3, device=DEVICE)
            vec_mid = sse_coords[len(sse_coords)//2] - com if len(sse_coords) > 0 else torch.zeros(3, device=DEVICE)
            vec_end = sse_coords[-1] - com if len(sse_coords) > 0 else torch.zeros(3, device=DEVICE)
            
            sse_list.append({
                'type': torch.tensor(self._one_hot(ss_type, "HEC"), dtype=torch.float32, device=DEVICE),
                'size': len(sse_coords),
                'start_idx': s_idx,
                'end_idx': e_idx,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'shape_descriptors': shape_desc,
                'vec_start': vec_start,
                'vec_mid': vec_mid,
                'vec_end': vec_end,
                'com': com
            })
            
        return sse_list

    def lift_rank3_global(self, ca_coords, residues):
        """
        Rank 3: Global Protein
        - Size, Global PCA, Shape Descriptors, Radius of Gyration
        """
        global_com = ca_coords.mean(dim=0)
        centered_coords = ca_coords - global_com
        
        # Radius of Gyration
        rg = torch.sqrt(torch.mean(torch.sum(centered_coords**2, dim=1)))
        
        # Global PCA
        eigenvalues, eigenvectors, shape_desc = self.get_pca_features(ca_coords)
        
        distances_to_com = torch.norm(centered_coords, dim=1)
        sorted_indices = torch.argsort(distances_to_com)
        
        return {
            'protein_size': len(residues),
            'radius_of_gyration': rg,
            'global_eigenvalues': eigenvalues,
            'global_shape_descriptors': shape_desc,
            'global_eigenvectors': eigenvectors,
            'nearest_10_vecs': centered_coords[sorted_indices[:10]],
            'furthest_10_vecs': centered_coords[sorted_indices[-10:]]
        }

    def process_pdb(self, pdb_content, pdb_name):
        """Executes the hierarchical lifting pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_pdb = os.path.join(tmp_dir, 'temp.pdb')
            with open(tmp_pdb, 'w') as f:
                f.write(pdb_content)
                
            # Run External Binaries
            foldseek_seq = self.run_foldseek(tmp_pdb, tmp_dir)
            if not foldseek_seq:
                logging.error(f"Foldseek failed for {pdb_name}. Skipping save.")
                return None
                
            try:
                pdb_file = pdb.PDBFile.read(tmp_pdb)
                array = pdb.get_structure(pdb_file, model=1)
                if array.array_length() == 0:
                    return None
                    
                chain_ids = np.unique(array.chain_id)
                if len(chain_ids) == 0:
                    return None
                    
                chain = array[array.chain_id == chain_ids[0]]
                chain = chain[~chain.hetero] # Filter out heteroatoms like water/ligands
                
                # --- NEW BIOTITE NATIVE LOGIC ---
                # 1. Native Dihedral Angles (Returns arrays of len N)
                phi, psi, omega = struc.dihedral_backbone(chain)
                
                # 2. Native Secondary Structure (Returns array of 'a', 'b', 'c')
                try:
                    sse_raw = struc.annotate_sse(chain)
                    # Map Biotite states to Topotein (HEC) states
                    dssp_seq = ['H' if s == 'a' else 'E' if s == 'b' else 'C' for s in sse_raw]
                except Exception as e:
                    logging.error(f"Biotite SSE failed for {pdb_name}: {e}. Skipping save.")
                    return None
                # --------------------------------
                
                # Build Pipeline
                rank0, residues = self.lift_rank0_residues(chain, foldseek_seq, phi, psi)
                ca_coords = rank0['ca_coords']
                
                if len(ca_coords) < 16:
                    logging.warning(f"PDB {pdb_name} is too small (<16 residues). Skipping.")
                    return None
                    
                rank1 = self.lift_rank1_edges(ca_coords)
                
                rank2 = self.lift_rank2_secondary_structures(ca_coords, dssp_seq)
                rank3 = self.lift_rank3_global(ca_coords, residues)
                
                logging.debug(f"Lifted {pdb_name} - Size: {rank3['protein_size']}, RoG: {rank3['radius_of_gyration']:.2f}")
                
                return {
                    'rank0': rank0,
                    'rank1': rank1,
                    'rank2': rank2,
                    'rank3': rank3
                }
                
            except Exception as e:
                logging.error(f"Failed to process {pdb_name}: {e}")
                return None

# --- Worker state (set per-process by _init_worker; spawn re-imports the module, so
# main()'s locals don't reach the workers — config has to be passed via initargs). ---
LIFT_RESIDUE = False
LIFT_POSITIONAL = False
OUT_DIR = PROC_DIR
_WORKER_ZIP = None
_WORKER_BATCH_COUNT = 0


def _init_worker(lift_residue, lift_positional, out_dir, device=None):
    global LIFT_RESIDUE, LIFT_POSITIONAL, OUT_DIR, _WORKER_BATCH_COUNT, DEVICE
    LIFT_RESIDUE = lift_residue
    LIFT_POSITIONAL = lift_positional
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _WORKER_BATCH_COUNT = 0
    if device is not None:
        # Per-protein tensor ops are tiny (PCA eigh is already CPU-forced); CPU avoids each
        # worker spinning up its own MPS context and the dispatch overhead of many small
        # kernels, which is usually a net win when lifting in parallel.
        DEVICE = torch.device(device)


def _get_zip():
    """Open the raw archive once per worker and reuse the handle (each process gets its
    own handle, so concurrent reads across workers are safe)."""
    global _WORKER_ZIP
    if _WORKER_ZIP is None:
        _WORKER_ZIP = zipfile.ZipFile(RAW_ZIP, 'r')
    return _WORKER_ZIP


def process_task(internal_paths):
    """Read a batch of PDBs from the archive inside the worker and lift them."""
    global _WORKER_BATCH_COUNT
    success_count = 0
    lifter = TopoteinLifter(lift_residue=LIFT_RESIDUE, lift_positional=LIFT_POSITIONAL)
    zip_ref = _get_zip()
    
    for internal_path in internal_paths:
        pdb_filename = os.path.basename(internal_path)
        try:
            pdb_content = zip_ref.read(internal_path).decode('utf-8')
        except Exception as e:
            logging.error(f"Error reading {pdb_filename} from zip: {e}")
            continue

        features = lifter.process_pdb(pdb_content, pdb_filename)
        if features:
            features_cpu = lifter.dict_to_cpu(features)
            pt_filename = Path(pdb_filename).with_suffix('.pt').name
            # Atomic write: torch.save to a temp file, then os.replace (atomic rename).
            # A concurrent reader (e.g. a training DataLoader scanning the same dir) thus
            # never sees a half-written .pt, and an interrupted run leaves no truncated file.
            final_path = OUT_DIR / pt_filename
            tmp_path = OUT_DIR / f"{pt_filename}.tmp.{os.getpid()}"
            try:
                torch.save(features_cpu, tmp_path)
                os.replace(tmp_path, final_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
            success_count += 1
            
    _WORKER_BATCH_COUNT += 1
    if _WORKER_BATCH_COUNT % 30 == 0:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    return success_count


def collect_paths(downsampled, max_protein):
    """Return the list of .pdb entry paths to lift. Whole archive by default; restricted to
    the refined downsampled list when downsampled=True. Returns paths only (not contents),
    so the parent process stays memory-flat."""
    if not RAW_ZIP.exists():
        logging.error(f"Raw zip file not found at {RAW_ZIP}.")
        return []

    target = None
    if downsampled:
        if not REFINED_LIST.exists():
            logging.error(f"Refined list not found at {REFINED_LIST}. Run the filter script first.")
            return []
        with open(REFINED_LIST, 'r') as f:
            target = set(line.strip() for line in f if line.strip())
        logging.info(f"Downsampled mode: {len(target)} target PDBs from {REFINED_LIST.name}.")
    else:
        logging.info("Whole-dataset mode: lifting every .pdb in the archive.")

    paths = []
    with zipfile.ZipFile(RAW_ZIP, 'r') as archive:
        for internal_path in archive.namelist():
            if not internal_path.endswith('.pdb'):
                continue
            if target is not None and internal_path not in target:
                continue
            paths.append(internal_path)
            if max_protein is not None and len(paths) >= max_protein:
                break
    return paths


def main():
    ap = argparse.ArgumentParser(
        description="Topotein lifting pipeline: PDB -> hierarchical .pt features.")
    ap.add_argument('--downsampled', action='store_true',
                    help="Lift only the refined downsampled subset (subdataset_files_refined.txt). "
                         "Default: lift the whole archive (~67,715 PDBs).")
    ap.add_argument('--lift-residue', action='store_true',
                    help="Lift real amino-acid identity (aa one-hot). Default OFF -> stored as a "
                         "zero (N,23) placeholder, matching the --no-residue training config.")
    ap.add_argument('--lift-positional', action='store_true',
                    help="Lift real sine/cosine positional encoding. Default OFF -> stored as a "
                         "zero (N,16) placeholder, matching the --no-positional training config.")
    ap.add_argument('--out-dir', default=str(PROC_DIR),
                    help="Output directory for the lifted .pt files (default: data/hoan_processed).")
    ap.add_argument('--max-protein', type=int, default=None,
                    help="Cap the number of PDBs processed (for testing).")
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="Parallel worker processes.")
    ap.add_argument('--skip-existing', action='store_true',
                    help="Skip PDBs whose .pt already exists in --out-dir (resume a big run).")
    ap.add_argument('--unprocessed-only', action='store_true',
                    help="Lift only not yet processed proteins (alias for --skip-existing).")
    ap.add_argument('--batch-size', type=int, default=32,
                    help="Number of PDBs to process per batch (default: 32).")
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu',
                    help="Torch device for the per-protein tensor ops. Default cpu: faster "
                         "here (tiny ops, no per-worker MPS init / GPU contention).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_paths(downsampled=args.downsampled, max_protein=args.max_protein)
    if not paths:
        return

    if args.skip_existing or args.unprocessed_only:
        before = len(paths)
        paths = [p for p in paths
                 if not (out_dir / Path(os.path.basename(p)).with_suffix('.pt').name).exists()]
        logging.info(f"unprocessed-only / skip-existing: {before - len(paths)} already lifted, {len(paths)} remaining.")

    batches = [paths[i:i + args.batch_size] for i in range(0, len(paths), args.batch_size)]

    logging.info(f"Lifting {len(paths)} PDBs in {len(batches)} batches of {args.batch_size} | "
                 f"lift_residue={args.lift_residue} lift_positional={args.lift_positional} | "
                 f"out={out_dir} | workers={args.workers}")

    ctx = mp.get_context('spawn')
    processed_count = 0
    
    # Use multiprocessing.Pool with maxtasksperchild to completely eliminate
    # linear memory leaks by forcing the OS to replace worker processes every 10 batches.
    with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.lift_residue, args.lift_positional, str(out_dir), args.device),
            maxtasksperchild=10) as pool:
            
        for result in tqdm(pool.imap_unordered(process_task, batches), total=len(batches),
                           desc="Lifting PDBs (Batched/MPS)"):
            if result:
                processed_count += result

    logging.info(f"Topotein Lifting Pipeline finished. Mapped {processed_count}/{len(paths)} "
                 f"structures into {out_dir}.")


if __name__ == "__main__":
    # Prerequisites:
    # 1. conda env ml_env (torch, biotite, numpy, scipy)
    # 2. Foldseek binary available in system PATH
    # Examples:
    #   python topotein_lifter.py                          # whole archive, zeroed aa/pe
    #   python topotein_lifter.py --downsampled            # only the 3,647 refined subset
    #   python topotein_lifter.py --lift-residue --lift-positional   # include real aa + PE
    main()