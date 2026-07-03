"""
Export clusters as FASTA files (one per cluster).

Two cluster sources:
  --source model    HDBSCAN clusters from the embedding file (default)
  --source nomburg  Nomburg merged clusters (merged_clusters.tax.tsv)

Sequences are parsed from the raw PDB files inside the virome zip using biotite
(Cα atom residue names -> standard one-letter codes).

Usage
-----
    # 10 largest HDBSCAN model clusters from ep4 embeddings
    python scripts/utilities/export_cluster_fastas.py \
        --emb data/emb_moco_geom_no3di_ep4.pt --n-clusters 10

    # 10 largest Nomburg clusters (proteins present in the embedding set)
    python scripts/utilities/export_cluster_fastas.py \
        --emb data/emb_moco_geom_no3di_ep4.pt --n-clusters 10 --source nomburg

    # Smallest multi-member model clusters
    python scripts/utilities/export_cluster_fastas.py \
        --emb data/emb_moco_geom_no3di_ep4.pt --n-clusters 10 --sort-by size_asc
"""
import argparse
import io
import os
import re
import zipfile
from collections import defaultdict

import numpy as np
import torch

PDB_ZIP       = "./data/hoan_raw_pdb/virome_pdbs.zip"
NOMBURG_TSV   = "./code_and_intermediate_data/intermediate_data/merged_clusters.tax.tsv"
OUT_DIR       = "./clusters/fasta"


def strip_ext(name):
    return re.sub(r'\.(pdb|pt)$', '', os.path.basename(name))


def build_zip_index(zip_path):
    """protein_name -> path inside zip."""
    z = zipfile.ZipFile(zip_path)
    index = {strip_ext(n): n for n in z.namelist() if n.endswith(".pdb")}
    return z, index


def get_sequence(z, zip_path):
    """Parse amino acid sequence from a PDB file in the zip via biotite Cα atoms."""
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.info as info
    pdb_bytes = z.read(zip_path)
    f = pdb_io.PDBFile.read(io.StringIO(pdb_bytes.decode("utf-8", errors="replace")))
    structure = pdb_io.get_structure(f, model=1)
    ca = structure[structure.atom_name == "CA"]
    return "".join(info.one_letter_code(r) or "X" for r in ca.res_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True,
                    help="Embeddings .pt file (dict: protein_name -> numpy array).")
    ap.add_argument("--pdb-zip", default=PDB_ZIP,
                    help="Path to virome_pdbs.zip.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-clusters", type=int, default=10,
                    help="Number of model clusters to export.")
    ap.add_argument("--sort-by", choices=["size_desc", "size_asc", "id"],
                    default="size_desc",
                    help="size_desc = largest first (default), size_asc = smallest "
                         "multi-member first, id = by cluster index.")
    ap.add_argument("--min-cluster-size", type=int, default=2,
                    help="HDBSCAN min_cluster_size (default 2). Ignored for --source nomburg.")
    ap.add_argument("--source", choices=["model", "nomburg"], default="model",
                    help="Cluster source: 'model' = HDBSCAN on embeddings (default); "
                         "'nomburg' = Nomburg merged clusters from merged_clusters.tax.tsv.")
    ap.add_argument("--nomburg-tsv", default=NOMBURG_TSV)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and L2-normalise embeddings
    try:
        raw = torch.load(args.emb, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(args.emb, map_location="cpu")
    ids = sorted(raw.keys())
    X = np.stack([np.asarray(raw[i], dtype=np.float64) for i in ids])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    print(f"Loaded {len(ids)} embeddings from {args.emb}")

    # 2. Build clusters
    clusters = defaultdict(list)
    if args.source == "nomburg":
        member_to_cluster = {}
        with open(args.nomburg_tsv) as f:
            for i, line in enumerate(f):
                if i < 2:
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 4:
                    member_to_cluster[strip_ext(cols[3])] = cols[0]
        for name in ids:
            cid = member_to_cluster.get(name)
            if cid is not None:
                clusters[cid].append(name)
        print(f"Nomburg: {len(clusters)} clusters represented in the embedding set")
    else:
        import hdbscan
        labels = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric="euclidean", algorithm="best", core_dist_n_jobs=-1,
        ).fit_predict(np.ascontiguousarray(X))
        n_noise = 0
        for name, label in zip(ids, labels):
            if label == -1:
                n_noise += 1
            else:
                clusters[int(label)].append(name)
        print(f"HDBSCAN: {len(clusters)} clusters, {n_noise} noise points "
              f"({n_noise / len(ids):.1%})")

    # 3. Select clusters
    if args.sort_by == "size_desc":
        ranked = sorted(clusters.items(), key=lambda x: -len(x[1]))
    elif args.sort_by == "size_asc":
        ranked = sorted(clusters.items(), key=lambda x: len(x[1]))
    else:
        ranked = sorted(clusters.items(), key=lambda x: x[0])
    selected = ranked[:args.n_clusters]

    # 4. Open PDB zip and build index
    print(f"Indexing {args.pdb_zip} ...")
    z, zip_index = build_zip_index(args.pdb_zip)
    print(f"  {len(zip_index)} PDB files indexed.")

    # 5. Write FASTA files
    print(f"Exporting {len(selected)} clusters to {args.out_dir}/")
    for cluster_id, members in selected:
        prefix = "nomburg_cluster" if args.source == "nomburg" else "model_cluster"
        cid_str = f"{cluster_id}" if args.source == "nomburg" else f"{cluster_id:04d}"
        fasta_path = os.path.join(args.out_dir, f"{prefix}_{cid_str}.fasta")
        lines = []
        missing = 0
        for name in members:
            zip_path = zip_index.get(name)
            if zip_path is None:
                missing += 1
                continue
            try:
                seq = get_sequence(z, zip_path)
            except Exception as e:
                print(f"  Warning: could not parse {name}: {e}")
                continue
            lines.append(f">{name}")
            for i in range(0, len(seq), 60):
                lines.append(seq[i:i+60])
        with open(fasta_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        note = f" ({missing} missing)" if missing else ""
        print(f"  cluster {str(cluster_id):>6}: {len(members):4d} members{note} -> "
              f"{os.path.basename(fasta_path)}")

    z.close()


if __name__ == "__main__":
    main()
