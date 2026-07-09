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


def load_nomburg_map(path):
    """member id (stripped) -> Nomburg cluster ID. Skips the 2 header rows."""
    m = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 4:
                m[strip_ext(cols[3])] = cols[0]
    return m


def export_split(ids, X, args):
    """For the --n-clusters largest reference (Nomburg/Foldseek) clusters, write ONE
    FOLDER per reference cluster containing:
      reference_cluster_<CID>.fasta   -- all members of the reference fold
      model_cluster_<K>.fasta         -- members the model (HDBSCAN, eps) put in cluster K
      model_noise.fasta               -- members left as noise (if any)
    i.e. the reference fold alongside the pieces the model shattered it into at this epoch.
    Reuses build_zip_index / get_sequence for the sequences (cached per protein)."""
    import hdbscan
    labels = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size, cluster_selection_epsilon=args.epsilon,
        metric="euclidean", algorithm="best", core_dist_n_jobs=-1,
    ).fit_predict(np.ascontiguousarray(X))
    name2model = {n: int(l) for n, l in zip(ids, labels)}
    n_clu = len({l for l in labels if l >= 0})
    print(f"HDBSCAN (eps={args.epsilon}): {n_clu} clusters, {(labels < 0).mean():.1%} noise")

    nom = load_nomburg_map(args.nomburg_tsv)
    ref = defaultdict(list)
    for n in ids:
        cid = nom.get(n)
        if cid is not None:
            ref[cid].append(n)

    # Per-cluster pair FNR = fraction of within-fold pairs the model leaves split
    # (noise members count as split) — the per-reference-cluster analogue of the
    # global pair FNR.
    def _cluster_fnr(mem):
        M = len(mem)
        if M < 2:
            return None
        by = defaultdict(int)
        for m in mem:
            by[name2model[m]] += 1
        tp = sum(c * (c - 1) // 2 for k, c in by.items() if k >= 0)
        return 1.0 - tp / (M * (M - 1) // 2)

    stats = [(cid, mem, _cluster_fnr(mem)) for cid, mem in ref.items()]
    stats = [s for s in stats if s[2] is not None]
    mean_fnr = sum(s[2] for s in stats) / len(stats)
    fnr_of = {cid: f for cid, mem, f in stats}
    print(f"Mean per-cluster FNR over {len(stats)} multi-member reference clusters: "
          f"{mean_fnr:.4f}")

    lo, hi = args.min_size, (args.max_size if args.max_size > 0 else 10 ** 9)
    cands = [s for s in stats if lo <= len(s[1]) <= hi]
    if args.select == "near-mean-fnr":
        cands.sort(key=lambda s: abs(s[2] - mean_fnr))     # closest to the mean first
    else:  # largest, optionally requiring real fragmentation
        cands = [s for s in cands
                 if (len({name2model[m] for m in s[1] if name2model[m] >= 0})
                     + (1 if any(name2model[m] < 0 for m in s[1]) else 0)) >= args.min_frag]
        cands.sort(key=lambda s: -len(s[1]))
    chosen = [(cid, mem) for cid, mem, f in cands[:args.n_clusters]]

    print(f"Indexing {args.pdb_zip} ...")
    z, zip_index = build_zip_index(args.pdb_zip)
    seq = {}
    for _, mem in chosen:
        for n in mem:
            if n in seq or n not in zip_index:
                continue
            try:
                seq[n] = get_sequence(z, zip_index[n])
            except Exception as e:
                print(f"  warn: seq {n}: {e}")

    def _write(path, members):
        lines = []
        for n in members:
            s = seq.get(n)
            if not s:
                continue
            lines.append(f">{n}")
            lines += [s[i:i + 60] for i in range(0, len(s), 60)]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def _write_grouped(path, tagged):
        """One combined FASTA, sequences ordered by model group with a group tag
        prefixed on each header (`>m0349|name`, `>noise|name`). In an alignment
        viewer the groups form contiguous, labelled blocks (and sorting by name
        keeps them together)."""
        lines = []
        for tag, n in tagged:
            s = seq.get(n)
            if not s:
                continue
            lines.append(f">{tag}|{n}")
            lines += [s[i:i + 60] for i in range(0, len(s), 60)]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    os.makedirs(args.out_dir, exist_ok=True)
    for cid, mem in chosen:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(cid))
        folder = os.path.join(args.out_dir, f"ref_{safe}")
        os.makedirs(folder, exist_ok=True)
        _write(os.path.join(folder, f"reference_cluster_{safe}.fasta"), mem)
        by_model = defaultdict(list)
        for m in mem:
            by_model[name2model[m]].append(m)
        parts = []
        for k in sorted(by_model, key=lambda k: (k < 0, k)):
            fn = "model_noise.fasta" if k < 0 else f"model_cluster_{k:04d}.fasta"
            _write(os.path.join(folder, fn), by_model[k])
            parts.append(f"{'noise' if k < 0 else k}:{len(by_model[k])}")
        # Combined FASTA for alignment: sequences grouped by model cluster (biggest
        # piece first, noise last), each header tagged with its group.
        tagged = []
        for k in sorted(by_model, key=lambda k: (k < 0, -len(by_model[k]), k)):
            tag = "noise" if k < 0 else f"m{k:04d}"
            tagged += [(tag, m) for m in by_model[k]]
        _write_grouped(os.path.join(folder, f"reference_cluster_{safe}.by_model.fasta"), tagged)
        print(f"  ref {cid}: {len(mem)} members, FNR={fnr_of[cid]:.3f} -> "
              f"{len(by_model)} pieces [{', '.join(parts)}]")
    z.close()
    print(f"Done -> {args.out_dir}/  ({len(chosen)} reference clusters)")


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
    ap.add_argument("--split-by-model", action="store_true",
                    help="Instead of flat per-cluster FASTA: for the N largest reference "
                         "(Nomburg) clusters, write one FOLDER per reference cluster holding "
                         "the reference FASTA + a FASTA per model HDBSCAN cluster it splits into.")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="HDBSCAN cluster_selection_epsilon for --split-by-model (default 0.1).")
    ap.add_argument("--min-frag", type=int, default=1,
                    help="With --split-by-model + --select largest, only export reference clusters "
                         "that split into >= this many model pieces (default 1 = all).")
    ap.add_argument("--select", choices=["largest", "near-mean-fnr"], default="largest",
                    help="Which reference clusters to export with --split-by-model: 'largest' "
                         "(default) or 'near-mean-fnr' (per-cluster FNR closest to the mean — the "
                         "typical case). Combine with --min-size/--max-size to target smaller folds.")
    ap.add_argument("--min-size", type=int, default=0,
                    help="Only consider reference clusters with >= this many members present.")
    ap.add_argument("--max-size", type=int, default=0,
                    help="Only consider reference clusters with <= this many members (0 = no cap).")
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

    if args.split_by_model:
        export_split(ids, X, args)
        return

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
