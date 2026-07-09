"""Foldseek interactive HTML for a controversial reference cluster.

Given a split-export reference-cluster folder (``ref_<CID>/`` with ``model_cluster_*.fasta``
+ ``model_noise.fasta``), extract its member PDBs from the virome zip — renamed
``<modeltag>__<name>.pdb`` so the model sub-cluster is visible in every hit — run
Foldseek all-vs-all, and emit the Pretty-HTML report (--format-mode 3). Lets you see
whether the model's split of the reference fold is structurally justified (distinct
sub-structures) or over-splitting (one fold cut into pieces).

Usage (on the box, from repo root):
  .venv/bin/python scripts/analysis/foldseek_cluster_html.py \
      --ref-dir clusters/split_ep25_meancase/ref_264 \
      --pdb-zip /data/pnardi/hoan_raw_pdb/virome_pdbs.zip \
      --out-dir clusters/foldseek/ref_264
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import zipfile


def read_members(ref_dir):
    """{protein_name: model_tag} from the per-piece FASTAs (m<K> or 'noise')."""
    members = {}
    for f in glob.glob(os.path.join(ref_dir, "model_*.fasta")):
        base = os.path.basename(f)
        if "noise" in base:
            tag = "noise"
        else:
            tag = "m" + re.search(r"model_cluster_(\d+)", base).group(1)
        for line in open(f):
            if line.startswith(">"):
                members[line[1:].strip().split("|")[-1]] = tag
    return members


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", required=True, help="ref_<CID> folder from the split export.")
    ap.add_argument("--pdb-zip", default="/data/pnardi/hoan_raw_pdb/virome_pdbs.zip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--foldseek", default="foldseek")
    ap.add_argument("--evalue", default="10", help="Foldseek -e (lenient shows more hits).")
    args = ap.parse_args()

    members = read_members(args.ref_dir)
    print(f"{len(members)} members in {args.ref_dir}")

    pdb_out = os.path.join(args.out_dir, "pdbs")
    os.makedirs(pdb_out, exist_ok=True)
    z = zipfile.ZipFile(args.pdb_zip)
    index = {re.sub(r"\.pdb$", "", os.path.basename(n)): n
             for n in z.namelist() if n.endswith(".pdb")}
    n_ok = 0
    for name, tag in members.items():
        zp = index.get(name)
        if zp is None:
            print(f"  missing PDB: {name}")
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:120]
        with open(os.path.join(pdb_out, f"{tag}__{safe}.pdb"), "wb") as fh:
            fh.write(z.read(zp))
        n_ok += 1
    z.close()
    print(f"extracted {n_ok} PDBs (tagged by model piece) -> {pdb_out}")

    html = os.path.join(args.out_dir, "foldseek_result.html")
    tmp = os.path.join(args.out_dir, "tmp")
    cmd = [args.foldseek, "easy-search", pdb_out, pdb_out, html, tmp,
           "--format-mode", "3", "-e", args.evalue, "--exhaustive-search", "1"]
    print("running:", " ".join(cmd)); sys.stdout.flush()
    subprocess.run(cmd, check=True)
    print(f"Pretty-HTML report -> {html}")


if __name__ == "__main__":
    main()
