"""Download the Nomburg (2024) eukaryotic-virome predicted-structure archive.

Saves under the shared DeltaFold data root (see deltafold_paths.py):
  * local:          ./data/hoan_raw_pdb/virome_pdbs.zip
  * --deltafold:    /data/pnardi/hoan_raw_pdb/virome_pdbs.zip  (the box's big volume)

Uses aria2c (16 parallel connections, resumable). Override the root explicitly
with --data-dir /path or DELTAFOLD_DATA_DIR=/path.
"""
import os
import sys
import argparse
import zipfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import deltafold_paths

ZENODO_URL = "https://zenodo.org/records/10291581/files/Nomburg_2023_structures.zip?download=1"


def download_pdbs_fast(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "virome_pdbs.zip")

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r'):
                print(f"Valid zip already present at {zip_path}; nothing to do.")
                return
        except zipfile.BadZipFile:
            print("Existing zip is incomplete/corrupt; aria2c will resume it (-c).")

    print(f"Downloading dataset (~20.8 GiB) to {zip_path} using aria2c (16 connections)...")
    cmd = [
        "aria2c",
        "-x", "16",              # 16 connections per server
        "-s", "16",              # split the file into 16 segments
        "-c",                    # continue/resume a partial download
        "-d", raw_dir,
        "-o", "virome_pdbs.zip",
        ZENODO_URL,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("ERROR: aria2c not found. Install it (e.g. `sudo apt-get install -y aria2`) "
                 "or add it to PATH.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"aria2c failed (exit {e.returncode}). Re-run to resume from where it stopped.")

    if not os.path.exists(zip_path):
        sys.exit("ERROR: download did not produce the zip file.")
    try:
        with zipfile.ZipFile(zip_path, 'r'):
            print("Download complete and zip is valid.")
    except zipfile.BadZipFile:
        sys.exit("ERROR: downloaded file is corrupt; re-run to resume/repair.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deltafold', action='store_true',
                    help="Use the box data root (/data/pnardi) instead of ./data.")
    ap.add_argument('--data-dir', default=None,
                    help="Explicit data root (overrides --deltafold / DELTAFOLD_DATA_DIR).")
    args = ap.parse_args()

    # --data-dir wins; otherwise deltafold_paths already resolved the root from
    # argv (--deltafold) / env at import time.
    data_dir = args.data_dir or deltafold_paths.DATA_DIR
    raw_dir = os.path.join(data_dir, 'hoan_raw_pdb')
    # Ensure the processed dir exists too, so later stages have somewhere to write.
    os.makedirs(os.path.join(data_dir, 'hoan_processed'), exist_ok=True)
    print(f"Data root: {data_dir}")
    download_pdbs_fast(raw_dir)


if __name__ == "__main__":
    main()
