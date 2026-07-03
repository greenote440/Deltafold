"""Single source of truth for the DeltaFold data root.

The dataset does not live in the same place on every machine:

  * local dev (Mac / CPU)  ->  ``./data``  (repo-relative, unchanged)
  * the deltafold box      ->  ``/data/pnardi``  (the big /data volume, NOT the
                                repo checkout — 11 TB, keeps the 20 GB dataset and
                                the lifted .pt files off the home partition)

Select the box layout by any of:
  * passing ``--deltafold`` on the command line (same flag that switches the
    device to CUDA), or
  * exporting ``DELTAFOLD_DEVICE=cuda``, or
  * exporting ``DELTAFOLD_DATA_DIR=/some/path`` to point at an explicit root
    (this always wins, and works even without --deltafold / CUDA).

Resolution happens from argv/env at import time so the module-level path
constants below are fixed before anything binds them. Every pipeline stage
(download -> lift -> split -> train -> extract) imports these, so they always
agree on where the data is.
"""
import os
import sys

DATA_DIR_ENV = 'DELTAFOLD_DATA_DIR'
BOX_DATA_DIR = '/data/pnardi'
LOCAL_DATA_DIR = './data'


def deltafold_requested():
    """Whether this run targets the deltafold box (drives BOTH the CUDA device
    switch in train.py and the data-root switch here). Intentionally does NOT
    include DELTAFOLD_DATA_DIR — relocating data must not silently force CUDA."""
    return ('--deltafold' in sys.argv) or (os.environ.get('DELTAFOLD_DEVICE') == 'cuda')


def resolve_data_dir():
    """Explicit DELTAFOLD_DATA_DIR wins; else /data/pnardi under --deltafold; else
    the repo-local ./data."""
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return env
    return BOX_DATA_DIR if deltafold_requested() else LOCAL_DATA_DIR


DATA_DIR = resolve_data_dir()
PROC_DIR = os.path.join(DATA_DIR, 'hoan_processed')      # lifted PCC .pt files
RAW_DIR = os.path.join(DATA_DIR, 'hoan_raw_pdb')         # downloaded raw structures
RAW_ZIP = os.path.join(RAW_DIR, 'virome_pdbs.zip')       # the Zenodo archive
CLUSTER_TSV = os.path.join(DATA_DIR, 'cluster.tsv')      # Foldseek clusters (eval labels)
SUBBASE_PREFIX = os.path.join(DATA_DIR, 'subbase_corrected')  # split manifests prefix
EMB_FILE = os.path.join(DATA_DIR, 'virome_embeddings.pt')     # extracted embeddings
