"""
Shared helpers for the Phase 3-5 downstream analysis pipeline (action_plan_phases_3_5.md).

Loads the learned embeddings, parses protein ids, and attaches the biological
metadata that every Phase 3-5 script needs:
  - viral family  -> from the directory component of data/subdataset_files.txt
                     (viral_structures/<FAMILY>/<protein>.pdb)
  - genome type   -> from a family -> Baltimore-group map (FAMILY_TO_GENOME below)
  - Foldseek cluster -> representative id from data/cluster.tsv (col0=rep, col1=member)
  - taxid/accession  -> parsed from the protein id itself

All artifacts use the same protein id with different extensions; everything here
is keyed by the canonical id (extension stripped):
    <name>__<accession>__<species>__<taxid>
"""
import os
import re
import numpy as np
import torch

EMB_FILE = './data/virome_embeddings.pt'
CLUSTER_TSV = './data/cluster.tsv'
SUBDATASET_FILES = './data/subdataset_files.txt'
TM_CACHE = './checkpoints/tm_score_cache.pt'
OUT_DIR = './clusters'

# The cosine-distance band that the TM-score analysis calls "structurally similar"
# (TM > ~0.4-0.7 -> mean cosine distance ~0.45-0.50). Used as the close/far threshold.
CLOSE_THRESHOLD = 0.45

# Baltimore genome group for every viral family in the catalogue (89 families).
# RT viruses are kept distinct (Baltimore VI/VII) rather than folded into ssRNA/dsDNA.
FAMILY_TO_GENOME = {
    # Group I - dsDNA
    'Adenoviridae': 'dsDNA', 'Adomaviridae': 'dsDNA', 'Alloherpesviridae': 'dsDNA',
    'Ascoviridae': 'dsDNA', 'Asfarviridae': 'dsDNA', 'Baculoviridae': 'dsDNA',
    'Herpesviridae': 'dsDNA', 'Hytrosaviridae': 'dsDNA', 'Iridoviridae': 'dsDNA',
    'Lavidaviridae': 'dsDNA', 'Marseilleviridae': 'dsDNA', 'Mimiviridae': 'dsDNA',
    'Nimaviridae': 'dsDNA', 'Nudiviridae': 'dsDNA', 'Papillomaviridae': 'dsDNA',
    'Phycodnaviridae': 'dsDNA', 'Pithoviridae': 'dsDNA', 'Polyomaviridae': 'dsDNA',
    'Polydnaviriformidae': 'dsDNA', 'Poxviridae': 'dsDNA',
    # Group II - ssDNA
    'Alphasatellitidae': 'ssDNA', 'Anelloviridae': 'ssDNA', 'Bacilladnaviridae': 'ssDNA',
    'Circoviridae': 'ssDNA', 'Geminiviridae': 'ssDNA', 'Genomoviridae': 'ssDNA',
    'Nanoviridae': 'ssDNA', 'Parvoviridae': 'ssDNA', 'Smacoviridae': 'ssDNA',
    'Tolecusatellitidae': 'ssDNA',
    # Group III - dsRNA
    'Amalgaviridae': 'dsRNA', 'Birnaviridae': 'dsRNA', 'Chrysoviridae': 'dsRNA',
    'Curvulaviridae': 'dsRNA', 'Partitiviridae': 'dsRNA', 'Picobirnaviridae': 'dsRNA',
    'Polymycoviridae': 'dsRNA', 'Reoviridae': 'dsRNA', 'Totiviridae': 'dsRNA',
    # Group IV - ssRNA(+)
    'Alphaflexiviridae': 'ssRNA(+)', 'Arteriviridae': 'ssRNA(+)', 'Astroviridae': 'ssRNA(+)',
    'Benyviridae': 'ssRNA(+)', 'Betaflexiviridae': 'ssRNA(+)', 'Bromoviridae': 'ssRNA(+)',
    'Caliciviridae': 'ssRNA(+)', 'Closteroviridae': 'ssRNA(+)', 'Coronaviridae': 'ssRNA(+)',
    'Deltaflexiviridae': 'ssRNA(+)', 'Dicistroviridae': 'ssRNA(+)', 'Flaviviridae': 'ssRNA(+)',
    'Hepeviridae': 'ssRNA(+)', 'Iflaviridae': 'ssRNA(+)', 'Kitaviridae': 'ssRNA(+)',
    'Luteoviridae': 'ssRNA(+)', 'Mesoniviridae': 'ssRNA(+)', 'Mitoviridae': 'ssRNA(+)',
    'Nodaviridae': 'ssRNA(+)', 'Picornaviridae': 'ssRNA(+)', 'Polycipiviridae': 'ssRNA(+)',
    'Potyviridae': 'ssRNA(+)', 'Secoviridae': 'ssRNA(+)', 'Sinhaliviridae': 'ssRNA(+)',
    'Solemoviridae': 'ssRNA(+)', 'Tobaniviridae': 'ssRNA(+)', 'Togaviridae': 'ssRNA(+)',
    'Tombusviridae': 'ssRNA(+)', 'Tymoviridae': 'ssRNA(+)', 'Virgaviridae': 'ssRNA(+)',
    # Group V - ssRNA(-)
    'Arenaviridae': 'ssRNA(-)', 'Bornaviridae': 'ssRNA(-)', 'Chuviridae': 'ssRNA(-)',
    'Filoviridae': 'ssRNA(-)', 'Fimoviridae': 'ssRNA(-)', 'Hantaviridae': 'ssRNA(-)',
    'Nairoviridae': 'ssRNA(-)', 'Nyamiviridae': 'ssRNA(-)', 'Orthomyxoviridae': 'ssRNA(-)',
    'Paramyxoviridae': 'ssRNA(-)', 'Peribunyaviridae': 'ssRNA(-)', 'Phasmaviridae': 'ssRNA(-)',
    'Phenuiviridae': 'ssRNA(-)', 'Pneumoviridae': 'ssRNA(-)', 'Rhabdoviridae': 'ssRNA(-)',
    'Tospoviridae': 'ssRNA(-)', 'Xinmoviridae': 'ssRNA(-)',
    # Group VI/VII - reverse transcribing
    'Retroviridae': 'ssRNA-RT', 'Hepadnaviridae': 'dsDNA-RT', 'Caulimoviridae': 'dsDNA-RT',
}

# Stable colour order for genome types (used by the visualisation).
GENOME_ORDER = ['dsDNA', 'ssDNA', 'dsRNA', 'ssRNA(+)', 'ssRNA(-)',
                'ssRNA-RT', 'dsDNA-RT', 'unknown']


def load(path):
    """torch.load that works across the weights_only default change in torch 2.6."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def strip_ext(name):
    return re.sub(r'\.(pdb|pt)$', '', os.path.basename(name))


def parse_protein_id(pid):
    """<name>__<accession>__<species>__<taxid> -> (name, accession, species, taxid).

    Parsed from the right so a name containing '__' is handled correctly.
    """
    parts = pid.split('__')
    if len(parts) < 4:
        return pid, '', '', ''
    taxid = parts[-1]
    species = parts[-2]
    accession = parts[-3]
    name = '__'.join(parts[:-3])
    return name, accession, species, taxid


def load_embeddings(path=EMB_FILE, normalize=True):
    """Return (ids, X) with ids sorted for determinism and X an (N,128) float64 array.

    Embeddings are already L2-normalised on disk; we renormalise defensively so that
    cosine distance == 0.5 * squared-euclidean and dot products are true cosines.
    """
    emb = load(path)
    ids = sorted(emb.keys())
    X = np.stack([np.asarray(emb[i], dtype=np.float64) for i in ids])
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return ids, X


def load_family_map(path=SUBDATASET_FILES):
    """canonical protein id -> viral family (from the directory component of the path)."""
    fam = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) < 2:
                continue
            fam[strip_ext(parts[-1])] = parts[-2]
    return fam


def genome_type(family):
    return FAMILY_TO_GENOME.get(family, 'unknown')


def load_foldseek_clusters(path=CLUSTER_TSV):
    """member id -> representative id (the Foldseek cluster id), extensions stripped."""
    rep_of = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            rep, member = strip_ext(cols[0]), strip_ext(cols[1])
            rep_of[member] = rep
    return rep_of


def load_tm_cache(path=TM_CACHE):
    """frozenset({id_a, id_b}) -> TM-score, with ids canonicalised (extension stripped)."""
    raw = load(path)
    cache = {}
    for (a, b), tm in raw.items():
        cache[frozenset((strip_ext(a), strip_ext(b)))] = float(tm)
    return cache


def tm_lookup(cache, a, b):
    return cache.get(frozenset((a, b)))


def build_metadata(ids):
    """Per-protein metadata dict keyed by id: name, accession, species, taxid, family, genome_type."""
    fam = load_family_map()
    meta = {}
    for pid in ids:
        name, acc, species, taxid = parse_protein_id(pid)
        family = fam.get(pid, 'unknown')
        meta[pid] = {
            'name': name, 'accession': acc, 'species': species, 'taxid': taxid,
            'family': family, 'genome_type': genome_type(family),
        }
    return meta


def ensure_out_dir(path=OUT_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def load_cluster_tsv(path):
    """Read a `(protein_id, cluster_id)` TSV (written by cluster_embeddings.py) into
    {protein_id: int cluster_id}. Tolerates a header row."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 2 or cols[1].strip().lower() in ('cluster_id', 'cluster'):
                continue
            out[cols[0]] = int(cols[1])
    return out


def cosine_distance_matrix(X):
    """Dense (N,N) cosine-distance matrix for L2-normalised rows, clipped to [0, 2]."""
    D = 1.0 - (X @ X.T)
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D
