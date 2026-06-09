"""Analyze model clusters that bridge >=2 Foldseek clusters.
Keys throughout are accessions (e.g. YP_009552282).
"""
import os, re, glob, collections
import numpy as np
import torch
import torch.nn.functional as F

MODEL_TSV  = './clusters/model_clusters.tsv'
FS_TSV     = './data/cluster.tsv'
EMBED_PT   = './data/virome_embeddings.pt'
TM_CACHE   = './checkpoints/tm_score_cache.pt'
PROC_DIR   = './data/hoan_processed'
TOP_N      = 25

def acc(name):
    name = re.sub(r'\.(pdb|pt)$', '', name)
    m = re.search(r'([A-Z]{1,2}_[0-9]{5,10})', name)
    return m.group(1) if m else (name.split('__')[1] if '__' in name else name)

# ── load model clusters (keyed by accession) ──────────────────────────────────
model_of = {}  # acc -> cluster_id (int)
raw_name_of = {}  # acc -> full protein name (for embedding lookup)
with open(MODEL_TSV) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2 and parts[1] != 'cluster_id':
            a = acc(parts[0])
            model_of[a] = int(parts[1])
            raw_name_of[a] = parts[0]

clusters = collections.defaultdict(list)  # cluster_id -> [accessions]
for a, c in model_of.items(): clusters[c].append(a)

# ── load foldseek clusters ────────────────────────────────────────────────────
fs_of = {}  # acc -> foldseek_rep_acc
with open(FS_TSV) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            fs_of[acc(parts[1])] = acc(parts[0])

# ── load family from processed filenames ─────────────────────────────────────
fam_of = {}  # acc -> family string
for path in glob.glob(os.path.join(PROC_DIR, '*.pt')):
    base = os.path.basename(path).replace('.pt', '')
    parts = base.split('__')
    a = parts[1] if len(parts) >= 2 else acc(base)
    fam_of[a] = parts[2] if len(parts) >= 4 else 'unknown'

# ── load embeddings ───────────────────────────────────────────────────────────
emb_data = torch.load(EMBED_PT, weights_only=False)
# virome_embeddings.pt is {full_protein_name: tensor(128)}
id_list = list(emb_data.keys())
emb_mat = F.normalize(torch.stack([torch.as_tensor(emb_data[k]) for k in id_list]).float(), dim=-1)
# index by accession
emb_idx = {acc(k): i for i, k in enumerate(id_list)}

def get_idxs(accs):
    return [emb_idx[a] for a in accs if a in emb_idx]

def avg_cosine_dist(accs):
    idxs = get_idxs(accs)
    if len(idxs) < 2: return float('nan')
    sub = emb_mat[idxs]
    sim = (sub @ sub.T).numpy()
    off = sim[~np.eye(len(idxs), dtype=bool)]
    return float(np.mean(1 - off))

# ── load TM cache ─────────────────────────────────────────────────────────────
tm_cache = torch.load(TM_CACHE, weights_only=False) if os.path.exists(TM_CACHE) else {}
# tm_cache keys appear to be (basename.pt, basename.pt)
def get_tm(a, b):
    ka, kb = a + '.pt', b + '.pt'
    return tm_cache.get((ka, kb)) or tm_cache.get((kb, ka))

# ── find bridge clusters ──────────────────────────────────────────────────────
bridge_clusters = {}
for c, prots in clusters.items():
    fs_reps = set(fs_of.get(p, '?') for p in prots if p in fs_of)
    if len(fs_reps) >= 2:
        fams = set(fam_of.get(p, 'unknown') for p in prots)
        bridge_clusters[c] = {'prots': prots, 'fs': fs_reps, 'fams': fams}

print(f"Total model clusters     : {len(clusters)}")
print(f"Bridge clusters (>=2 FS) : {len(bridge_clusters)}")
print(f"Proteins in bridges      : {sum(len(v['prots']) for v in bridge_clusters.values())}")
print()

# ── family-pair frequency ─────────────────────────────────────────────────────
pair_counts = collections.Counter()
for v in bridge_clusters.values():
    fams = sorted(v['fams'])
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            pair_counts[(fams[i], fams[j])] += 1

print("=== Top 20 bridged family pairs ===")
for (a, b), n in pair_counts.most_common(20):
    print(f"  {n:4d}  {a} <-> {b}")
print()

# ── top bridges by size ───────────────────────────────────────────────────────
top = sorted(bridge_clusters.items(), key=lambda x: len(x[1]['prots']), reverse=True)[:TOP_N]

print(f"=== Top {TOP_N} bridge clusters by size ===")
print(f"{'cid':>7} {'sz':>4} {'n_fs':>4} {'n_fam':>5}  avg_d   tight   families")
tight_total = 0
for cid, v in top:
    prots = v['prots']
    d = avg_cosine_dist(prots)
    tight = d < 0.20
    # TM stats for cross-FS pairs only
    tm_vals = []
    fs_of_p = {p: fs_of.get(p) for p in prots}
    for i in range(len(prots)):
        for j in range(i+1, len(prots)):
            if fs_of_p[prots[i]] != fs_of_p[prots[j]]:
                tm = get_tm(prots[i], prots[j])
                if tm is not None: tm_vals.append(float(tm))
    tm_str = f"TM={np.mean(tm_vals):.2f}(n={len(tm_vals)})" if tm_vals else "TM=n/a"
    fam_str = ', '.join(sorted(v['fams']))[:70]
    print(f"  {cid:>7} {len(prots):>4} {len(v['fs']):>4} {len(v['fams']):>5}  {d:.3f}  {'YES' if tight else ' no'}  {tm_str}  [{fam_str}]")

# ── size distribution ─────────────────────────────────────────────────────────
print()
sizes = sorted(len(v['prots']) for v in bridge_clusters.values())
print("=== Bridge cluster size distribution ===")
for lo, hi in [(2,2),(3,5),(6,10),(11,25),(26,50),(51,999)]:
    n = sum(1 for s in sizes if lo <= s <= hi)
    print(f"  size {lo:3d}-{hi:<3d} : {n:5d} clusters")
print(f"  median={np.median(sizes):.0f}  mean={np.mean(sizes):.1f}  max={max(sizes)}")

# ── tightness summary ─────────────────────────────────────────────────────────
tight_n = sum(1 for v in bridge_clusters.values() if avg_cosine_dist(v['prots']) < 0.20)
loose_n = sum(1 for v in bridge_clusters.values() if avg_cosine_dist(v['prots']) >= 0.45)
print(f"\nTight bridges (avg_d < 0.20) : {tight_n}/{len(bridge_clusters)}  ({100*tight_n/len(bridge_clusters):.1f}%)")
print(f"Loose bridges (avg_d >= 0.45) : {loose_n}/{len(bridge_clusters)}  ({100*loose_n/len(bridge_clusters):.1f}%)")
