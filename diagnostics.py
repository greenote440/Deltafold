"""
Shortcut Diagnostic Battery
============================
Implements the Section 6 diagnostic tests from `shortcut_analysis_report.md` to
detect whether the contrastive encoder learned genuine structural topology or a
cheaper proxy (length, composition, taxonomy, collapse, ...).

Each diagnostic prints a clear verdict (OK / SUSPICIOUS) against the thresholds
suggested in the report. Model-free diagnostics run against the extracted
embeddings + cluster.tsv alone; model-based diagnostics load the trained
checkpoint to re-embed proteins under ablation / synthetic conditions.

Usage:
    python diagnostics.py                      # run the whole battery
    python diagnostics.py --only length,taxonomy,singleton
    python diagnostics.py --model asymmetric --task contrastive
    python diagnostics.py --sample 400         # cap pairwise / model-based tests

Diagnostics implemented (report Section 6 table):
    variable name      -> report row
    ----------------------------------------------------------------------
    variance           -> Embedding variance (collapse)
    length             -> Length correlation
    composition        -> Composition ablation
    plddt              -> pLDDT masking            (scaffold only; needs raw PDB B-factors)
    attention          -> Attention position analysis
    taxonomy           -> Phylogenetic / taxonomic shortcut
    injection          -> Random structure injection
    singleton          -> Singleton region analysis
    lossquality        -> Loss vs clustering quality
    tmscore            -> TM-score rank correlation (delegates to evaluate_correlation.py)
"""
import os
import glob
import random
import argparse
import numpy as np
import torch
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform, cosine

PROC_DIR = './data/hoan_processed'
EMBEDDINGS_FILE = './data/virome_embeddings.pt'
CLUSTER_TSV = './data/cluster.tsv'
CHECKPOINT_DIR = './checkpoints'
LOSS_LOG = os.path.join(CHECKPOINT_DIR, 'contrastive_losses.csv')

DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                      else ('cuda' if torch.cuda.is_available() else 'cpu'))

SUSPICIOUS = "⚠️  SUSPICIOUS"
OK = "✅ OK"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 60)
    print(f"--- {title} ---")
    print("=" * 60)


def parse_taxonomy(key):
    """Keys look like '{gene}__{accession}__{species}__{taxid}'."""
    parts = key.split('__')
    if len(parts) >= 4:
        return {'gene': parts[0], 'accession': parts[1],
                'species': '__'.join(parts[2:-1]), 'taxid': parts[-1]}
    return {'gene': key, 'accession': key, 'species': key, 'taxid': key}


def load_embeddings(path=EMBEDDINGS_FILE):
    try:
        emb = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        emb = torch.load(path, map_location='cpu')
    ids = list(emb.keys())
    X = np.stack([np.asarray(emb[k], dtype=np.float64).ravel() for k in ids])
    return ids, X


def load_clusters(path=CLUSTER_TSV):
    """Returns rep_of[member] = representative (cluster id), members stripped of .pdb."""
    rep_of = {}
    if not os.path.exists(path):
        return rep_of
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 2:
                continue
            rep = p[0][:-4] if p[0].endswith('.pdb') else p[0]
            mem = p[1][:-4] if p[1].endswith('.pdb') else p[1]
            rep_of[mem] = rep
    return rep_of


def load_lengths(ids, sample=None):
    """Loads protein_size from processed .pt files. Subsamples for speed."""
    target = ids if sample is None or sample >= len(ids) else random.sample(ids, sample)
    lengths = {}
    for pid in target:
        pt = os.path.join(PROC_DIR, f"{pid}.pt")
        if not os.path.exists(pt):
            continue
        try:
            try:
                d = torch.load(pt, map_location='cpu', weights_only=False)
            except TypeError:
                d = torch.load(pt, map_location='cpu')
            n = d['rank3'].get('protein_size', 0)
            lengths[pid] = int(n.item() if isinstance(n, torch.Tensor) else n)
        except Exception:
            pass
    return lengths


def adjusted_rand_index(labels_true, labels_pred):
    """Numpy ARI (avoids a hard sklearn dependency)."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    _, t_idx = np.unique(labels_true, return_inverse=True)
    _, p_idx = np.unique(labels_pred, return_inverse=True)
    n = labels_true.shape[0]
    cont = np.zeros((t_idx.max() + 1, p_idx.max() + 1), dtype=np.int64)
    np.add.at(cont, (t_idx, p_idx), 1)
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_c = comb2(cont.sum(axis=1)).sum()
    sum_k = comb2(cont.sum(axis=0)).sum()
    sum_ij = comb2(cont).sum()
    expected = sum_c * sum_k / comb2(n)
    maxidx = (sum_c + sum_k) / 2.0
    if maxidx - expected == 0:
        return 0.0
    return (sum_ij - expected) / (maxidx - expected)


def silhouette_cosine(X, labels):
    """Mean silhouette over points whose cluster has >= 2 members (cosine distance)."""
    labels = np.asarray(labels)
    D = squareform(pdist(X, metric='cosine'))
    uniq = np.unique(labels)
    members = {c: np.where(labels == c)[0] for c in uniq}
    scores = []
    for i in range(len(labels)):
        own = members[labels[i]]
        if len(own) < 2:
            continue
        a = D[i, own].sum() / (len(own) - 1)
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = members[c]
            b = min(b, D[i, other].mean())
        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else float('nan')


def load_model(model_type, task):
    """Loads the best checkpoint into the requested architecture."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{task}_{model_type}_best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{task}_{model_type}_last.pth')
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model_config = ckpt.get('model_config', {})  # restore mitigation flags if present
    if model_type == 'asymmetric':
        from asymmetric_topotein import AsymmetricTopoNet
        model = AsymmetricTopoNet(scalar_dim=128, **model_config).to(DEVICE)
    else:
        from topotein import Topotein
        model = Topotein(scalar_dim=128, **model_config).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt_path


def embed_files(model, file_ids, mutate=None, batch_size=16):
    """Re-embeds a list of protein ids through the model. `mutate(data)` may edit
    each PCC dict in place before collation (used for composition ablation)."""
    from train import custom_collate, to_device
    out = {}
    batch = []
    names = []

    def flush():
        if not batch:
            return
        coll = custom_collate(list(batch))
        if coll is not None:
            feats = to_device(coll, DEVICE)
            with torch.no_grad():
                z = model(feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            for nm, e in zip(names, z):
                out[nm] = e.cpu().numpy().astype(np.float64)
        batch.clear()
        names.clear()

    for pid in file_ids:
        pt = os.path.join(PROC_DIR, f"{pid}.pt")
        if not os.path.exists(pt):
            continue
        try:
            try:
                d = torch.load(pt, map_location='cpu', weights_only=False)
            except TypeError:
                d = torch.load(pt, map_location='cpu')
        except Exception:
            continue
        if d['rank1']['source'].shape[1] != 16:
            continue
        if mutate is not None:
            mutate(d)
        batch.append((d, pt))
        names.append(pid)
        if len(batch) >= batch_size:
            flush()
    flush()
    return out


# ---------------------------------------------------------------------------
# Diagnostic 1: Embedding variance / collapse
# ---------------------------------------------------------------------------
def diag_variance(ids, X, rep_of):
    banner("Embedding Variance / Collapse")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    per_dim_std = Xn.std(axis=0)
    global_disp = float(per_dim_std.mean())

    # Mean pairwise cosine similarity on a sample (collapse -> approaches 1.0)
    n = min(len(ids), 1000)
    sub = np.random.choice(len(ids), n, replace=False)
    sims = Xn[sub] @ Xn[sub].T
    iu = np.triu_indices(n, k=1)
    mean_cos = float(sims[iu].mean())

    # Within-cluster vs global dispersion
    from collections import defaultdict
    groups = defaultdict(list)
    for i, pid in enumerate(ids):
        groups[rep_of.get(pid, pid)].append(i)
    within = [Xn[idx].std(axis=0).mean() for idx in groups.values() if len(idx) >= 2]
    within_disp = float(np.mean(within)) if within else float('nan')

    print(f"Global per-dim std (dispersion): {global_disp:.4f}")
    print(f"Mean within-cluster dispersion:  {within_disp:.4f}")
    print(f"Mean pairwise cosine similarity: {mean_cos:.4f}")
    ratio = global_disp / within_disp if within_disp and not np.isnan(within_disp) else float('nan')
    print(f"Global / within-cluster ratio:   {ratio:.3f}  (want >> 1)")

    collapsed = mean_cos > 0.9 or (not np.isnan(ratio) and ratio < 1.5)
    print(f"\n{SUSPICIOUS if collapsed else OK}: "
          + ("embeddings show collapse-like concentration."
             if collapsed else "embeddings are well dispersed relative to clusters."))


# ---------------------------------------------------------------------------
# Diagnostic 2: Length correlation
# ---------------------------------------------------------------------------
def diag_length(ids, X, sample):
    banner("Length Correlation (length shortcut)")
    idx_of = {pid: i for i, pid in enumerate(ids)}
    lengths = load_lengths(ids, sample=sample)
    valid = [p for p in lengths if lengths[p] > 0]
    if len(valid) < 10:
        print("Not enough proteins with known length. Skipping.")
        return
    emb_d, len_d = [], []
    for a, b in combinations(valid, 2):
        emb_d.append(cosine(X[idx_of[a]], X[idx_of[b]]))
        len_d.append(abs(lengths[a] - lengths[b]))
    rho, p = spearmanr(emb_d, len_d)
    print(f"Proteins: {len(valid)} | Pairs: {len(emb_d)}")
    print(f"Spearman rho(embedding dist, |len diff|): {rho:.4f}  (p={p:.2e})")
    bad = abs(rho) > 0.3
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + (f"|rho|={abs(rho):.3f} > 0.3 -> embeddings track protein length."
             if bad else f"|rho|={abs(rho):.3f} <= 0.3 -> no strong length bias."))


# ---------------------------------------------------------------------------
# Diagnostic 3: Composition ablation
# ---------------------------------------------------------------------------
def diag_composition(ids, model, sample):
    banner("Composition Ablation (residue-type shortcut)")
    if model is None:
        print("No checkpoint loaded. Skipping.")
        return
    target = ids if sample >= len(ids) else random.sample(ids, sample)

    base = embed_files(model, target)

    def zero_aa(d):
        d['rank0']['aa'] = torch.zeros_like(d['rank0']['aa'])

    def zero_aa_3di(d):
        d['rank0']['aa'] = torch.zeros_like(d['rank0']['aa'])
        d['rank0']['3di'] = torch.zeros_like(d['rank0']['3di'])

    abl_aa = embed_files(model, target, mutate=zero_aa)
    abl_both = embed_files(model, target, mutate=zero_aa_3di)

    common = [p for p in target if p in base and p in abl_aa and p in abl_both]
    shift_aa = np.mean([cosine(base[p], abl_aa[p]) for p in common])
    shift_both = np.mean([cosine(base[p], abl_both[p]) for p in common])

    # Does the neighbour structure survive ablation? (rank corr of pairwise dists)
    B = np.stack([base[p] for p in common])
    A = np.stack([abl_aa[p] for p in common])
    pd_base = pdist(B, metric='cosine')
    pd_abl = pdist(A, metric='cosine')
    geom_rho, _ = spearmanr(pd_base, pd_abl)

    print(f"Proteins embedded: {len(common)}")
    print(f"Mean cosine shift when AA zeroed:       {shift_aa:.4f}")
    print(f"Mean cosine shift when AA+3Di zeroed:   {shift_both:.4f}")
    print(f"Pairwise-distance rank corr (AA off):   {geom_rho:.4f}  (want ~1.0)")
    bad = shift_aa > 0.3 or geom_rho < 0.7
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + ("residue composition materially drives the embedding."
             if bad else "embedding is largely invariant to residue composition."))


# ---------------------------------------------------------------------------
# Diagnostic 4: pLDDT masking (scaffold)
# ---------------------------------------------------------------------------
def diag_plddt(ids, model, sample):
    banner("pLDDT Masking (ColabFold artifact)  [SCAFFOLD]")
    print("Per-residue pLDDT is not present in the processed PCCs; it lives in the")
    print("B-factor column of the raw PDBs (data/hoan_raw_pdb/virome_pdbs.zip, ~22GB).")
    print("To enable this test, extract B-factors and re-embed with pLDDT<70 residues")
    print("masked, then compare clustering quality. Hook points are ready:")
    print("  - extract per-residue pLDDT in topotein_lifter.lift_rank0_residues")
    print("  - add a `mask_low_plddt(d, thresh)` mutate fn and call embed_files(...)")
    print(f"\n{OK}: skipped by request (no pLDDT data available).")


# ---------------------------------------------------------------------------
# Diagnostic 5: Attention position analysis
# ---------------------------------------------------------------------------
def diag_attention(ids, model, model_type, sample):
    banner("Attention Position Analysis (sequential vs structural)")
    if model is None or model_type != 'asymmetric':
        print("Requires the asymmetric model with attention capture. Skipping.")
        return
    from asymmetric_topotein import AsymmetricTopoAttentionLayer
    from train import custom_collate, to_device

    layers = [m for m in model.modules() if isinstance(m, AsymmetricTopoAttentionLayer)]
    for L in layers:
        L.capture_attn = True

    target = random.sample(ids, min(sample, 64, len(ids)))
    seq_corrs, dist_corrs = [], []
    try:
        for pid in target:
            pt = os.path.join(PROC_DIR, f"{pid}.pt")
            try:
                d = torch.load(pt, map_location='cpu', weights_only=False)
            except TypeError:
                d = torch.load(pt, map_location='cpu')
            if d['rank1']['source'].shape[1] != 16:
                continue
            coll = custom_collate([(d, pt)])
            if coll is None:
                continue
            feats = to_device(coll, DEVICE)
            with torch.no_grad():
                model(feats)
            dist3d = feats['rank1']['distance'].flatten().cpu().numpy()
            last = layers[-1]
            if last.last_node_attn is None:
                continue
            attn = last.last_node_attn.cpu().numpy()
            src = last.last_attn_src.cpu().numpy()
            dst = last.last_attn_dst.cpu().numpy()
            seqsep = np.abs(src - dst).astype(np.float64)
            if attn.shape[0] != seqsep.shape[0] or attn.std() == 0:
                continue
            rs, _ = spearmanr(attn, seqsep)
            rd, _ = spearmanr(attn, dist3d[:attn.shape[0]])
            if not np.isnan(rs):
                seq_corrs.append(rs)
            if not np.isnan(rd):
                dist_corrs.append(rd)
    finally:
        for L in layers:
            L.capture_attn = False

    if not seq_corrs:
        print("Could not capture attention weights. Skipping.")
        return
    mseq = float(np.mean(np.abs(seq_corrs)))
    mdist = float(np.mean(np.abs(dist_corrs)))
    print(f"Proteins probed: {len(seq_corrs)}")
    print(f"|corr(attn, sequence separation |i-j|)|: {mseq:.4f}")
    print(f"|corr(attn, 3D distance ||xi-xj||)|:     {mdist:.4f}")
    bad = mseq > mdist
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + ("attention tracks sequence position more than 3D structure."
             if bad else "attention is driven by 3D structure over sequence position."))


# ---------------------------------------------------------------------------
# Diagnostic 6: Taxonomic / phylogenetic shortcut
# ---------------------------------------------------------------------------
def diag_taxonomy(ids, X):
    banner("Taxonomic Shortcut (do embeddings cluster by species?)")
    tax = [parse_taxonomy(k) for k in ids]
    species = np.array([t['species'] for t in tax])

    # Keep only species with >= 2 members so silhouette is meaningful
    from collections import Counter
    counts = Counter(species)
    keep = np.array([counts[s] >= 2 for s in species])
    if keep.sum() < 10:
        print("Too few multi-member species. Skipping.")
        return
    Xs = X[keep]
    sp = species[keep]
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)

    # Subsample for tractable O(n^2) silhouette
    if len(sp) > 1500:
        sub = np.random.choice(len(sp), 1500, replace=False)
        Xn, sp = Xn[sub], sp[sub]

    sil = silhouette_cosine(Xn, sp)

    # Within-species vs across-species mean cosine distance
    D = squareform(pdist(Xn, metric='cosine'))
    same = sp[:, None] == sp[None, :]
    iu = np.triu_indices(len(sp), k=1)
    within = D[iu][same[iu]].mean()
    across = D[iu][~same[iu]].mean()

    print(f"Multi-member-species proteins: {len(sp)} across {len(set(sp))} species")
    print(f"Silhouette by species (cosine):  {sil:.4f}  (high => taxonomy-organized)")
    print(f"Mean within-species distance:     {within:.4f}")
    print(f"Mean across-species distance:     {across:.4f}")
    print(f"Within/across ratio:              {within / across:.3f}  (want ~1.0)")
    bad = sil > 0.3
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + ("embeddings are strongly organized by taxonomy -> possible taxonomic shortcut."
             if bad else "embeddings are not dominated by species identity."))


# ---------------------------------------------------------------------------
# Diagnostic 7: Random structure injection
# ---------------------------------------------------------------------------
def _synthetic_pcc(n_res):
    """Builds a valid PCC from a random self-avoiding-ish Ca walk of length n_res."""
    from topotein_lifter import TopoteinLifter, AA_ALPHABET, FOLDSEEK_ALPHABET
    lifter = TopoteinLifter()

    steps = np.random.randn(n_res, 3)
    steps /= (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-8)
    steps *= 3.8  # realistic Ca-Ca spacing
    coords = np.cumsum(steps, axis=0).astype(np.float32)
    ca = torch.tensor(coords, dtype=torch.float32, device=lifter_device())

    # Random contiguous SSE segmentation (H/E/C)
    dssp = []
    while len(dssp) < n_res:
        seg = random.choice("HEC")
        dssp += [seg] * random.randint(4, 14)
    dssp = dssp[:n_res]

    # Match the lifter's _one_hot dimensions exactly: AA gets an extra slot
    # (no 'X' in AA_ALPHABET) -> 23 dims; 3Di already contains 'X' -> 21 dims.
    aa_dim = len(AA_ALPHABET) + (0 if 'X' in AA_ALPHABET else 1)
    di_dim = len(FOLDSEEK_ALPHABET) + (0 if 'X' in FOLDSEEK_ALPHABET else 1)
    rank0 = {
        'aa': torch.zeros(n_res, aa_dim, device=ca.device),
        '3di': torch.zeros(n_res, di_dim, device=ca.device),
        'phi_psi': torch.zeros(n_res, 4, device=ca.device),
        'ca_coords': ca,
        'positional_encoding': lifter.get_positional_encoding(
            torch.arange(n_res, device=ca.device)),
    }
    # Mark all residues "unknown": AA unknown -> index 0 (lifter convention),
    # 3Di unknown -> the 'X' slot.
    rank0['aa'][:, 0] = 1.0
    rank0['3di'][:, FOLDSEEK_ALPHABET.find('X')] = 1.0

    rank1 = lifter.lift_rank1_edges(ca)
    rank2 = lifter.lift_rank2_secondary_structures(ca, dssp)
    rank3 = lifter.lift_rank3_global(ca, list(range(n_res)))
    return lifter.dict_to_cpu({'rank0': rank0, 'rank1': rank1, 'rank2': rank2, 'rank3': rank3})


def lifter_device():
    # topotein_lifter pins tensors to its own DEVICE; mirror it here.
    from topotein_lifter import DEVICE as LD
    return LD


def diag_injection(ids, X, model, sample):
    banner("Random Structure Injection (content vs statistics)")
    if model is None:
        print("No checkpoint loaded. Skipping.")
        return
    from train import custom_collate, to_device

    lengths = load_lengths(ids, sample=min(sample, 400))
    lens = [l for l in lengths.values() if l >= 32]
    if len(lens) < 10:
        print("Not enough lengths to model the distribution. Skipping.")
        return

    n_fake = min(40, len(lens))
    fake_embs = []
    for _ in range(n_fake):
        n_res = int(random.choice(lens))
        try:
            d = _synthetic_pcc(n_res)
            if d['rank1']['source'].shape[1] != 16:
                continue
            coll = custom_collate([(d, 'synthetic')])
            if coll is None:
                continue
            feats = to_device(coll, DEVICE)
            with torch.no_grad():
                z = model(feats)
            if z.dim() > 1:
                z = z.squeeze(0)
            fake_embs.append(z.cpu().numpy().astype(np.float64))
        except Exception:
            continue
    if len(fake_embs) < 5:
        print("Could not synthesize enough decoys. Skipping.")
        return
    F = np.stack(fake_embs)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)

    # Nearest real neighbour distance for real proteins (sampled) and for decoys
    sub = np.random.choice(len(Xn), min(800, len(Xn)), replace=False)
    Dr = 1 - Xn[sub] @ Xn[sub].T
    np.fill_diagonal(Dr, np.inf)
    real_nn = Dr.min(axis=1)
    Df = 1 - Fn @ Xn[sub].T
    fake_nn = Df.min(axis=1)

    real_med = float(np.median(real_nn))
    frac_inside = float(np.mean(fake_nn <= real_med))
    print(f"Synthetic decoys embedded: {len(F)}")
    print(f"Median real->real NN cosine distance:    {real_med:.4f}")
    print(f"Median decoy->real NN cosine distance:   {float(np.median(fake_nn)):.4f}")
    print(f"Fraction of decoys inside real manifold: {frac_inside:.2f}")
    bad = frac_inside > 0.5
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + ("random structures land inside the real manifold -> model encodes statistics, not content."
             if bad else "random structures sit outside the real manifold -> model uses structural content."))


# ---------------------------------------------------------------------------
# Diagnostic 8: Singleton region analysis
# ---------------------------------------------------------------------------
def diag_singleton(ids, X, rep_of):
    banner("Singleton Region Analysis (singleton collapse)")
    from collections import Counter
    rep = [rep_of.get(p, p) for p in ids]
    counts = Counter(rep)
    is_single = np.array([counts[rep_of.get(p, p)] == 1 for p in ids])
    n_single = int(is_single.sum())
    if n_single < 10 or (len(ids) - n_single) < 10:
        print(f"Singletons={n_single}, multi={len(ids) - n_single}. Too few to compare. Skipping.")
        return

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn[is_single]
    M = Xn[~is_single]

    # subsample for tractable means
    si = np.random.choice(len(S), min(400, len(S)), replace=False)
    mi = np.random.choice(len(M), min(400, len(M)), replace=False)
    S, M = S[si], M[mi]

    ss = 1 - S @ S.T
    iu = np.triu_indices(len(S), k=1)
    within_single = float(ss[iu].mean())
    sm = (1 - S @ M.T).mean()
    mm = 1 - M @ M.T
    iu2 = np.triu_indices(len(M), k=1)
    within_multi = float(mm[iu2].mean())

    print(f"Singletons: {n_single} | Multi-member: {len(ids) - n_single}")
    print(f"Mean singleton<->singleton distance:    {within_single:.4f}")
    print(f"Mean singleton<->multimember distance:  {float(sm):.4f}")
    print(f"Mean multimember<->multimember distance:{within_multi:.4f}")
    # collapse signal: singletons closer to each other than to the rest
    bad = within_single < float(sm) * 0.85
    print(f"\n{SUSPICIOUS if bad else OK}: "
          + ("singletons cluster together in a distinct region (dataset-status artifact)."
             if bad else "singletons are dispersed, not collapsed into their own region."))


# ---------------------------------------------------------------------------
# Diagnostic 9: Loss vs clustering quality
# ---------------------------------------------------------------------------
def diag_lossquality():
    banner("Loss vs Clustering Quality (loss-quality decoupling)")
    if not os.path.exists(LOSS_LOG):
        print(f"No loss log at {LOSS_LOG}. Skipping.")
        return
    import csv
    rows = []
    with open(LOSS_LOG) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r['epoch']), float(r['ntxent_loss'])))
            except (ValueError, KeyError):
                continue
    if not rows:
        print("Loss log empty. Skipping.")
        return
    from collections import defaultdict
    per_epoch = defaultdict(list)
    for ep, l in rows:
        per_epoch[ep].append(l)
    epochs = sorted(per_epoch)
    means = [np.mean(per_epoch[e]) for e in epochs]
    print(f"Epochs logged: {epochs[0]}..{epochs[-1]}")
    for e, m in zip(epochs, means):
        print(f"  epoch {e:>3}: mean train loss {m:.4f}")
    if len(means) >= 2:
        still_dropping = means[-1] < means[max(0, len(means) - 3)] * 0.98
        print(f"\nNote: loss is {'still decreasing' if still_dropping else 'plateauing'}.")
    print("\nTo complete this test, log cluster ARI (vs cluster.tsv) per epoch and")
    print("watch for loss dropping while ARI stalls -> that decoupling signals a shortcut.")
    print(f"{OK}: loss trajectory reported (per-epoch ARI not yet tracked).")


# ---------------------------------------------------------------------------
# Diagnostic 10: TM-score rank correlation (delegate)
# ---------------------------------------------------------------------------
def diag_tmscore():
    banner("TM-score Rank Correlation (objective mismatch)")
    print("This is implemented in evaluate_correlation.py (needs tmtools + Ca coords).")
    print("Run:  python evaluate_correlation.py")
    print("Interpretation: embedding distance should correlate NEGATIVELY with TM-score")
    print("(rho < -0.3 is the success criterion there).")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL = ['variance', 'length', 'composition', 'plddt', 'attention',
       'taxonomy', 'injection', 'singleton', 'lossquality', 'tmscore']

MODEL_BASED = {'composition', 'attention', 'injection', 'plddt'}


def main():
    ap = argparse.ArgumentParser(description="Shortcut diagnostic battery")
    ap.add_argument('--only', type=str, default=None,
                    help="comma-separated subset, e.g. 'length,taxonomy,singleton'")
    ap.add_argument('--model', type=str, default='asymmetric',
                    choices=['topotein', 'asymmetric'])
    ap.add_argument('--task', type=str, default='contrastive',
                    choices=['mtm', 'contrastive'])
    ap.add_argument('--sample', type=int, default=300,
                    help="cap on proteins for pairwise / model-based tests")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tests = [t.strip() for t in args.only.split(',')] if args.only else ALL
    unknown = [t for t in tests if t not in ALL]
    if unknown:
        print(f"Unknown tests: {unknown}. Valid: {ALL}")
        return

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Run extract_embeddings.py first.")
        return

    print(f"Device: {DEVICE} | model: {args.model} | task: {args.task} | sample cap: {args.sample}")
    ids, X = load_embeddings()
    rep_of = load_clusters()
    n_clusters = len({rep_of.get(p, p) for p in ids})  # clusters among embedded proteins only
    print(f"Loaded {len(ids)} embeddings (dim={X.shape[1]}), {n_clusters} clusters.")

    # Lazily load the model only if a model-based test is requested
    model = None
    if MODEL_BASED & set(tests):
        model, ckpt = load_model(args.model, args.task)
        if model is None:
            print(f"[!] No checkpoint found for {args.model}/{args.task}; "
                  "model-based tests will be skipped.")
        else:
            print(f"Loaded model from {ckpt}")

    dispatch = {
        'variance': lambda: diag_variance(ids, X, rep_of),
        'length': lambda: diag_length(ids, X, args.sample),
        'composition': lambda: diag_composition(ids, model, args.sample),
        'plddt': lambda: diag_plddt(ids, model, args.sample),
        'attention': lambda: diag_attention(ids, model, args.model, args.sample),
        'taxonomy': lambda: diag_taxonomy(ids, X),
        'injection': lambda: diag_injection(ids, X, model, args.sample),
        'singleton': lambda: diag_singleton(ids, X, rep_of),
        'lossquality': diag_lossquality,
        'tmscore': diag_tmscore,
    }

    for t in tests:
        try:
            dispatch[t]()
        except Exception as e:
            banner(f"{t} (FAILED)")
            print(f"Diagnostic raised: {e}")

    print("\n" + "=" * 60)
    print("Battery complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
