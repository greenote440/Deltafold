"""
Contrastive Learning Engine for Deltafold.

This module holds the two halves of the self-/weakly-supervised contrastive
objective:

  1. `StructuralAugmentations` — generates two augmented "views" of a Protein
     Combinatorial Complex (coordinate jitter, optional SSE cropping, feature
     masking).
  2. The loss family that turns those views (and optional cluster labels /
     TM-scores) into a training signal:
       * `NTXentLoss`              — unsupervised InfoNCE, optional hard-negative
                                     reweighting.
       * `supervised_ntxent_loss`  — SupCon over cluster labels (binary positives).
       * `soft_supcon_loss`        — SupCon with TM-score-weighted positives.
       * `tm_score_aux_loss[_cached]` — auxiliary TM-score regression.
       * `build_tm_matrix`         — dense in-batch TM-score matrix from the cache.
       * `collapse_metrics`        — representation-collapse diagnostics.

The §-references throughout point to tm_score_analysis.md, which motivates each
objective variant. This module depends only on torch/numpy (+ optional tmtools);
it is deliberately free of any training-loop or dataset imports.
"""
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralAugmentations:
    """
    Generates on-the-fly augmented 'views' of a Protein Combinatorial Complex (PCC).
    Applies Coordinate Jitter, Sub-Complex Cropping, and Feature Masking.
    """
    def __init__(self, jitter_sigma=0.5, drop_ratio_range=(0.1, 0.2), mask_ratio=0.15,
                 use_crop=True):
        self.jitter_sigma = jitter_sigma
        self.drop_min = drop_ratio_range[0]
        self.drop_max = drop_ratio_range[1]
        self.mask_ratio = mask_ratio
        # TM-score analysis §5.5: SSE-level cropping drops 10-20% of secondary
        # structure elements, which can drive the pairwise TM-score of two views of
        # the SAME protein down to ~0.5-0.65. Training the model to treat those views
        # as identical teaches invariance to exactly the domain-level structural
        # differences TM-score measures. use_crop=False (recommended for TM-score
        # correlation) restricts augmentation to measurement-noise jitter + masking.
        # Default True preserves the original behavior for existing callers.
        self.use_crop = use_crop

    def clone_pcc(self, obj):
        """Fast recursive clone to bypass Python's disastrously slow copy.deepcopy for tensors."""
        if isinstance(obj, dict):
            return {k: self.clone_pcc(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clone_pcc(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.clone()
        return obj

    def __call__(self, pcc_features):
        """
        Takes a single, unbatched PCC dictionary and returns two augmented views.
        """
        view1 = self.apply_augmentations(self.clone_pcc(pcc_features))
        view2 = self.apply_augmentations(self.clone_pcc(pcc_features))
        return view1, view2

    def apply_augmentations(self, features):
        if self.use_crop:  # §5.5: disabled by default in the TM-score training path
            features = self.crop_subcomplex(features)
        features = self.mask_features(features)
        features = self.jitter_coordinates(features)
        return features

    def jitter_coordinates(self, features):
        """
        Micro-Augmentation: Adds Gaussian noise to Rank 0 spatial coordinates.
        Recomputes the Rank 1 interaction graph distances to maintain geometric integrity.
        """
        r0 = features['rank0']
        if 'ca_coords' in r0 and r0['ca_coords'] is not None:
            noise = torch.randn_like(r0['ca_coords']) * self.jitter_sigma
            r0['ca_coords'] = r0['ca_coords'] + noise
            
            # Recompute local interaction graph properties affected by spatial jitter
            if 'rank1' in features and 'distance' in features['rank1']:
                r1 = features['rank1']
                src, dst = r1['source'], r1['target']
                new_vectors = r0['ca_coords'][dst] - r0['ca_coords'][src]
                new_distances = torch.norm(new_vectors, dim=-1)
                
                r1['vector'] = new_vectors
                r1['distance'] = new_distances
                
                # Re-generate distance encoding if needed by the layers
                d_model = r1['distance_encoding'].shape[-1]
                div_term = torch.exp(torch.arange(0, d_model, 2, device=new_vectors.device).float() * -(math.log(10000.0) / d_model))
                pe = torch.zeros(new_distances.numel(), d_model, device=new_vectors.device)
                dist_flat = new_distances.flatten().unsqueeze(1)
                pe[:, 0::2] = torch.sin(dist_flat * div_term)
                pe[:, 1::2] = torch.cos(dist_flat * div_term)
                r1['distance_encoding'] = pe.view(new_distances.shape[0], new_distances.shape[1], d_model)
                
        return features

    def crop_subcomplex(self, features):
        """
        Macro-Augmentation: Randomly drops 10% to 20% of Rank 2 secondary structures 
        and their corresponding Rank 0/1 constituents to simulate evolutionary indels.
        """
        r2 = features['rank2']
        if not r2 or len(r2) <= 2:
            return features # Protein is too small to crop safely

        num_drop = max(1, int(len(r2) * random.uniform(self.drop_min, self.drop_max)))
        drop_indices = set(random.sample(range(len(r2)), num_drop))

        new_r2 = []
        nodes_to_keep = []
        node_mapping = {} # Maps old node indices to new contiguous indices
        
        current_new_idx = 0
        for i, sse in enumerate(r2):
            if i not in drop_indices:
                new_start = current_new_idx
                start, end = sse['start_idx'], sse['end_idx']
                
                for old_idx in range(start, end + 1):
                    nodes_to_keep.append(old_idx)
                    node_mapping[old_idx] = current_new_idx
                    current_new_idx += 1
                
                new_sse = sse.copy()
                new_sse['start_idx'] = new_start
                new_sse['end_idx'] = current_new_idx - 1
                new_r2.append(new_sse)

        if not nodes_to_keep:
            return features # Failsafe
            
        # Failsafe: Ensure we have enough nodes left to maintain K-NN graph shape
        K = features['rank1']['source'].shape[1]
        if len(nodes_to_keep) <= K:
            return features

        features['rank2'] = new_r2

        # Filter Rank 0 Node properties
        r0 = features['rank0']
        for k, v in r0.items():
            if isinstance(v, torch.Tensor) and v.shape[0] > 0:
                r0[k] = v[nodes_to_keep]

        # Reconstruct Rank 1 Edge Graph on the remaining nodes
        new_coords = r0['ca_coords']
        n_res = new_coords.shape[0]
        k_actual = K + 1
        
        dist_matrix = torch.cdist(new_coords, new_coords)
        distances, indices = torch.topk(dist_matrix, k=k_actual, largest=False, dim=1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        sources = torch.arange(n_res, device=new_coords.device).view(-1, 1).expand(-1, k_actual - 1)
        
        features['rank1']['source'] = sources
        features['rank1']['target'] = indices
        features['rank1']['distance'] = distances
        
        # Update Rank 3 global tracker
        features['rank3']['protein_size'] = n_res
        
        return features

    def mask_features(self, features):
        """
        Feature Masking: Randomly masks out the amino acid scalar type.
        """
        r0 = features['rank0']
        n_res = r0['aa'].shape[0]
        if n_res > 0:
            num_mask = max(1, int(n_res * self.mask_ratio))
            mask_indices = random.sample(range(n_res), num_mask)
            r0['aa'][mask_indices] = 0.0
        return features

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (InfoNCE).

    Hard-negative mining (report 7, "the single most important intervention"):
    when `hard_neg_beta` > 0, negatives are reweighted toward the hardest ones
    (those most similar to the anchor) following Robinson et al. (2021),
    "Contrastive Learning with Hard Negative Samples". beta=0 recovers the
    standard, uniformly-weighted InfoNCE (exact fast path preserved).
    """
    def __init__(self, temperature=0.1, hard_neg_beta=0.0):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_beta = hard_neg_beta
        self.criterion = nn.CrossEntropyLoss()
        self._mask_cache = {}
        self._label_cache = {}

    def forward(self, z_i, z_j):
        # Normalize the global Rank 3 embeddings to the unit hypersphere
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0) # (2B, dim)
        N = 2 * B

        sim_matrix = torch.mm(z, z.t()) / self.temperature

        if B not in self._mask_cache:
            self._mask_cache[B] = torch.eye(N, dtype=torch.bool, device=z.device)
            self._label_cache[B] = torch.cat([torch.arange(B, 2*B, device=z.device),
                                              torch.arange(B, device=z.device)])
        self_mask = self._mask_cache[B]
        labels = self._label_cache[B]

        # Fast path: standard InfoNCE (uniform negatives)
        if self.hard_neg_beta <= 0:
            sim_matrix = sim_matrix.masked_fill(self_mask, -float('inf'))
            return self.criterion(sim_matrix, labels)

        # Hard-negative-weighted InfoNCE (log-space for numerical stability).
        rows = torch.arange(N, device=z.device)
        pos = sim_matrix[rows, labels]                      # (N,) positive logits

        neg_mask = ~self_mask.clone()
        neg_mask[rows, labels] = False                      # exclude the positive

        num_neg = neg_mask.sum(dim=1).clamp(min=1).float()  # (N,)

        # Per-negative weights from detached similarity, normalized so the
        # weights over each row's negatives sum to num_neg (=> beta=0 is uniform).
        w_logits = (self.hard_neg_beta * sim_matrix.detach()).masked_fill(~neg_mask, -float('inf'))
        log_norm = torch.logsumexp(w_logits, dim=1, keepdim=True)
        log_w = torch.log(num_neg).unsqueeze(1) + w_logits - log_norm  # (N,N), -inf off-negatives

        weighted = sim_matrix + log_w                       # scale negative logits
        weighted[rows, labels] = pos                        # positive keeps weight 1
        weighted = weighted.masked_fill(self_mask, -float('inf'))

        log_denom = torch.logsumexp(weighted, dim=1)
        return (log_denom - pos).mean()


# Amino-acid alphabet used to reconstruct one-letter sequences for tmtools
# alignment (mirrors topotein_lifter / evaluate_correlation).
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO"


def collapse_metrics(z):
    """Per-batch collapse signals (report 7, "collapse detection during training").
    Returns (embedding std, mean off-diagonal cosine similarity). A shrinking std
    or a mean cosine approaching 1.0 indicates representation collapse."""
    with torch.no_grad():
        zn = F.normalize(z, p=2, dim=-1)
        emb_std = zn.std(dim=0).mean().item()
        n = zn.size(0)
        sims = zn @ zn.t()
        off = sims[~torch.eye(n, dtype=torch.bool, device=zn.device)]
        mean_cos = off.mean().item()
    return emb_std, mean_cos


def tm_score_aux_loss(z, features, num_pairs=8):
    """Auxiliary TM-score regression (report 7, "add a TM-score regression loss").
    Directly supervises embedding cosine similarity to track TM-score ordering,
    reducing the contrastive/TM-score objective mismatch (report 3.4). Computed on
    a few sampled in-batch pairs; tmtools alignment runs on CPU. Returns None if no
    pair could be aligned."""
    try:
        import tmtools
    except ImportError:
        return None
    import numpy as np

    batch_idx = features['batch_idx_0']
    ca = features['rank0']['ca_coords']
    aa = features['rank0']['aa']
    B2 = z.size(0)

    coords, seqs = [], []
    for b in range(B2):
        m = (batch_idx == b)
        c = ca[m].detach().cpu().numpy().astype(np.float32)
        if c.shape[0] < 3:
            coords.append(None); seqs.append(None); continue
        idx = aa[m].detach().cpu().numpy().argmax(axis=1)
        seqs.append("".join(AA_ALPHABET[i] if i < len(AA_ALPHABET) else "X" for i in idx))
        coords.append(c)

    valid = [b for b in range(B2) if coords[b] is not None]
    if len(valid) < 2:
        return None

    zn = F.normalize(z, p=2, dim=-1)
    terms = []
    for _ in range(num_pairs):
        i, j = random.sample(valid, 2)
        try:
            res = tmtools.tm_align(coords[i], coords[j], seqs[i], seqs[j])
            tm = max(res.tm_norm_chain1, res.tm_norm_chain2)
        except Exception:
            continue
        target = 2.0 * tm - 1.0          # map TM [0,1] -> cosine target [-1,1]
        pred = (zn[i] * zn[j]).sum()     # cosine similarity (z is unit-normalized)
        terms.append((pred - target) ** 2)
    if not terms:
        return None
    return torch.stack(terms).mean()


def tm_score_aux_loss_cached(z, paths, tm_cache, num_pairs=16):
    """Cached variant of the TM-score regression auxiliary (analysis §5.2).

    Uses pre-computed pairwise TM-scores (see build_tm_cache.py) instead of paying
    ~0.5s/pair on-the-fly tmtools alignment, which is what kept tm_score_aux_loss
    disabled. Directly optimises rho: it pulls each pair's embedding cosine toward
    2*TM-1, so minimising it is minimising the squared deviation of cosine from
    TM-score. `paths` is the 2B view-path list (aligned to z); two augmented views
    of the same protein (equal basename) get target TM=1."""
    B = z.size(0)
    if B < 2 or not tm_cache:
        return None
    zn = F.normalize(z, p=2, dim=-1)
    bns = [os.path.basename(p) for p in paths]
    terms = []
    for _ in range(num_pairs):
        i, j = random.sample(range(B), 2)
        if bns[i] == bns[j]:
            tm = 1.0  # two augmented views of the same protein
        else:
            tm = tm_cache.get((bns[i], bns[j]))
            if tm is None:
                tm = tm_cache.get((bns[j], bns[i]))
        if tm is None:
            continue
        target = 2.0 * float(tm) - 1.0      # TM [0,1] -> cosine target [-1,1]
        pred = (zn[i] * zn[j]).sum()
        terms.append((pred - target) ** 2)
    if not terms:
        return None
    return torch.stack(terms).mean()


def build_tm_matrix(paths, tm_cache, device):
    """Dense N x N TM-score matrix for the in-batch proteins from the sparse cache
    (analysis §5.6). Unknown pairs are NaN (soft_supcon_loss falls back to binary
    weights there); identical basenames (the two views of one protein) are 1.0."""
    n = len(paths)
    bns = [os.path.basename(p) for p in paths]
    mat = torch.full((n, n), float('nan'), device=device)
    for i in range(n):
        for j in range(i + 1, n):
            if bns[i] == bns[j]:
                tm = 1.0
            else:
                tm = tm_cache.get((bns[i], bns[j]))
                if tm is None:
                    tm = tm_cache.get((bns[j], bns[i]))
            if tm is not None:
                mat[i, j] = mat[j, i] = float(tm)
    return mat


def soft_supcon_loss(embeddings, labels, tm_matrix, temperature=0.1):
    """Continuous (soft) supervised contrastive loss (analysis §5.6).

    Identical structure to supervised_ntxent_loss, but positive pairs are weighted
    by their actual TM-score rather than a binary same-cluster label. This turns the
    objective from "collapse each cluster to a point" (rho ceiling = 0) into "place
    proteins at embedding distances proportional to their structural distance" --
    the correct objective for a TM-score-correlated metric. Same-cluster pairs with
    no cached TM-score fall back to binary weight 1.0."""
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    N = embeddings.shape[0]
    dev = embeddings.device

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    self_mask = 1.0 - torch.eye(N, device=dev)
    labels = labels.contiguous().view(-1, 1)
    labels_eq = torch.eq(labels, labels.T).float().to(dev) * self_mask   # same cluster, not self

    # Positive weights: TM-score where same-cluster AND cached, else binary fallback.
    pos_weights = labels_eq.clone()
    known = ~torch.isnan(tm_matrix)
    if known.any():
        pos_weights[known] = labels_eq[known] * tm_matrix[known].clamp(0, 1)
    row_sum = pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pos_weights_norm = pos_weights / row_sum

    exp_logits = torch.exp(logits) * self_mask
    log_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    log_prob = logits - log_denom

    has_pos = labels_eq.sum(1) > 0
    loss = -(pos_weights_norm * log_prob).sum(1)
    return loss[has_pos].mean() if has_pos.any() else logits.sum() * 0.0


def supervised_ntxent_loss(embeddings, labels, temperature=0.1, hard_neg_beta=0.0):
    """
    Supervised Contrastive Loss (SupCon) with optional hard-negative reweighting.

    hard_neg_beta > 0 up-weights negatives that are most similar to the anchor
    (the hardest ones) in the denominator, matching the same mechanism used by
    NTXentLoss on the unsupervised path.  beta=0 is the standard SupCon loss.
    """
    # embeddings: (N, dim), labels: (N,)
    N = embeddings.shape[0]
    dev = embeddings.device

    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(dev)       # 1 where same cluster

    logits = torch.matmul(embeddings, embeddings.T) / temperature

    # Numerical stability shift
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # self_mask[i,i] = 0, elsewhere 1
    self_mask = 1.0 - torch.eye(N, device=dev)
    # positive mask excludes self
    pos_mask = pos_mask * self_mask
    # negative mask: not self, not positive
    neg_mask = self_mask * (1.0 - pos_mask)

    if hard_neg_beta <= 0:
        # Standard SupCon denominator: all non-self terms
        exp_logits = torch.exp(logits) * self_mask
        log_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    else:
        # Reweight negatives toward hard ones (same log-space trick as NTXentLoss).
        # Positives keep weight 1; negatives get weight proportional to exp(beta*sim).
        num_neg = neg_mask.sum(1, keepdim=True).clamp(min=1)
        w_logits = (hard_neg_beta * logits.detach()).masked_fill(neg_mask == 0, -float('inf'))
        log_norm = torch.logsumexp(w_logits, dim=1, keepdim=True)
        log_w_neg = torch.log(num_neg) + w_logits - log_norm   # (N,N), -inf off-negatives

        # Weighted denominator: positives contribute exp(logit), negatives exp(logit+log_w)
        pos_contrib = (torch.exp(logits) * pos_mask).sum(1, keepdim=True)
        neg_contrib = torch.exp(logits + log_w_neg.masked_fill(neg_mask == 0, -float('inf')))
        neg_contrib = neg_contrib.sum(1, keepdim=True)
        log_denom = torch.log(pos_contrib + neg_contrib + 1e-6)

    log_prob = logits - log_denom
    # Mean log-likelihood over positives (ignore rows with no positive)
    n_pos = pos_mask.sum(1)
    has_pos = n_pos > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (n_pos + 1e-6)
    return -mean_log_prob_pos[has_pos].mean() if has_pos.any() else logits.sum() * 0.0