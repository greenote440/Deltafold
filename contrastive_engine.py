"""
Contrastive Learning Engine for Deltafold
Implements structural augmentations and InfoNCE loss to train Topological 
Representations without labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

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