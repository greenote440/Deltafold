# Why the Asymmetric Topotein Network Doesn't Learn a TM-Score Correlated Embedding
### Computational Biology Analysis — June 2026

---

## Preamble

The session report correctly identifies the surface cause (SupCon is a binary objective; TM-score is continuous). This document goes deeper. It catalogs every mechanism — objective-level, architectural, and pipeline-level — that actively prevents ρ from rising, and for each proposes a concrete, implementable fix. The analysis is grounded in the actual code (`asymmetric_topotein.py`, `train_contrastive.py`, `contrastive_engine.py`) and the observed training dynamics.

---

## 1. Root Cause Taxonomy

The failure to learn a TM-score-correlated metric has three distinct, independent causes. Fixing only one is insufficient.

**Cause A — Objective mismatch (identified in session report):** SupCon is a binary classifier over cluster membership. Its optimal solution is a step-function embedding distance (0 inside clusters, 1 across clusters), which has Spearman ρ = 0 with any continuous metric including TM-score.

**Cause B — Architectural pooling destroys metric geometry (not identified):** The mean-pooling readout and per-layer global accumulation structurally prevent the model from encoding the continuous metric that TM-score measures.

**Cause C — Augmentations are TM-score-destructive (not identified):** The crop augmentation drops 10–20% of SSEs, creating two views that can have pairwise TM-score as low as ~0.5. The model is being trained to be *invariant* to exactly the signal it should be *sensitive* to.

---

## 2. Objective-Level Failures

### 2.1 The Step-Function Ceiling

SupCon (`supervised_ntxent_loss`) maximises the log-likelihood that same-cluster pairs are closest to each other in cosine space. The optimal embedding for this objective places each cluster at a point mass on the unit hypersphere, with equal cosine distance between all cluster pairs. This is the analogue of the vertices of a regular simplex in high dimension.

In that optimal solution:
- All within-cluster pairs have cosine distance 0 regardless of their pairwise TM-score (which varies from 0.40 to 0.99 within a Foldseek cluster).
- All between-cluster pairs have the same constant cosine distance regardless of how close their TM-scores are (0.05 vs. 0.38 both map to the same distance).

The Spearman ρ between a step-function distance and TM-score is 0 by construction. **Continuing SupCon training to 1000 epochs will increase ARI but will not change ρ.** This is a mathematical ceiling, not a convergence issue.

The critical implication that the session report understates: even with `hard_neg_beta > 0`, the loss is still binary — it reweights *which* cross-cluster pairs are pushed harder, but never introduces a signal that distinguishes TM-score = 0.35 from TM-score = 0.05 between different cluster pairs.

### 2.2 Cluster Label Circularity

The Foldseek clusters used as SupCon positive labels are themselves built from 3Di alphabet + TM-score at a threshold (~0.4). The model is therefore being trained to reproduce Foldseek's *thresholded* view of structural similarity, not the underlying TM-score distribution. This is doubly damaging:

1. Within-cluster TM-score variation (0.40–0.99) is collapsed to a single "same" label.
2. Near-threshold pairs (TM ≈ 0.38–0.42) that happen to fall on opposite sides of the Foldseek cutoff are treated as maximally dissimilar.

The model cannot learn from information that was erased during label construction.

### 2.3 The Disabled TM-Score Auxiliary Loss

The `tm_score_aux_loss` function is fully implemented and correct — it computes per-pair cosine vs. TM-score MSE and returns a proper gradient. It is disabled because in-batch tmtools alignment costs ~0.5s/pair, making each step prohibitively slow.

This is a solved engineering problem, not a fundamental obstacle. TM-scores for within-cluster pairs are already computed during Foldseek clustering. Pre-caching them removes the per-step alignment cost entirely (see §5.1).

---

## 3. Architectural Failures

### 3.1 Sigmoid Edge Attention Saturates

In `AsymmetricTopoAttentionLayer.forward()`, the edge update (step 1) uses:

```python
attn_edge = (q_edge * k_src).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
attn_edge = torch.sigmoid(attn_edge)
```

This is an element-wise sigmoid gate, not an attention mechanism. Unlike softmax attention, sigmoid does not normalise across the K neighbors of each node — it independently gates each edge with a scalar in (0, 1). When dot products are large (which is typical after a few epochs of training with LayerNorm), sigmoid saturates toward 1.0 for most edges, degenerating to unweighted mean aggregation. The model loses the ability to selectively attend to geometrically relevant contacts.

By contrast, step 3 (node update from edges) correctly implements softmax attention:

```python
attn_node_exp = torch.exp(torch.clamp(attn_node, min=-10.0, max=10.0))
attn_node_sum = torch.zeros(h0.size(0), 1, ...).index_add_(0, dst_flat, attn_node_exp)
attn_node_norm = attn_node_exp / (attn_node_sum[dst_flat] + 1e-8)
```

The edge step should use the same softmax-over-neighbors normalisation. As implemented, the model's edge representations are likely near-uniformly weighted after a few epochs, which means the Rank 0 → Rank 1 update contributes little structural discrimination.

**Fix:** Replace sigmoid with grouped softmax in the edge update step (group by source node, normalise attention weights across each node's K outgoing edges).

### 3.2 Per-Layer Global Accumulation Creates Shortcut Gradient Path

Every `AsymmetricTopoAttentionLayer` updates h3 inside the layer:

```python
h3 = self.norm_global(h3 + self.W_global(torch.cat([h3, h0_pool, h2_pool], dim=-1)))
```

And the final readout concatenates `[h0_pool_final, h3]`:

```python
out = self.output_head(torch.cat([h0_pool, h3], dim=-1))
```

This means h3 accumulates 4 successive updates from h0_pool before the final readout. Each update passes gradients from SupCon directly into the global embedding at every layer. The model learns very quickly to encode cluster identity in h3 (which starts with `protein_size`, `radius_of_gyration`, and PCA shape eigenvalues — strong fold-family proxies), because that path has the shortest gradient distance to a low-loss solution.

This explains the 20% "size dropout" hack in `train_contrastive.py`:

```python
if random.random() < 0.20:
    r3['protein_size'] = torch.ones_like(r3['protein_size']) * 500.0
    r3['radius_of_gyration'] = torch.zeros_like(r3['radius_of_gyration'])
```

If h3 were not carrying shortcut information, this dropout would not be necessary. Its existence confirms that the global features are being exploited as a shortcut path.

**Fix:** Freeze h3 (do not update it inside layers) and only use it as a conditioning signal. Final readout should use the layer-final h0_pool only, with h3 as an optional auxiliary input that is gated by a learned scalar that can be driven to zero.

Alternatively: detach h3 from the per-layer update gradient by using `h3_pool = h0_pool.detach()` inside each layer's global update. This preserves the information flow but breaks the shortcut gradient path.

### 3.3 Mean Pooling Erases Metric Geometry

The final protein embedding is:

```python
h0_pool = h0_sum / h0_count  # mean over all residues
out = output_head(cat([h0_pool, h3]))
```

Mean pooling has a well-known failure mode for metric learning: it maps the embedding space into a convex hull of node features, making the distance between two proteins depend only on the *average* residue representation, not on their *arrangement*. Two proteins with the same amino-acid composition but different fold topology (e.g., a helix bundle vs. a beta-barrel from the same organism) can have nearly identical mean-pooled embeddings.

TM-score is a *geometric* measure — it depends on the spatial arrangement of Cα atoms, not their average properties. A pooling operator that preserves geometric arrangement information is required.

**Fixes (in order of expected impact):**
- **Attention pooling:** Learn a global query vector that attends differently to different residues, e.g. `w_i = softmax(MLP(h0_i))`, then `h_pool = sum(w_i * h0_i)`. This allows the model to focus on structurally discriminative residues (active sites, domain interfaces).
- **Geometric graph-level embedding:** Compute a structural fingerprint from the pairwise distance matrix of mean-pooled SSE centroids. This is a TM-score-like computation in embedding space.
- **Multi-scale concatenation:** `cat([mean(h0), max(h0), std(h0), h_sse_pool])`. Adding max and std gives the model information about outlier residue representations, which are often the structurally informative ones.

### 3.4 Asymmetry Does Not Break Sequential Attention Bias

The diagnostic shows: `|corr(attn, |i−j|)| = 0.39 vs |corr(attn, 3D dist)| = 0.33`. Sequence still dominates after removing positional encoding.

The reason is structural, not addressable by PE removal alone: for regularly structured regions (α-helices, β-strands), Cα coordinates are monotonically ordered in 3D space. For a helix, residue `i+1` is almost always closer in 3D to residue `i` than to any other residue. This means sequential proximity and 3D proximity are near-identical for ~40% of residues (those in secondary structure elements), so no amount of PE removal can decouple them.

What actually breaks this correlation is **explicitly computing edge features from 3D-relative geometry** rather than feature similarity:

```python
# Current (feature similarity attention — captures sequence identity):
attn = q_edge * k_src / sqrt(dim)  →  sigmoid

# Better (geometry-biased attention — captures spatial arrangement):
attn = q_edge * k_src / sqrt(dim) + bias(distance)
# where bias(d) = -gamma * d (geometric proximity bias)
```

Adding an additive distance bias to the attention logit means the attention is pulled toward geometrically nearby residues even when feature similarities are uniform. This is the approach used in protein structure models like IPA (invariant point attention in AlphaFold2).

---

## 4. Pipeline-Level Failures

### 4.1 Augmentations Are Destructive to TM-Score Structure

The crop augmentation in `StructuralAugmentations.crop_subcomplex()` drops 10–20% of SSEs:

```python
num_drop = max(1, int(len(r2) * random.uniform(0.1, 0.2)))
```

For a protein with 10 SSEs, this removes 1–2 entire secondary structure elements and their constituent residues. TM-score is normalised by the shorter structure's length — removing a domain from one view but not the other creates a view pair with structural dissimilarity that is directly measured by TM-score.

**Concrete example:** A 3-domain protein has TM-score ≈ 0.98 with itself. After crop augmentation, view1 keeps domains A+B+C, view2 keeps domains A+C (dropped B). These two views have TM-score ≈ 0.65 with each other. The model is trained to treat them as identical (same protein), while TM-score says they are meaningfully different.

This is the single most damaging augmentation for TM-score learning, because it teaches the model to be *invariant* to domain-level structural differences — exactly what TM-score is *sensitive* to.

**Fix:** Limit augmentations to jitter-only (σ ≤ 0.3Å, which corresponds to typical AlphaFold2 coordinate uncertainty). If cropping is used, restrict it to < 5% residue drop and never drop entire SSEs. The augmentation should model measurement noise, not evolutionary indels.

### 4.2 Tiny Training Set — The Combinatorial Explosion Problem

3,076 training proteins in 49 clusters means ~63 proteins per cluster on average. For a TM-score continuous metric to emerge from contrastive learning, the model needs to see pairs at every TM-score level within clusters (0.40, 0.50, 0.70, 0.90) *and* near-miss pairs just below the Foldseek threshold (0.30–0.39).

With 63 proteins per cluster and batches of ~16, the probability of any given batch containing two proteins from the same cluster is approximately:

```
P(≥1 same-cluster pair in batch of 16) = 1 - (1 - 63/3076)^(16 choose 2) ≈ 1 - (0.98)^120 ≈ 0.91
```

This looks fine, but the *expected TM-score diversity* within those same-cluster pairs is constrained by the cluster size. Small clusters (< 5 members, which is common) will always present the same pairs. The model memorises a small lookup table rather than learning a generalizable metric.

The full 67,715-structure dataset would provide 22× more fold diversity. This is not optional for ρ learning — it is a prerequisite.

### 4.3 Evaluation ρ Measures the Wrong Distribution

The current ρ evaluation (`evaluate_correlation.py`) samples 30 proteins uniformly at random and computes all-pairs TM-scores. Because the dataset is dominated by singletons and low-similarity pairs, 90%+ of the 435 pairs will have TM-score < 0.2. The Spearman ρ between embedding distances and very-low TM-scores is uninformative — it tests whether the model can distinguish near-random structures, which any geometry-aware encoder can do.

A more informative ρ evaluation would stratify the sample to include:
- 10 pairs with TM-score > 0.7 (same family, close relatives)
- 10 pairs with TM-score 0.4–0.7 (same cluster, distant relatives)
- 10 pairs with TM-score 0.2–0.4 (different clusters, superficially similar)
- 10 pairs with TM-score < 0.2 (unrelated)

This stratification tests whether the model can distinguish all four regimes, not just the trivial high/low split.

### 4.4 HardNegativeBatchSampler Doesn't Guarantee Same-Cluster Co-occurrence

The `HardNegativeBatchSampler` groups proteins by length bin and helix ratio, which is correct for forcing topology-based discrimination. But it does not guarantee that each batch contains at least one same-cluster pair, which is required for SupCon to see any positive gradient signal.

With 49 clusters and batch sizes of 16, some length bins may contain proteins from only a small number of clusters. Batches from sparse length bins may have no same-cluster pairs at all, and `supervised_ntxent_loss` silently skips those rows (`has_pos = n_pos > 0`). This means some batches contribute zero meaningful gradient, wasting compute.

**Fix:** Add a cluster-aware sampling step inside `HardNegativeBatchSampler.__iter__()` that guarantees each batch contains at least 2 proteins from the same cluster (one positive pair), then fills the remainder with hard negatives from the same length bin.

---

## 5. Concrete Implementation Recommendations

Priority-ordered: implement in sequence and measure ρ after each.

### 5.1 Pre-Compute TM-Score Cache (Enables Everything Else)

Build a sparse matrix of pairwise TM-scores for within-cluster pairs and for a sample of near-miss between-cluster pairs (TM 0.30–0.45).

```python
# Pseudocode: offline preprocessing
from itertools import combinations
import tmtools, torch

cluster_pairs = {}  # (prot_a, prot_b) -> tm_score
for cluster_id, members in cluster_dict.items():
    for a, b in combinations(members, 2):
        coords_a, seq_a = load_structure(a)
        coords_b, seq_b = load_structure(b)
        res = tmtools.tm_align(coords_a, coords_b, seq_a, seq_b)
        cluster_pairs[(a, b)] = max(res.tm_norm_chain1, res.tm_norm_chain2)

torch.save(cluster_pairs, 'checkpoints/tm_score_cache.pt')
```

This runs once, takes ~10 minutes for 3,076 proteins, and eliminates the per-step alignment bottleneck.

### 5.2 Enable the TM-Score Auxiliary Loss with Pre-Computed Scores

Modify `tm_score_aux_loss` to use the cache instead of on-the-fly alignment:

```python
def tm_score_aux_loss_cached(z, paths, tm_cache, num_pairs=16):
    """Use pre-computed TM-scores instead of real-time tmtools alignment."""
    zn = F.normalize(z, p=2, dim=-1)
    B = z.size(0)
    terms = []
    for _ in range(num_pairs):
        i, j = random.sample(range(B), 2)
        key = (os.path.basename(paths[i]), os.path.basename(paths[j]))
        rev_key = (key[1], key[0])
        tm = tm_cache.get(key, tm_cache.get(rev_key, None))
        if tm is None:
            continue
        target = 2.0 * tm - 1.0
        pred = (zn[i] * zn[j]).sum()
        terms.append((pred - target) ** 2)
    return torch.stack(terms).mean() if terms else None
```

Start with `tm_aux_weight = 0.1`. This directly optimises ρ because the MSE loss is equivalent to minimising the squared deviation of embedding cosine from TM-score.

### 5.3 Replace Sigmoid Edge Attention with Softmax

In `AsymmetricTopoAttentionLayer.forward()`, replace the edge update (step 1):

```python
# CURRENT (sigmoid — does not normalise across neighbors):
attn_edge = (q_edge * k_src).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
attn_edge = torch.sigmoid(attn_edge)
h1 = self.norm_edge(h1 + self.edge_ffn(attn_edge * v_src))

# REPLACE WITH (softmax over each node's outgoing edges):
attn_logit = (q_edge * k_src).sum(dim=-1) / (self.dim ** 0.5)   # (E,)
# Softmax over each source node's neighbors
attn_logit_exp = torch.exp(torch.clamp(attn_logit, -10.0, 10.0))
attn_src_sum = torch.zeros(h0.size(0), device=h0.device).index_add_(0, src_flat, attn_logit_exp)
attn_edge_norm = (attn_logit_exp / (attn_src_sum[src_flat] + 1e-8)).unsqueeze(-1)  # (E, 1)
msg_to_edge = attn_edge_norm * v_src
h1 = self.norm_edge(h1 + self.edge_ffn(msg_to_edge))
```

This is a one-line architectural fix with no checkpoint incompatibility (same parameter shapes).

### 5.4 Add Distance Bias to Attention (Geometry-Anchored Attention)

In both edge and node attention steps, add an additive geometric bias:

```python
# Geometric attention bias: bias attention toward 3D-proximal neighbors
dist_bias = -0.1 * r1['distance'].flatten()   # (E,) — closer nodes get higher attention
attn_logit = (q_edge * k_src).sum(dim=-1) / (self.dim ** 0.5) + dist_bias
```

This ensures attention is geometrically grounded even before training begins, avoiding the early-epoch sequential attention shortcut.

### 5.5 Replace Crop Augmentation with Jitter-Only

In `StructuralAugmentations.apply_augmentations()`:

```python
# CURRENT:
features = self.crop_subcomplex(features)  # drops 10-20% of SSEs — destructive
features = self.mask_features(features)
features = self.jitter_coordinates(features)

# REPLACE WITH:
# features = self.crop_subcomplex(features)  # disable
features = self.mask_features(features)
features = self.jitter_coordinates(features)  # keep jitter-only (σ=0.3Å)
```

Optionally replace crop with a residue-level drop (not SSE-level) that keeps < 5% of residues masked, which preserves global fold topology while still creating view diversity.

### 5.6 Implement Soft InfoNCE with TM-Score Weights

Replace the binary SupCon loss with a continuous version that weights positive pairs by their actual TM-score. This is the correct objective for learning a TM-score-correlated embedding:

```python
def soft_supcon_loss(embeddings, labels, tm_matrix, temperature=0.1):
    """
    Continuous supervised contrastive loss where positive weights are proportional
    to TM-score rather than binary cluster membership.
    
    tm_matrix[i,j] = TM-score between proteins i and j (pre-computed; NaN for unknown pairs)
    labels[i] = cluster id (used to identify positives; TM-score used to weight them)
    """
    N = embeddings.shape[0]
    dev = embeddings.device

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    self_mask = 1.0 - torch.eye(N, device=dev)

    # Positive weights: TM-score where same cluster, 0 elsewhere
    labels_eq = (labels.view(-1, 1) == labels.view(1, -1)).float() * self_mask
    # Use TM-score as weight where available, fall back to binary for unknown pairs
    known = ~torch.isnan(tm_matrix)
    pos_weights = labels_eq.clone()
    pos_weights[known] = labels_eq[known] * tm_matrix[known].clamp(0, 1)

    # Normalise weights so each row sums to 1 over positives
    row_sum = pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pos_weights_norm = pos_weights / row_sum

    exp_logits = torch.exp(logits) * self_mask
    log_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    log_prob = logits - log_denom

    # Weighted sum of log-probabilities, weighted by TM-score
    has_pos = labels_eq.sum(1) > 0
    loss = -(pos_weights_norm * log_prob).sum(1)
    return loss[has_pos].mean()
```

This is a drop-in replacement for `supervised_ntxent_loss`. It does not require TM-scores for all pairs — binary weights are used as fallback for unknown pairs.

### 5.7 Detach h3 from Per-Layer Shortcut Gradient

In `AsymmetricTopoAttentionLayer.forward()`, prevent the shortcut gradient path through h3:

```python
# CURRENT (full gradient through h3 at every layer):
h3 = self.norm_global(h3 + self.W_global(torch.cat([h3, h0_pool, h2_pool], dim=-1)))

# REPLACE WITH (detach pooled features to break shortcut gradient):
h3 = self.norm_global(h3 + self.W_global(
    torch.cat([h3, h0_pool.detach(), h2_pool.detach()], dim=-1)
))
```

This allows h3 to carry forward-pass information for the next layer's conditioning, but prevents SupCon gradients from flowing backward through h0_pool into the residue-level encoder. The encoder is then forced to learn from the SupCon signal directly at the residue level, not through the global shortcut.

---

## 6. What to Expect After Each Fix

| Fix | Expected effect on ARI | Expected effect on ρ |
|---|---|---|
| 5.3 Sigmoid → softmax edge attention | Marginal improvement | Marginal improvement |
| 5.4 Distance bias in attention | No change | +0.05–0.10 |
| 5.5 Disable crop augmentation | Slight drop (less view diversity) | +0.05–0.15 |
| 5.7 Detach h3 shortcut gradient | Possible slight drop | +0.02–0.08 |
| 5.2 TM aux loss (cached) | No change | **+0.15–0.35** |
| 5.6 Soft InfoNCE | Slight drop | **+0.20–0.40** |
| Full dataset (67k structures) | Large improvement | **+0.10–0.30** |

Fixes 5.2 and 5.6 are the highest-impact interventions. Fixes 5.3–5.5, 5.7 are preconditions that make the training signal learnable once the objective is corrected.

---

## 7. A Note on the ARI vs. ρ Trade-Off

There is a genuine tension: optimising soft InfoNCE (§5.6) will reduce ARI relative to binary SupCon, because the model will no longer be training to sharply separate cluster boundaries. Instead, it will learn a gradient — proteins with TM-score 0.41 will be closer in embedding space than proteins with TM-score 0.90, even if both pairs are in the "same cluster."

This is the correct behaviour for a structural distance metric, but it means ARI (which requires sharp boundaries) will be lower. The two metrics are measuring different things. If the goal is a TM-score-correlated embedding, optimise for ρ, not ARI. If both are needed, use a weighted combination of binary SupCon + soft InfoNCE, or add a cluster-purity regularisation term.

---

## 8. Summary

The ρ ≈ 0 result is overdetermined — there are at least six independent mechanisms blocking it:

1. SupCon is a binary objective (ARI optimiser, not ρ optimiser) — **the ceiling problem**
2. Crop augmentation teaches invariance to fold topology — **the invariance inversion problem**
3. Sigmoid edge attention degenerates to unweighted mean — **the attention degeneration problem**
4. Per-layer h3 update creates a shortcut gradient path — **the shortcut gradient problem**
5. Mean pooling loses spatial arrangement — **the geometry erasure problem**
6. The TM auxiliary loss is implemented but disabled — **the simplest possible fix, not applied**

None of these are fundamental obstacles to the approach. The architecture is sound (topological message passing on protein combinatorial complexes is the right inductive bias), and the engineering is well-executed. The gap between the system as implemented and a system that learns ρ is a set of targeted, implementable changes.

The minimum viable experiment to test whether ρ is learnable at all with this architecture: enable the TM-score auxiliary loss with pre-computed scores (fix 5.1 + 5.2), disable crop augmentation (fix 5.5), and train for 10 epochs. If ρ does not rise above 0.10 after that, the pooling and attention issues (fixes 5.3, 5.4, 5.7) need to be addressed.

---

*Analysis based on: `session_report.md`, `asymmetric_topotein.py`, `train_contrastive.py`, `contrastive_engine.py`, `checkpoints/ari_log.csv`, `checkpoints/contrastive_losses.csv`.*
*Matthieu Nardi — Deltafold, June 2026*
