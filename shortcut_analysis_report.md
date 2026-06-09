# Shortcut Analysis: Asymmetric Attention Network with Contrastive Learning
### Eukaryotic Virome Structural Classification — Internship Report

---

## Executive Summary

Contrastive learning over protein structural representations is a compelling approach to learning a topology-aware embedding space. However, the combination of (1) asymmetric attention architecture, (2) structural data derived from computational predictions, and (3) a highly imbalanced, taxonomically biased dataset creates a dense landscape of shortcut opportunities. This report catalogs the most likely shortcut mechanisms, ranked roughly by probability, and proposes diagnostic tests for each.

The central concern is this: **a model can achieve low contrastive loss without learning structurally meaningful representations**. The shortcuts below are paths by which that happens.

---

## 1. Contrastive Pair Construction — The Root of Most Shortcuts

Before examining the architecture, the pair construction strategy deserves scrutiny, because every other shortcut ultimately flows through it.

### 1.1 Taxonomic Positive Pairs

If positive pairs (structurally similar examples) are constructed from proteins within the same viral species or family, the model is handed a proxy signal: taxonomic identity. Viral proteins within the same family share not only folds but also compositional biases, length distributions, genome-type-specific codon usage patterns, and even signal peptides. The model doesn't need to understand topology — it can learn "same family → similar embedding" using any surface-level feature that co-varies with taxonomy.

**Why this is dangerous here specifically:** The dataset spans 4,463 eukaryotic viral species, meaning inter-species variation is large. If positives are intra-species or intra-family, the model faces easy discrimination: push intra-family proteins together, push inter-family proteins apart. That's a taxonomic classifier, not a structural encoder.

### 1.2 Easy Negatives

If negatives are sampled uniformly at random from the full dataset, the model will mostly see hard positives (structurally similar proteins that look superficially different) against trivially easy negatives (unrelated proteins with obvious differences). Once the model masters easy negatives, loss stagnates — but the representations are coarse. The 62% "dark matter" proteins, being singletons with no known homologues, will almost always appear as negatives. The model learns to push singletons away from everything else, which means it has learned "structurally unusual = far from all clusters" — a clustering of absence, not structure.

### 1.3 The Singleton Majority Problem

12,422 singletons vs. 5,770 multi-member clusters. If singletons are used as anchors or negatives in training, the model sees a strong implicit signal: a protein that never forms a positive pair with anything is structurally "alone." The model may learn this absence as a structural feature, producing embeddings where singletons collapse into a distinct region of latent space not because of their topology but because of their dataset status.

---

## 2. Dataset-Specific Shortcuts

### 2.1 Sequence Length as a Proxy for Fold Class

Protein fold families have characteristic length distributions — TIM barrels, beta-propellers, and coiled-coils occupy different length ranges. If your positive pairs tend to have similar lengths (which they will if they come from the same cluster, since TM-score-based clustering implicitly selects similarly-sized structures), the model can use `|len(A) - len(B)|` as an embedding distance proxy.

This is particularly insidious in TNN-based representations: if you're building simplicial complexes with edges at a fixed distance cutoff (e.g., 8Å Cα contacts), longer proteins will systematically have denser, more complex complexes. The model can use topological complexity — a graph-theoretic property — as a length proxy.

**Diagnostic:** Compute the Spearman correlation between embedding distances and absolute length differences across all pairs. Any correlation above ~0.3 is suspicious.

### 2.2 Amino Acid Composition Bias

Viral proteins have strong phylum-level amino acid composition biases. dsDNA viruses (herpesviruses, poxviruses) show different GC-content-driven compositional profiles than ssRNA viruses. If your structural representation carries any amino acid identity information (sequence-informed graph nodes, one-hot encoded residue types), the model can cluster by composition rather than topology.

This is particularly relevant for the test cases in Phase 4:

- **Test Case 1 (cGAMP/Acb1):** Avian poxvirus proteins (dsDNA) vs. phage proteins — very different compositional profiles. If your model clusters them correctly, make sure it's for structural reasons, not because you inadvertently included cross-viral compositional features in a "structural" node attribute.
- **Test Case 3 (I3L/T7 SSB):** The OB-fold is a very common topology. A composition-based model might get this right for the wrong reasons.

**Diagnostic:** Replace all residue-type node features with a constant (or remove them entirely) and measure clustering performance. If performance drops sharply, composition is carrying the signal.

### 2.3 ColabFold Prediction Artifacts

All 67,715 structures are ColabFold predictions, not experimentally determined. ColabFold (AlphaFold2-based) introduces systematic, learnable biases:

- **pLDDT-correlated artifacts:** Low-confidence regions are modeled inconsistently. Proteins with similar pLDDT profiles get structurally similar predicted structures — not because they share a fold, but because AlphaFold defaults to similar fallback conformations for uncertain regions (often extended loops or disordered alpha-helices).
- **Template contamination:** If ColabFold used sequence-based templates to generate structures, then proteins sharing sequence homologues in the PDB will have structurally similar predictions for reasons that are fundamentally sequence-driven, not topology-driven.
- **Systematic helix bias:** AlphaFold2 tends to over-predict alpha-helical content in ambiguous regions. Your model may learn "more helices = structurally similar" simply because uncertain regions in proteins from the same family get the same helical fallback.

**Diagnostic:** Mask all residues below pLDDT < 70 (or 50) before computing structural representations. If clustering quality changes substantially, you're learning from confident predictions only — which is what you want. If quality improves after masking, the model was previously using low-confidence noise as a signal.

---

## 3. Architecture-Specific Shortcuts

### 3.1 Asymmetric Attention: The Collapse Risk

Asymmetric architectures in contrastive learning (inspired by BYOL, SimSiam, or cross-attention encoders) introduce a fundamental instability: **representation collapse**.

In a symmetric system, the contrastive loss explicitly prevents collapse by pushing dissimilar pairs apart. In an asymmetric system — where one branch (the "online" or query branch) attends to a fixed or slowly-updated "target" branch — the gradient flow is asymmetric. The online branch can converge to a constant vector that minimizes loss against a fixed target, achieving collapse without the loss ever signaling it.

**Signs of collapse in your setting:**
- Embeddings of structurally very different proteins (e.g., a beta-barrel vs. an all-alpha protein) cluster tightly in latent space.
- The variance of embeddings across the dataset is small relative to within-cluster variance.
- Contrastive loss continues decreasing even as downstream clustering quality plateaus or degrades.

### 3.2 Attention Over Sequence Position Rather Than Structural Context

Attention mechanisms natively operate over sequences of tokens. If your residues are ordered by sequence index, self-attention will learn positional co-occurrence patterns — essentially rediscovering secondary structure from sequential proximity — rather than the tertiary contacts that define a "functional fold."

This is a critical point for your Phase 2 objective: *"capture the tertiary interactions that define a functional fold."* Residues i and j that are far apart in sequence but close in 3D space (the signature of beta-sheets, disulfide bonds, domain-domain interfaces) will receive low attention weight if the model learns that sequential distance predicts structural contact. For disordered or repeat proteins, this shortcut is especially likely to dominate.

**Diagnostic:** Check whether your attention maps correlate more strongly with `|i - j|` (sequence distance) than with `||x_i - x_j||` (Euclidean 3D distance). If attention is sequence-position-driven, you haven't escaped the sequence-based methods you're competing against.

### 3.3 The Asymmetric Information Routing Problem

In an asymmetric attention network, the two branches process inputs differently. If one branch (say, the query branch) is deeper or has more parameters, it will learn richer representations. The contrastive objective then becomes: "make the richer branch's output similar to the shallower branch's output." The model can satisfy this by ignoring the structural content in the richer branch and simply learning to match the shallower branch's coarse features.

More concretely: if your asymmetry means one branch processes local structural neighborhoods while the other processes global topology, the model may learn to solve the contrastive task using only the features the shallower/simpler branch can compute — likely secondary structure motifs, loop statistics, or length — because those are the features common to both branches.

### 3.4 The TM-Score Objective Mismatch

You're training with contrastive learning but validating against TM-score ≥ 0.4. TM-score is a specific geometric measure; contrastive learning optimizes a very different objective (margin-based separation in embedding space). The model can achieve low contrastive loss with representations that don't reproduce TM-score ordering, because the mapping from embedding distance to TM-score is not guaranteed to be monotone.

Specifically: TM-score is normalized by the shorter protein's length. Two proteins with very different lengths can have TM-score ≥ 0.4 if the shorter one is entirely contained within the fold of the longer one. A contrastive loss that treats all negatives symmetrically will push such a pair apart — exactly the wrong behavior for length-asymmetric folds.

---

## 4. Training Procedure Shortcuts

### 4.1 Data Leakage Through Cluster Labels

If you construct positive pairs using cluster membership from the paper's 18,192 clusters (or any pre-computed TM-score clustering), and your validation set is sampled from the same clusters, you have label leakage. The model learns to reproduce the paper's clustering rather than discovering structural similarity from first principles.

This is especially subtle: even if your test set proteins are held out, if their *cluster assignments* were used to select training positives, you've implicitly told the model the answer.

### 4.2 Augmentation Asymmetry as a Shortcut

Contrastive learning typically uses augmentations to generate positive pairs. Common structural augmentations include backbone jitter, rotation/translation invariance, or subgraph sampling. If your augmentation strategy is too aggressive (e.g., dropping too many residues), the model learns to be invariant to content — effectively learning representations that ignore structural information. If augmentation is too conservative, positive pairs are nearly identical and the model learns nothing beyond the encoder's inductive bias.

For protein structures from ColabFold, there's an additional problem: the "natural augmentation" of using multiple conformations or temperature-factor perturbations doesn't exist because these are single predicted structures. Your augmentation is therefore artificial, and the model may learn to recognize the augmentation signature rather than the structural content.

### 4.3 Gradient Shortcuts Through the Attention Mechanism

Asymmetric attention introduces an additional gradient flow: the query branch receives gradients both from the contrastive loss and from the cross-attention to the key/value branch. If the key/value branch has a shortcut representation (e.g., length-encoded), the gradients will steer the query branch to also encode length. The asymmetry propagates shortcuts across branches rather than isolating them.

---

## 5. The Benchmark Shortcut

You're benchmarking against Foldseek's 3Di alphabet. Foldseek represents 3D structure as sequences in a 20-letter structural alphabet, essentially compressing 3D context into a 1D sequence. If your training data or validation pipeline involves any 3Di-encoded features, your model may learn to reproduce the 3Di alphabet rather than learning a genuinely richer structural representation. This would inflate benchmark scores without providing any advantage over Foldseek.

---

## 6. Diagnostic Test Battery

The following tests, roughly ordered by ease of implementation, will identify which shortcuts are active:

| Test | What it reveals | How to do it |
|---|---|---|
| **Embedding variance** | Collapse | Compute std of embeddings across all proteins; should be >> intra-cluster std |
| **Length correlation** | Length shortcut | Spearman rank corr. between embedding L2 distance and `\|len_A - len_B\|` |
| **Composition ablation** | Composition shortcut | Zero-out / randomize residue-type node features; measure ARI drop |
| **pLDDT masking** | ColabFold artifact | Re-run with pLDDT < 70 residues masked; compare cluster quality |
| **Attention position analysis** | Sequential vs. structural attention | Correlate attention weights with sequence separation vs. 3D distance |
| **Phylogenetic split** | Taxonomic shortcut | Train on dsDNA viruses only; test on ssRNA viruses; measure performance |
| **Random structure injection** | Content vs. statistics | Inject random contact maps of matching length distributions; measure how often they cluster with real proteins |
| **Singleton region analysis** | Singleton collapse | Check if singletons cluster together in latent space |
| **Loss vs. clustering quality** | Loss-quality decoupling | Plot training loss vs. cluster ARI over time; decoupling signals shortcut |
| **TM-score rank correlation** | Objective mismatch | For held-out pairs, compute Spearman corr. of embedding distance with TM-score |

---

## 7. Mitigation Strategies

If shortcuts are confirmed, the following architectural and procedural interventions are most likely to help, in order of impact:

**Hard negative mining** is the single most important intervention. Replace random negative sampling with mining negatives that are structurally similar at a superficial level (same secondary structure composition, similar length) but topologically distinct. This forces the model to use topology rather than proxies.

**Remove residue-type features** from node attributes if composition contamination is confirmed. Operate purely on backbone geometry (Cα coordinates, backbone dihedral angles φ/ψ/ω) or contact maps.

**Structural augmentation via backbone resampling:** Since you have only one predicted structure per protein, generate synthetic variants by perturbing Cα coordinates within physically plausible bounds (Gaussian noise with σ calibrated to pLDDT uncertainty). This decouples ColabFold artifacts from structural content.

**Phylogenetic train/validation/test splits:** Split not by protein identity but by viral family or genome type. This is the correct way to test whether your model has learned generalizable structural features rather than virome-specific statistics.

**Add a TM-score regression auxiliary loss** alongside contrastive loss. This directly supervises the embedding distance to reproduce TM-score ordering, reducing the mismatch between training objective and evaluation metric.

**Collapse detection during training:** Monitor the standard deviation of batch embeddings and the average pairwise cosine similarity within batches. If std drops or cosine similarity approaches 1.0, stop training and investigate.

---

## 8. Summary Assessment

The most likely active shortcuts, in order of probability:

1. **Taxonomic shortcut via positive pair construction** — almost certain if positives are sampled within species/family
2. **Length proxy shortcut** — high probability given TM-score's implicit length normalization
3. **Easy negative exploitation** — high probability if negatives are random
4. **Singleton collapse** — high probability given the 2:1 singleton-to-cluster ratio
5. **ColabFold helix bias artifact** — moderate probability
6. **Sequential attention instead of structural attention** — moderate probability without explicit position-invariant encoding
7. **Representation collapse** — moderate probability in asymmetric architectures without explicit variance regularization
8. **Composition bias** — moderate probability if residue types are node features

None of these are fatal to the approach. Each has a known mitigation. The goal of this report is to ensure that when your model produces good results, you can verify it's because it learned protein topology — not because it learned something cheaper.

---

*Report written for Matthieu Nardi — Deltafold Internship, Phase 2 analysis.*
*Date: June 2, 2026*
