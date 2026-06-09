# Action Plan: Phases 3–5 Prototype on Downsampled Dataset
*For the coding agent — June 2026*

---

## Context and Current State

The contrastive model has been trained for 14 epochs on the downsampled dataset (~3,600 proteins). Validation shows Spearman ρ = −0.74 (p < 10⁻¹⁸) between cosine distance and TM-score, with strictly monotonic mean distances across TM bands (TM > 0.7 → dist ≈ 0.26; TM 0.4–0.7 → dist ≈ 0.50; TM 0.2–0.4 → dist ≈ 0.91). The model encodes a genuine continuous structural metric.

Available artifacts:
- `virome_embeddings.pt` — 3,647 protein embeddings, 128-dimensional, L2-normalised
- `checkpoints/tm_score_cache.pt` — pre-computed pairwise TM-scores for within-cluster pairs
- `checkpoints/checkpoint_contrastive_asymmetric_best.pth` — best model checkpoint
- `cluster.tsv` — Foldseek cluster assignments (used as training labels)
- Processed PCC files in `data/hoan_processed/` with per-protein metadata (taxid, accession, genome type)

The objective for this phase is to build an end-to-end downstream analysis pipeline on the current small dataset. Correctness and biological interpretability are the goals, not scale. No new training in this phase.

---

## Phase 3 — Unsupervised Clustering of the Embedding Space

### Goal
Cluster the 3,647 embeddings using the learned cosine distance metric, without using the Foldseek cluster labels as input. Evaluate whether the learned clusters recover biologically meaningful groupings.

### Tasks

**3.1 — Implement clustering pipeline (`cluster_embeddings.py`)**

Cluster the embeddings using HDBSCAN (preferred: does not require specifying K, handles singletons natively) and optionally agglomerative clustering with a distance threshold as a cross-check. The cosine distance threshold for "structurally similar" is approximately 0.40–0.45 based on the TM band analysis. Clustering should run on the full 3,647-protein embedding set.

Output: a TSV file with columns `(protein_id, cluster_id)` where singletons get a unique negative cluster id. Log: total clusters found, number of singletons, number of multi-member clusters, size distribution.

**3.2 — Compare against Foldseek clusters**

Compute ARI and NMI between the model's clusters and the Foldseek `cluster.tsv` assignments, restricted to multi-member clusters in both. This is a sanity check: the model should partially recover Foldseek clusters (ARI ~0.40 as seen during training) but may also split or merge them based on the continuous metric rather than the binary threshold.

Log cases where the model merges two Foldseek clusters (potential discovery: Foldseek may have over-split a structural family) and cases where the model splits a Foldseek cluster (potential discovery: a Foldseek cluster may contain structurally distinct sub-families).

**3.3 — Singleton ratio check**

Report the fraction of proteins assigned as singletons by the model. The reference paper reports ~2/3 of viral proteins are structural "dark matter" with no detectable homologue. If the model's singleton fraction is far below this, it may be over-merging; far above, it may be under-sensitive. This is not a failure criterion — it is a calibration check for the distance threshold.

---

## Phase 4 — Biological Validation ("Easter Eggs")

### Goal
Verify that the embedding space recovers specific known structural homologies, including cases where sequence similarity is undetectable. These are the biological test cases from the original paper.

### Tasks

**4.0 — Inventory check (do this first)**

Before running any analysis, check which of the following proteins are present in the downsampled dataset by accession. If a protein is absent, identify the nearest substitute in the dataset (same cluster or known structural homologue):

- **cGAMP/Acb1 case:** Viral proteins homologous to the cellular cGAMP-binding domain (anti-viral immunity evasion). Accession prefix: check for avian poxvirus proteins annotated as `Acb1` or related.
- **ENT4 mimicry case:** Viral proteins structurally similar to human Equilibrative Nucleoside Transporter 4. Genome type: likely dsDNA large virus.
- **Poxvirus I3L / T7 SSB case:** OB-fold single-stranded DNA binding proteins from Poxviridae and T7 phage — classic case of structural conservation without sequence similarity.

If none of these are in the downsampled set, identify 2–3 pairs of proteins that are in different Foldseek clusters but belong to the same known structural superfamily (SCOP or ECOD level). These become the substitute validation cases.

**4.1 — Pairwise distance lookup for test cases**

For each confirmed test case pair (A, B), report:
- Cosine distance in the learned embedding space
- Foldseek cluster assignment (same or different cluster)
- TM-score from cache if available, or note that alignment is needed
- Whether the model places them within the "close" band (distance < 0.45) or not

A successful result: the model places structurally homologous proteins close in embedding space (distance < 0.45) even when they are in different Foldseek clusters.

**4.2 — Nearest-neighbor retrieval test**

For each test protein, retrieve its 10 nearest neighbors in embedding space (by cosine distance). For each neighbor, report: accession, Foldseek cluster, genome type, and available annotation. This mimics the practical use case of "I have an unknown viral protein — what does the model think it is structurally similar to?" and constitutes the most biologically interpretable result in the internship report.

---

## Phase 5 — Visualization and Annotation Mapping

### Goal
Produce a 2D visualization of the embedding space annotated with biological metadata. This is the primary figure for the internship report and the main tool for exploring what the model has learned.

### Tasks

**5.1 — UMAP projection**

Project the 128D embeddings to 2D using UMAP with cosine metric (consistent with the training objective). Run UMAP twice with different `n_neighbors` values (15 and 50) to show both local and global structure. Save both projections as TSV files with columns `(protein_id, umap_x, umap_y)`.

**5.2 — Annotate with biological metadata**

For each protein in the UMAP, attach from the processed PCC metadata:
- Genome type (dsDNA, ssDNA, dsRNA, ssRNA+, ssRNA−)
- Viral family (from taxid lineage, top-level family)
- Foldseek cluster membership (singleton vs. multi-member, cluster id)
- Model cluster membership from Phase 3

**5.3 — Generate primary visualization**

Produce an HTML scatter plot (using Plotly or equivalent) of the UMAP projection with:
- Color by genome type (primary figure: tests whether structural families cross genome-type boundaries)
- Hover tooltip showing protein_id, viral family, cluster id, genome type

Produce a second version colored by Foldseek cluster id (restricted to multi-member clusters; singletons in gray). This shows whether the learned metric space respects the Foldseek boundaries.

**5.4 — Cross-genome-type bridge clusters**

Identify clusters (from Phase 3) that contain proteins from two or more different genome types. These are the most scientifically interesting result: structural conservation across genome types indicates a fold that predates the divergence of these viral lineages, or convergent structural evolution. For each such cluster, log: cluster id, member count, genome types represented, accessions.

**5.5 — Export cluster annotation table**

Produce a TSV `cluster_annotations.tsv` with one row per protein and columns: `protein_id, model_cluster_id, foldseek_cluster_id, genome_type, viral_family, taxid, umap_x, umap_y`. This is the master output table for the internship report.

---

## Decision Gate: When to Scale to Full Dataset

**Criteria to proceed to full 67k training:**
1. Phase 3 clustering pipeline runs end-to-end without errors and produces a biologically plausible singleton ratio.
2. Phase 4 retrieves at least 2 of the 3 target test cases correctly (or equivalent substitute cases), confirming the model detects cross-sequence structural homology.
3. Phase 5 UMAP shows visible structure (non-random clustering) and at least one cross-genome-type bridge cluster.
4. The entire Phase 3–5 pipeline runs in under 5 minutes on the downsampled set (confirming it will be practical on 67k proteins).

If all four criteria are met, launch full training. While training runs (estimated several days), the Phase 3–5 scripts require no modification — they take `virome_embeddings.pt` as input, which will simply be larger.

**Do not scale yet if:** Phase 4 test cases fail entirely. This indicates an architecture or threshold calibration problem that would be amplified, not solved, by more data.

---

## File Output Summary

| Script | Primary output |
|---|---|
| `cluster_embeddings.py` | `clusters/model_clusters.tsv` |
| `compare_clusters.py` | Terminal report (ARI, NMI, merge/split log) |
| `validate_test_cases.py` | Terminal report (distances, neighbor lists) |
| `project_umap.py` | `clusters/umap_coords.tsv` |
| `visualize_embedding.py` | `clusters/embedding_viz.html` |
| `annotate_clusters.py` | `clusters/cluster_annotations.tsv` |

All outputs go in a new `clusters/` subdirectory.
