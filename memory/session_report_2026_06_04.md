# Session Report — 2026-06-04

## What this session was about

The full-dataset (67k protein) training run was underway, and we spent the session building the infrastructure to make it scale safely and produce interpretable signals each epoch: extraction optimisations, pipeline scaling, an automated per-epoch evaluator, a memory governor that can't let the process swap, and a consolidated log format.

---

## 1. Embedding Extraction — `extract_embeddings.py`

**Problem:** the old script used a fixed `batch_size=16` loader with `num_workers=0`, no padding, and no per-step cleanup — none of the three MPS memory mitigations from the training loop.

**What changed:**

- **`LengthBudgetSampler`** — new residue-budgeted batch sampler that sorts proteins by length and packs each batch under a configurable cap. Reuses `batch_keys_cache.pt` (built by the training loop). Extraction is forward-only (no backward, no augmentation views), so `--max-residues` defaults to **8000** vs training's 4000.
- **`pad_to_buckets`** — imported from `train_contrastive`; bounds the set of distinct MPS kernel shapes. Verified output-preserving to 1.8×10⁻⁷ on the real asymmetric checkpoint.
- **Throttled `gc + empty_cache`** every `--cleanup-every` (default 50) steps; immediate `.cpu()` of each embedding after extraction so MPS tensors don't accumulate in the dict.
- **Parallel loading** — `num_workers`, `prefetch_factor=1`, `worker_init_fn`.
- New CLI flags: `--max-residues`, `--cleanup-every`, `--no-pad-buckets`. `--batch_size` is now the count cap (default 32).

---

## 2. Phase 3–5 Pipeline Scaling to 67k

**Problem:** the pipeline was designed for the 3,647-protein downsampled set. At 67k proteins, a dense N×N cosine distance matrix requires ~37 GB — completely infeasible on 16 GB.

**Audit result:** only two scripts touched that matrix. The other four (`compare_clusters`, `project_umap`, `annotate_clusters`, `visualize_embedding`) were already O(N).

**`cluster_embeddings.py`** — dual-path on `--max-dense` (default 15,000):
- N ≤ max-dense: the exact original precomputed-cosine HDBSCAN + agglomerative cross-check (bit-for-bit identical to the downsampled run).
- N > max-dense: HDBSCAN on the **euclidean** metric over L2-normalised vectors via space-trees (O(N) memory). On L2-normalised embeddings euclidean is monotonic to cosine so the MST/hierarchy is identical, but HDBSCAN's EOM stability extraction (which integrates 1/distance) shifts slightly — verified ARI ~0.96 vs the dense cosine path, NOT bit-identical. All space-tree algorithms (`boruvka_kdtree`, `boruvka_balltree`, `prims_kdtree`) give the exact same result as `generic`, confirming the difference is the metric transform, not approximation. Agglomerative skipped above `max-dense`. Full 67k run: **522 s, 686 MB peak, 0 swaps**.

**`validate_test_cases.py`** — `nn_retrieval` now computes only the single query distance row on-the-fly (`1 - X @ X[qi]`) instead of the full matrix. Full 67k run: 2 s, 691 MB.

---

## 3. Full-67k Results Analysis (Epoch 4 Checkpoint)

**Key finding:** the embeddings extracted were from **epoch 4 of the full-dataset run**, with val ARI 0.159 — less than half-converged. The converged downsampled model reached val ARI ~0.40–0.44 over 30–45 epochs.

| Signal | Full 67k (epoch 4) | Downsampled (converged) | Read |
|---|---|---|---|
| Val ARI (training) | 0.16 | ~0.40 | Undertrained, but ≫ chance |
| ARI vs Foldseek | +0.053 | +0.27 | Heavy over-segmentation |
| NMI vs Foldseek | 0.83 | 0.93 | High NMI + low ARI = over-splitting |
| Genome-type purity | 0.97 (baseline 0.73) | — | Real coherent structure |
| I3L↔SSB homology | ✓ cosine 0.099 | ✓ 0.24 | Cross-fold case recovered |
| TM=0.97 pair | cosine 0.007 | — | Highest-confidence pair recovered |
| TM 0.7–0.8 pairs | missed (cosine 0.68–0.76) | — | Not yet in range |
| UMAP within/global | 0.029 | 0.054 | Tighter than converged (driven by min_size=2) |

**Verdict:** the model has demonstrably learned genome-type coherence and the very highest-TM homologies, but is over-segmenting (3,309 model clusters merge ≥2 Foldseek clusters; 2,170 Foldseek clusters shattered into hundreds of model pieces). Many of those "splits" are legitimately breaking up junk Foldseek clusters (e.g. "hypothetical_protein_2" spanning 1,388 proteins across 25+ families), so ARI vs Foldseek understates quality somewhat. Need to let training converge.

---

## 4. Per-Epoch Evaluator — `epoch_eval.py`

**New file.** Runs every epoch on the val embeddings already computed by the training loop (no re-extraction overhead). Logs to `checkpoints/training_log.jsonl.gz` (see §6).

Metrics per epoch:
- `hdbscan_ari`, `hdbscan_nmi` — HDBSCAN partition of val embeddings vs Foldseek labels, mutual-multimember restricted (same methodology as `compare_clusters.py`).
- `n_clusters`, `singleton_frac` — structural health of the partition.
- `tm_rho` — Spearman ρ(cosine-distance, TM-score) over cached pairs present in val set. Strongly negative = continuous structural metric. Target: ρ < −0.5.
- `tm_recall` — fraction of high-TM (>0.5) cached pairs placed in the close cosine band (<0.45). Target: >0.7.

Also has a standalone deep-check mode: `python epoch_eval.py --emb data/virome_embeddings.pt` runs on the full extracted set (HDBSCAN on 67k ≈ 8 min — too heavy every epoch, hence the val-subset in-loop version).

**Caveat:** HDBSCAN-ARI is scale-sensitive. On a 4k subsample it returns ~0.40; on all 67k it returns ~0.05. Read the in-loop number as a trend signal, not an absolute value comparable to the full-pipeline run.

---

## 5. Memory Governor — `MemoryGovernor` in `train_contrastive.py`

**Problem:** training leaked to ~27 GB (Activity Monitor) by epoch 5 — DataLoader worker buffers + fragmented MPS allocator pool. The old RSS-based monitoring was invisible to it (RSS showed ~5 GB while footprint was 27 GB).

**Architecture — three-tier escalation per step:**

1. **Throttled baseline** — `gc + empty_cache` every `cleanup_every` steps (unchanged from before).
2. **Soft cap** (`--mem-soft-gb`, default 11 GB) — off-schedule reclaim without interrupting the batch. Also fires in the validation loop (which has no workers to restart).
3. **Hard cap** (`--mem-hard-gb`, default 14 GB) — after a reclaim attempt still exceeds the cap: **cold-restart the DataLoader workers** (tear down the loader, rebuild it, workers release their pooled memory — always cheaper than swap) **and shrink the residue budget** ×0.85 (floor `--min-residues`, default 2000). Budget recovers ×1.1 per clean epoch.

**Restartable batch generator:** the epoch loop is restructured around a `_restartable_batches()` generator. The loader is built by a `make_train_loader` factory; a restart re-seeds the sampler with `epoch*1000 + restart_pass` (fresh batches each pass) while the `done` step counter persists across restarts so the epoch still completes exactly `total_steps` gradient steps.

**Verified:** with artificially low caps (0.5 GB hard), 5 cold restarts fired, budget auto-shrank 3000→1329, epoch completed cleanly.

New CLI flags: `--mem-soft-gb`, `--mem-hard-gb`, `--min-residues`. `--max-residues` is now the *starting* budget the governor adapts from.

---

## 6. Memory Metric — `phys_footprint` replaces `psutil.RSS`

**Root cause discovery:** `psutil.RSS` on Apple Silicon does not count MPS (GPU) allocations — the MPS driver manages its own pool outside the process's traditional resident set.

Empirical proof:

| State | `psutil.RSS` | `phys_footprint` | Delta |
|---|---|---|---|
| Fresh process | 0.021 GB | 0.010 GB | — |
| After model load + forward | **0.341 GB** | **1.481 GB** | **+1.14 GB (MPS)** |

`phys_footprint` is read via the Mach `task_info(TASK_VM_INFO)` API using ctypes. The struct field is at byte offset 144 in `task_vm_info_data_t` (`virtual_size`(8) + `region_count`(4) + `page_size`(4) + 17×`mach_vm_size_t`(8) = 144 bytes). This is exactly what Activity Monitor's "Memory" column displays, and the value that was showing 27 GB during the leak. Falls back to `psutil.RSS` on non-macOS. The pbar postfix label changed from `ram=` to `fp=`.

---

## 7. Consolidated Log — `training_log.jsonl.gz`

**Gone:** `contrastive_losses.csv` (~7,000 rows/epoch), `collapse_metrics.csv` (~700 rows/epoch), `ari_log.csv` (1 row/epoch), `epoch_eval.csv` (1 row/epoch).

**Replaced by:** `checkpoints/training_log.jsonl.gz` — one compact JSON line per epoch plus sparse events only when they actually occur (worker restart, collapse threshold breach). Example:

```json
{"t":"epoch","ep":5,"steps":789,"loss":1.23,"vloss":0.41,"ari":0.31,"fp":9.2,"restarts":0,"budget":4000,"loss_range":[0.8,2.1],"eval":{"hdbscan_ari":0.39,"nmi":0.88,"tm_rho":-0.79,"tm_recall":0.87,"n_clusters":845,"singleton_frac":0.36}}
{"t":"restart","ep":5,"s":234,"fp":14.3,"budget":3400}
```

Size: 422 bytes for 2 epochs (~5 KB projected for 50 epochs). Read with:

```bash
python -c "import gzip,json; [print(json.loads(l)) for l in gzip.open('checkpoints/training_log.jsonl.gz')]"
```

---

## What to watch as training converges

From `training_log.jsonl.gz` each epoch, the trajectory that signals a healthy run:

| Field | Now (ep 4) | Target |
|---|---|---|
| `ari` (kmeans val) | 0.16 | 0.40+ |
| `eval.hdbscan_ari` | ~0.05 (67k) | trend ↑ |
| `eval.tm_rho` | −0.81 (subsample) | < −0.5, stable |
| `eval.tm_recall` | 0.93 (subsample) | > 0.7, stable |
| `fp` | ~9 GB | ≤ 14 GB (hard cap) |
| `restarts` | 0 | 0 most epochs |

The kmeans val-ARI (`ari`) climbing toward 0.40 and `eval.tm_rho` going more negative are the two clearest convergence signals. If `restarts > 0` on an epoch, memory pressure was real — the governor fired to prevent swap.
