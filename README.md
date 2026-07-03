# DeltaFold

**Unsupervised structural clustering of the eukaryotic virome.**

Viral proteins diverge in sequence far faster than in shape, so sequence-homology
searches miss real relationships that are still visible at the fold level.
DeltaFold learns — *without labels* — an embedding space where two viral proteins
land close together when they share a fold, so that clustering the embeddings
recovers known protein families and can suggest structural links sequence methods
cannot find.

The encoder is **Topotein / TCPNet**, an SE(3)-equivariant topological neural
network that runs on a *Protein Combinatorial Complex* (residues → contacts →
secondary-structure elements → whole protein). It is trained contrastively
(InfoNCE) on substructure views. The full method is written up in
[`documents/deltafold_protocol.tex`](documents/deltafold_protocol.tex).

---

## 1. Installation

Python 3.10+ is recommended. Create an environment and install the dependencies:

```bash
# conda (the project's env is called ml_env)
conda create -n ml_env python=3.11 -y
conda activate ml_env

# Install the right PyTorch build FIRST (see note below), then the rest:
pip install -r requirements.txt
```

**PyTorch build matters — pick the one for your hardware:**

| Machine | Command |
|---|---|
| **deltafold box** (NVIDIA L40S, CUDA 12.2) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| **macOS** (Apple Silicon / MPS) or CPU | `pip install torch` (default wheel) |

Then verify the model builds and is equivariant:

```bash
python topotein.py          # runs the SE(3)/chirality self-test (should print PASS)
```

---

## 2. Data preparation

The dataset is the predicted structural proteome of the eukaryotic virome
(Nomburg et al. 2024): ~67k predicted structures across ~4.5k viral species.

```bash
# 1. Download the raw predicted PDB structures (~zip into data/hoan_raw_pdb/)
python scripts/utilities/download_dataset.py

# 2. "Lift" each structure into a Protein Combinatorial Complex (.pt files in
#    data/hoan_processed/). Cα trace -> kNN contact graph -> DSSP SSEs -> protein.
python scripts/utilities/topotein_lifter.py --workers 16
#    (add --downsampled for a small prototyping subset; --skip-existing to resume)

# 3. (optional) Build a SMALL prototyping sub-base (~4k proteins) for quick runs:
#    cold Foldseek-cluster split, exact-sequence dedup, size-distribution preserving.
#    Writes data/subbase_corrected_{train,val}.txt. Not needed for a full run.
python scripts/utilities/build_corrected_subbase.py
```

Training defaults to the **full lifted dataset** with a cold cluster split
(`--split cluster`) built from the **Nomburg merged clusters**
(`code_and_intermediate_data/intermediate_data/merged_clusters.tax.tsv`): whole
clusters go entirely to train or val, so no structural cluster spans the split.
Make sure that file is present (it is gitignored — copy it to the box manually).
Cluster/family labels are used **only** to define the split and for evaluation,
never as a training signal.

---

## 3. Training

The entry point is [`train.py`](train.py). The Topotein encoder is
`--model topotein`; training is contrastive by default.

```bash
# full dataset, cold Foldseek-cluster split (the default):
python train.py --model topotein --task contrastive \
    --epochs 50 --unsupervised --temperature 0.2
# ...or --split corrected for a quick ~4k-protein prototyping run.
```

### On the deltafold CUDA box

Add `--deltafold` to switch from MPS to CUDA and apply a high-throughput preset
(pins one GPU, enables TF32, raises the residue budget / batch / workers, and
disables the macOS-oriented RAM governor):

```bash
python train.py --model topotein --deltafold \
    --epochs 50 --unsupervised --temperature 0.2
```

Any knob you pass explicitly still wins over the preset (e.g. `--max-residues`,
`--batch_size`, `--num-workers`). Watch `nvidia-smi` on the first run and raise
`--max-residues` if there is VRAM headroom, or lower it if it OOMs.

### Key flags

| Flag | Meaning |
|---|---|
| `--model {topotein,equivariant,asymmetric}` | Encoder. `topotein` = full TCPNet on the PCC (default here); `equivariant` = lightweight Cα-only geometric variant; `asymmetric` = invariant baseline. |
| `--tensor-diagram` | Message-passing order/channels: `default` (Topotein 4-step), `residue_hub` (no inter-SSE channel — the decisive ablation), `no_rank3`, `reordered`, or a custom `edge,node,sse,protein` order. |
| `--readout {node,protein}` | Graph readout: node pooling (default) or the rank-3 protein cell. |
| `--scalarize {frame,norm}` | `frame` = edge-centric SE(3)+chiral (default); `norm` = O(3), reflection-blind (ablation). |
| `--vector-dim`, `--num-layers` | Vector width `d_v` (default 16) and interaction depth `L` (default 4). |
| `--split {cluster,phylo,corrected}` | Train/val split. **`cluster`** (default) = cold cluster split over the full dataset using the Nomburg merged clusters (`merged_clusters.tax.tsv`); `phylo` = cold taxonomic split; `corrected` = the ~4k subsampled prototyping base (step 2). |
| `--objective {infonce,moco}` | Contrastive objective. `moco` adds a momentum queue for more negatives. |
| `--sub-f-lo/--sub-f-hi`, `--sub-mode` | Substructure positive-pair size fraction and sampling mode. |
| `--deltafold`, `--num-workers` | CUDA hardware preset (above) and DataLoader worker count. |

Run `python train.py --help` for the full list. Checkpoints and the training log
(`training_log.jsonl.gz`) are written to `./checkpoints/` (override with the
`DELTAFOLD_CKPT_DIR` env var).

Health checks (embedding std, mean off-diagonal cosine, effective rank) are logged
every epoch to catch representational collapse — interpret a checkpoint only when
these are healthy.

---

## 4. Embeddings & evaluation

```bash
# Extract normalized embeddings for a trained checkpoint
python scripts/utilities/extract_embeddings.py --model topotein \
    --file-list data/subbase_corrected_train.txt data/subbase_corrected_val.txt \
    --out data/virome_embeddings.pt

# Cluster the embeddings (HDBSCAN) into a partition
python scripts/analysis/cluster_embeddings.py --emb data/virome_embeddings.pt

# Score the embeddings vs the Foldseek reference (homogeneity/completeness,
# fragmentation/fusion, pair FPR/FNR, TM-rho)
python scripts/analysis/epoch_eval.py --emb data/virome_embeddings.pt
```

---

## 5. Repository layout

```
topotein.py                  Full Topotein / TCPNet encoder (the main model)
equivariant_topotein.py      Lightweight Cα-only geometric variant
asymmetric_topotein.py       Invariant (non-geometric) baseline
train.py                     Training entry point (models + tasks + CLI)
train_contrastive.py         Contrastive training loop
substructure.py, moco.py     Substructure positive-pair sampling; MoCo wrapper
contrastive_*.py             Loss/engine, data collation, memory governor, metrics
rbf.py                       Gaussian RBF distance encoding
scripts/utilities/           Data: download, lifting, split, embedding extraction
scripts/analysis/            Clustering + evaluation + sweep summaries
documents/                   Protocol write-up (.tex/.pdf) + figures + references
```

---

## 6. Notes

- **Device selection is automatic**: MPS on Apple Silicon, CUDA if available, else
  CPU. `--deltafold` forces CUDA and pins a single GPU.
- The Topotein architecture, featurization, message-passing scheme and the
  experimental sweep are specified in
  [`documents/deltafold_protocol.tex`](documents/deltafold_protocol.tex);
  it is the authoritative reference for the design.
