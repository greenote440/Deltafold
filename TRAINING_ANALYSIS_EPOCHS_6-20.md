# Full Training Analysis: Epochs 6-20

**Date**: 2026-06-12  
**Dataset**: 67k proteins, full scale  
**Hardware**: 16GB Apple Silicon (MPS)  
**Configuration**: `--mem-hard-gb 12.0 --max-residues 4000` (initial)

---

## Executive Summary

**The training is in critical distress.** The system is manifesting a **cascading failure pattern** across three independent systems that reinforce each other:

1. **Uncontrolled memory creep** (10.2GB → 24.9GB, +145%)
2. **Chronic embedding collapse** (44 collapse signals in 15 epochs)
3. **Memory governor thrashing** (6 restarts/epoch, continuous budget shrinking)

The model is NOT learning meaningfully—metrics are flat despite 59% ARI improvement. This is window dressing; the system is sick.

---

## Part 1: Memory Footprint Crisis

### The Creep Pattern

```
Epoch  6: 10.2GB  (baseline)
Epoch 12:  5.1GB  (momentary dip after big restart sequence)
Epoch 13:  9.7GB  (immediately rebounded)
Epoch 15: 14.0GB  (crossed 12GB hard cap)
Epoch 20: 24.9GB  (peak, +145% from baseline)
```

**Key observation**: Memory stabilizes at ~5GB for only 1 epoch (12), then explodes. This is NOT a one-time leak being patched—it's **structural**.

### Memory Velocity (GB/epoch growth)

| Transition | Delta | % Growth |
|-----------|-------|----------|
| 6→7 | -0.06 | -0.6% |
| 7→8 | +0.18 | +1.8% |
| 8→9 | -1.44 | -14.0% |
| 12→13 | **+4.64** | **+91.7%** |
| 13→14 | +2.21 | +22.8% |
| 14→15 | +2.13 | +17.9% |
| 15→16 | +2.17 | +15.5% |
| 16→20 | +2.17/ep | +10-13%/ep (steady) |

**From epoch 13 onward, memory leaks ~2GB per epoch.**

### Root Cause: The Residue Budget Collapse

Initial configuration:
- `max_residues=4000` (baseline budget)
- MemoryGovernor shrink factor: `0.85x`

Timeline:
- **Epoch 6**: No restarts, budget held at 4000 ✓
- **Epoch 7**: 5 restarts → budget shrunk to 2000 (50% reduction)
- **Epoch 8**: 5 restarts → budget stuck at 2000
- **Epoch 9–12**: Restarts increase to 6/ep → budget grows to 3000–3500 (grow factor `1.1x` each clean epoch)
- **Epoch 13+**: Budget wedged at 3500, but memory **keeps rising**

**The fundamental issue**: Once budget shrinks below a threshold (~3500 residues), the MemoryGovernor cannot shrink further without hitting the `min_residues=2000` floor. The governor's mitigation—shrink-and-restart—is **exhausted**. Memory continues to leak, restarts continue (futilely), and footprint climbs.

---

## Part 2: Embedding Collapse — Persistent & Accelerating

### Collapse Signal Definitions

**Collapse detection** (line 1062): `mean_cos > 0.9 OR emb_std < 0.02`

- `emb_std`: standard deviation of normalized embedding vector across dimension. Low → all vectors point the same direction.
- `mean_cos`: average off-diagonal cosine similarity. High → all pairs are similar.

### Collapse Prevalence Over Time

```
Epoch  6:  3 signals | std: 0.0179 avg | cos: 0.9467 avg
Epoch  7:  6 signals | std: 0.0159 avg | cos: 0.9519 avg
Epoch  8:  4 signals | std: 0.0172 avg | cos: 0.9492 avg
Epoch  9:  1 signal  | (post-restart calm)
Epoch 10:  3 signals | std: 0.0145 avg | cos: 0.9629 avg ← WORSE
Epoch 11:  0 signals | (brief respite)
Epoch 12:  0 signals | (budget bottleneck slowed training)
Epoch 13:  1 signal  | (recovery partial)
Epoch 14:  1 signal  | std: 0.0121 avg | cos: 0.9766 avg ← EXTREME COLLAPSE
Epoch 15:  1 signal  | std: 0.0127 avg | cos: 0.9715 avg
Epoch 16:  2 signals | std: 0.0141 avg | cos: 0.9656 avg
Epoch 17:  3 signals | std: 0.0182 avg | cos: 0.9425 avg
Epoch 18:  7 signals | std: 0.0182 avg | cos: 0.9371 avg ← MAJOR SPIKE
Epoch 19:  4 signals | std: 0.0195 avg | cos: 0.9374 avg
Epoch 20:  8 signals | std: 0.0180 avg | cos: 0.9472 avg ← MOST COLLAPSE SIGNALS
```

### Interpretation

1. **Collapse is systemic & persistent**. Even "good" epochs have near-threshold signals (`emb_std` oscillating between 0.012–0.020).

2. **Collapse accelerates in the latter half** (18–20 epochs have 7–8 signals each, vs 1–3 earlier). The model is **regressing**.

3. **The threshold is hair-trigger**: `mean_cos > 0.9` fires constantly. For a 128-dim embedding, `cos > 0.9` on average means vectors are **highly aligned** but not necessarily degenerate. However, combined with `emb_std < 0.02` (1.5–2x lower than healthy), this is genuine collapse.

4. **Worst single collapse**: Epoch 14 step 620 has `emb_std=0.0121, cos=0.9766`. That's **extreme**—embeddings are nearly parallel, losing ~98% of their variance.

---

## Part 3: Loss Instability — Growing Variance

### Loss Spread Analysis (indicator of numerical instability)

Healthy training: low spread, low CV (coefficient of variation).  
This training: **exploding spread**, especially late.

```
Epoch  6: mean=0.5257, spread=5.0578, CV=9.6
Epoch  7: mean=0.4248, spread=3.6072, CV=8.5
...
Epoch 18: mean=0.4416, spread=4.2169, CV=9.5
Epoch 19: mean=0.4141, spread=3.9731, CV=9.6
Epoch 20: mean=0.4258, spread=5.6765, CV=13.3 ← WORST
```

**CV exploded from 8–9 early to 13+ late.**

Note: `loss_range=[0.0, 5.67]` contains a 0.0 value (numerical underflow? NaN handling?). This is suspicious—the log records both min and max, and if min is exactly 0.0 on some batches, that suggests **loss clipping or pathological numerics**.

---

## Part 4: Training Metrics — Flat & Noisy

### ARI Trend (K-means clustering)

```
Epoch  6: 0.1737
Epoch 20: 0.2302
Growth: +32.5%
```

**Modest improvement, but within noise.**

### HDBSCAN ARI (deeper clustering health)

```
Epoch  6: 0.2501
Epoch 20: 0.3981
Growth: +59.2%
```

**Larger improvement, BUT**: HDBSCAN forms clusters greedily (no fixed k). With high embedding collapse, fewer distinct directions means HDBSCAN clusters will merge—artifactually boosting ARI if the cluster boundaries are artifacts. **This metric is not reliable under collapse.**

### TM-Rho Trend (homology correlation)

```
Epoch  6: -0.7943
Epoch 20: -0.7767
Change: +0.0176 (≈2.2% improvement)
```

**Essentially flat.** TM-rho measures whether structurally similar proteins co-cluster. If embeddings are collapsing, TM-rho should crash (no signal). It's not crashing—it's just static. This means:

- **Either** the model learned a good fold-structure signal early and has plateaued.
- **Or** the TM-score evaluation is insensitive to collapse (e.g., all proteins scoring equally on a degenerate metric).

The recall@close metric (0.836–0.909) is also flat, suggesting no improvement in hard homology detection.

---

## Part 5: Restart Escalation — Futility

### Restart Count Over Time

```
Epoch  6: 0 restarts (happy state)
Epoch  7: 5 restarts → budget 4000→2000
Epoch  8: 5 restarts → budget stuck at 2000
Epoch  9: 6 restarts → budget 2000→3000 (grow phase)
Epoch 10: 6 restarts → budget 3000→3500
Epoch 11: 6 restarts → budget 3500→3500 (stalled)
Epoch 12: 6 restarts → budget 3500→3500 (stalled)
...
Epoch 20: 6 restarts → budget 3500→3500 (stalled for 10 epochs)
```

**From epoch 11 onward, restarts are *constant* and *useless*.** The budget can't shrink further (floor is 2000, we're at 3500). The memory doesn't recover. Restarts are just **noise and overhead**.

---

## Part 6: Correlation Analysis

### Do Restarts Reduce Collapse?

```
Epochs with restarts (7–20):
  - Restarts: 6/epoch (constant)
  - Collapse signals: 1–8/epoch (variable, trending UP)
  
Epoch 6 (no restarts): 3 collapse signals
Epoch 7 (5 restarts): 6 collapse signals (WORSE!)
Epoch 9 (6 restarts): 1 collapse signal (brief respite)
Epoch 11–12 (6 restarts, budget flat): 0–1 collapse signals
Epoch 18–20 (6 restarts, budget flat, memory high): 7–8 collapse signals (WORST)
```

**Restarts do NOT correlate with reduced collapse.** In fact, epochs 18–20 have:
- Maximum restarts (6 each)
- Maximum memory (23–25GB)
- Maximum collapse signals (7–8 each)

This is a **negative correlation**: as restarts become futile (budget exhausted), collapse accelerates.

### Causality Hypothesis

The memory creep from epochs 13–20 happens even with 6 restarts/epoch. This suggests **the leak is not in DataLoader worker buffers** (which are flushed by restarts) but in:

1. **MPS allocator fragmentation**: Each backward pass allocates intermediate tensors. Even with `torch.mps.empty_cache()`, the allocator itself is fragmented, so each subsequent allocation takes more room.

2. **Activation cache bloat**: The model is a GNN; each forward+backward accumulates intermediate edge/node features. With residues capped at 3500, the budget is tight, and the model's computation graph grows large in memory.

3. **Gradient accumulation or ReduceLROnPlateau scheduler state**: The scheduler is tracking loss history; the optimizer has momentum buffers. These are not freed at epoch boundaries.

---

## Part 7: The Collapse Mechanism

### Loss Numerics

Typical loss: 0.42–0.44  
Loss spread: 3.6–5.7 (CV=8–13)

Looking at the raw loss recording code (train_contrastive.py, line 1089–1093):
```python
lv = loss.item()
epoch_loss += lv
step_losses.append(lv)
```

The min/max are recorded separately and clipped to `loss_range`. The presence of 0.0 values in many epochs is suspicious. This could indicate:

1. **Numerical underflow**: NTXentLoss with small logits and `temperature=0.1` can produce `exp(-200) ≈ 0` in softmax, leading to `log(0)` → `-inf` → clipping to 0.0.

2. **Degenerate batches**: If a batch has all augmented pairs from the same cluster and embeddings are nearly parallel, the denominator of the contrastive loss can explode, driving loss to exactly 0.0 (model has "won" trivially).

### Embedding Collapse → Loss Signal Collapse

In NTXentLoss (contrastive_engine.py):
```python
logits = (z[:B] @ z[B:].T) / temperature
```

With `temperature=0.1` (very low), if embeddings collapse:
- All dot products ≈ 1.0 (parallel vectors)
- `logits ≈ 1.0 / 0.1 = 10.0` for all pairs
- Softmax of `[10, 10, 10, ...]` ≈ uniform → no gradient signal
- Loss plateaus (no distinctive positives)

**Then, with gradient clipping** (clip_grad_value_=1.0 for MPS):
- Gradients are clamped to `[-1, 1]`
- Small gradients from degenerate loss are unaffected
- No strong signal to recover embeddings

**And with a low learning rate** (lr=1e-4):
- Small updates to embeddings
- Collapsed state persists

**Result**: Once collapse happens, the loss function **cannot pull embeddings apart**. The system is stuck.

---

## Part 8: Why This Happened (Root Cause Analysis)

### Chain of Events

1. **Epoch 6**: Healthy. Budget=4000, memory=10.2GB, 0 restarts, good throughput.

2. **Epoch 7**: First stress. Memory pressure → 5 restarts, budget shrunk to 2000 (50% cut).
   - Smaller batches = less diverse contrastive pairs
   - Higher variance per step
   - Collapse starts appearing (6 signals)

3. **Epoch 8**: Continued stress. Budget still 2000, more restarts, memory leaks, collapse persists.

4. **Epoch 9**: Budget recovered to 3000 (grow phase), but 6 restarts still occurring.
   - Residue budget is still 25% below baseline
   - Memory pressure continues

5. **Epoch 10–12**: Budget slowly climbs to 3500. Memory stabilizes briefly at epoch 12 (5.1GB) after a large restart sequence, but **then explodes**.

6. **Epoch 13**: Budget at 3500, memory jumps to 9.7GB (+91% from epoch 12). This is the **critical turning point**.
   - The MemoryGovernor is exhausted. Restarts no longer free memory effectively.
   - Residue budget is stuck at 3500 (can't shrink further without hitting min_residues=2000 floor).
   - Memory begins **structural accumulation**: ~2GB per epoch.

7. **Epoch 14–20**: Embedded memory climb. 6 restarts per epoch are futile. Collapse accelerates as:
   - Batches are smaller → less contrastive diversity
   - Memory pressure causes frequent GC pauses → slower training
   - Loss signal weakens → embeddings don't recover from collapse
   - Collapse persists → loss signal stays weak (positive feedback loop)

### The Original Configuration Was Miscalibrated

```
--mem-hard-gb 12.0
--max-residues 4000
--min-residues 2000
```

This configuration assumes:
- **Typical batch size**: ~8–16 proteins
- **Typical residues/protein**: ~300–400
- **In-flight memory for forward+backward**: 4000 residues * 2 views * 128 dims * 4 bytes ≈ 4MB (embeddings) + 100MB+ (intermediates) = ~150MB per batch

But on the 67k full dataset:
- **Large proteins exist**: Some proteins have 5000+ residues; bucketing to nearest 1024-residue boundary inflates them further.
- **Augmentation**: Each protein is loaded twice (two augmented views), doubling memory pressure.
- **Worker prefetch**: DataLoader prefetches 2 batches ahead by default.
- **MPS allocator fragmentation**: Each step allocates intermediate tensors; fragmentation bleeds into the next epoch.

**Result**: Actual peak memory during a forward pass ≈ 5–7GB (data + model + intermediates + MPS cache). With two workers prefetching and training loop state, we hit 10–12GB quickly. Once past 12GB, the governor kicks in and shrinks the budget, which immediately causes **under-utilization** of the GPU (smaller batches → less work per step) while memory still leaks (allocation patterns don't reset).

---

## Part 9: Why Metrics Still Improve (Apparent Paradox)

Despite collapse and chaos, HDBSCAN ARI improved from 0.2501 → 0.3981 (+59%). This seems contradictory. Explanations:

1. **HDBSCAN is flexible**: It forms clusters based on density. Even if embeddings are collapsed, as long as there's *any* variation, it can find clusters. With collapsed embeddings all pointing ~45° apart, HDBSCAN may find many small clusters, raising ARI if the true labels happen to split that way.

2. **Early epochs set a baseline**: Epoch 6's ARI=0.2501 might be sub-optimal. Moving from 0.25 → 0.40 could just be the model learning *something* about protein structure, even if the latter epochs are broken.

3. **TM-rho is stable**: TM-rho=-0.796 (epochs 6–20) is consistent. This metric is based on structural similarity, not embedding geometry. If embeddings collapse but preserve some structural signal (e.g., hard negatives are always dissimilar in structure), TM-rho won't crash.

4. **Metric insensitivity to collapse**: ARI and TM-rho measure clustering quality, not embedding quality. They don't penalize collapse directly. The model can have degenerate embeddings (cos=0.97) and still cluster well if the clusters are small and the label distribution is favorable.

**Bottom line**: Metrics are misleading. They rose because the model learned *something* in the first few epochs. Late-epoch metrics are artifacts of a broken system.

---

## Part 10: Diagnosis Summary Table

| System | Status | Evidence | Severity |
|--------|--------|----------|----------|
| **Memory** | 🔴 CRITICAL FAILURE | +145% creep, 2GB/epoch leak | BLOCKING |
| **Embedding Collapse** | 🔴 CHRONIC & WORSENING | 44 signals in 15 epochs, acceleration late | HIGH |
| **Loss Numerics** | 🟠 UNSTABLE | CV=13, suspicious 0.0 min values | MEDIUM |
| **Training Velocity** | 🟠 DEGRADED | Restarts 6/epoch (futile), throughput variable | MEDIUM |
| **Metrics** | 🟡 MISLEADING | ARI rising but collapse accelerating | LOW (false positive) |

---

## Recommended Fixes (Priority Order)

### 1. **Immediate: Increase Memory Cap**
```bash
--mem-hard-gb 16.0
```
Rationale: The current 12.0GB cap is too tight for the 67k dataset on 16GB hardware. Let memory grow to the actual physical limit (16GB). The restarts are futile below 3500 residues; you're just causing churn.

**Expected result**: Budget stabilizes at 4000 (or higher if MPS needs it), restarts drop to ~0–1/epoch, memory creep slows.

### 2. **Short-term: Reduce Batch Diversity Pressure (Contrastive Loss)**
```bash
--temperature 0.2  # (up from 0.1)
```
Rationale: With `temperature=0.1`, logits blow up, causing numerical underflow and loss collapse. A higher temperature spreads the distribution, giving gradients more room to maneuver.

**Expected result**: Loss range shrinks, collapse signals drop, gradient flow improves.

### 3. **Short-term: Disable Hard-Negative Mining (if enabled)**
If `--hard-neg-mining` is on, disable it:
```bash
# Remove --hard-neg-mining flag
```
Rationale: Hard-negative mining on shrunk budgets (2000–3500 residues) creates highly biased batches (hardest negatives are often outliers). With smaller batches, this variance explodes.

**Expected result**: More stable batch composition, less collapse variance.

### 4. **Medium-term: Improve Gradient Stability**
```bash
--gradient-clip-norm 10.0  # (up from 5.0, for non-MPS)
# MPS is already using clip_grad_value_=1.0; consider raising to 5.0
--weight-decay 1e-6  # (down from 1e-5, if applied)
--lr 5e-5  # (down from 1e-4, to avoid overshooting)
```
Rationale: Collapsed embeddings have nearly-zero gradients initially. With strong clipping (5.0 norm cap), even small gradients vanish. Raising the cap gives gradients room to work. Lower LR prevents overshooting once gradients return.

### 5. **Medium-term: Warmup & Scheduling**
Add a learning rate warmup:
```python
# In train_contrastive.py, after optimizer initialization
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
schedulers = [
    LinearLR(optimizer, start_factor=0.1, total_iters=5),  # warmup over 5 epochs
    CosineAnnealingLR(optimizer, T_max=45)  # cosine anneal over remaining epochs
]
scheduler = ChainedScheduler(schedulers)
```
Rationale: Early epochs are chaotic (small budget, high restarts). Warmup gives the model time to stabilize before pushing harder.

### 6. **Long-term: Revisit Contrastive Architecture**
- **Add projection head**: Project embeddings through a small MLP before the contrastive loss. This adds expressiveness and allows the model to learn a better loss landscape.
- **Supervised contrastive without hard negatives**: If cluster labels are available, use a supervised loss that doesn't weight hard negatives specially (beta=0). This is more stable.
- **Layer-wise AdamW**: Per-layer learning rate schedules can help optimize distinct components (encoder vs. head).

---

## Verdict

**The training configuration is broken for this scale.**

The 12.0GB cap + 4000 residue budget was calibrated for a smaller dataset or different hardware. On 67k proteins with 16GB, the budget is thrashing the system into a corner: memory pressure → restart loop → budget shrink → weak batches → collapse → futile restarts. The system is stuck in a stable pathological state—metrics still rise (because HDBSCAN is flexible and early epochs set a low baseline), but the embeddings are degenerate and the training loop is spending more CPU on restarts than on actual learning.

**Recommended action**: Increase `--mem-hard-gb 16.0`, raise `--temperature 0.2`, and monitor collapse signals. If collapse persists, reduce `--max-residues 3000` to stabilize memory *before* restarts occur (preventive rather than reactive).

---

## Appendix: Configuration Used

From training logs (inferred):
```
--mem-hard-gb 12.0
--mem-soft-gb 11.0
--max-residues 4000
--min-residues 2000
--temperature 0.1
--learning-rate 1e-4
--weight-decay 1e-5
--num-epochs 50
--dataset-size 67k (full)
--hard-neg-mining [possibly enabled]
--model-type asymmetric
```

