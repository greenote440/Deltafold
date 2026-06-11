# Memory Audit: Training Loop

**Purpose:** Systematically trace all GPU/CPU memory allocations in the contrastive training loop and verify cleanup is complete.

**Device:** Apple Silicon M1 Pro, 16GB unified memory  
**Bottleneck:** MPS allocator pool (shared GPU + CPU), not separate VRAM like CUDA

---

## Section 1: Training Loop Memory Lifecycle

### A. Per-Step Allocations (Lines 986-1114)

#### 1. **Batch Loading** (Line 989)
```python
for batch in _restartable_batches():  # batch = (features_dict, paths)
```
**Allocation:**
- `batch`: 2×(features from hardneg sampler)
  - rank0: ~2000 nodes × 1 dim = 8KB
  - rank1: ~2000 edges × 16 neighbors = 128KB
  - rank2_features: ~250 SSEs × 8 dim = 8KB
  - rank3: 8 proteins
  - Total per batch: ~2-3 MB

**Cleanup:**
- ✓ Line 1111: `del features, model_in, z, batch`
- ✓ Autogarbage at iteration end

---

#### 2. **Feature Augmentation** (Lines 1000-1010)
```python
r3 = features['rank3']
r3['radius_of_gyration'] *= (1 + 0.10 * torch.randn_like(...))
r3['global_shape_descriptors'] *= (1 + 0.05 * torch.randn_like(...))
r3['protein_size'] = torch.ones_like(...) * 500.0
r3['radius_of_gyration'] = torch.zeros_like(...)
```
**Allocation:**
- `torch.randn_like()`: Creates temp tensor same shape as input
  - radius_of_gyration: 8 × 1 = 64 bytes
  - global_shape_descriptors: 8 × variable = ~2KB
  - `torch.ones_like()`: 8 × 1 = 64 bytes
  - `torch.zeros_like()`: 8 × 1 = 64 bytes
- All temporary; overwritten immediately

**Cleanup:**
- ✓ Implicit (all in-place ops, no holding references)
- **NOTE:** In-place ops on r3 don't require explicit del

---

#### 3. **Bucketing / Padding** (Line 1015)
```python
model_in, real_B = pad_to_buckets(features)
```
**Allocation:** (See contrastive_data.py:33)
```python
out = dict(features)  # shallow copy
out['rank0'] = new_r0  # new padded tensors
out['rank1'] = new_r1
out['rank2_features'] = r2f  # padded
out['rank3'] = new_r3
```
**Size breakdown:**
- Original: 2000 nodes, 250 SSEs, 8 proteins
- Padded to buckets (1024, 256, 8):
  - rank0: (2048 - 2000) × dims = 48 × 1 = 48 bytes
  - rank1: 48 × 16 = 768 bytes
  - rank2: (256 - 250) × 8 = 48 bytes
  - rank3: unchanged (already 8 bucket size)
  - **Total new:** ~1KB

**Cleanup:**
- ✓ Line 1111: `del model_in`
- **ISSUE:** `features` is still referenced, kept in memory

---

#### 4. **Forward Pass** (Line 1018)
```python
z = model(model_in)  # AsymmetricTopoNet forward
```
**Allocation:**
- Model parameters: ~1.2M × 4 bytes = ~5MB (shared, not per-step)
- **Per-step GPU activations:**
  - rank0 (padded): 2048 × hidden × 4 bytes = 2048 × 128 × 4 = 1MB
  - rank1 (edges): 2048 × 16 × hidden × 4 = ~4MB
  - rank2 (SSEs): 256 × hidden × 4 = ~1MB
  - rank3 (proteins): 8 × hidden × 4 = ~4KB
  - **Intermediate buffers:** ~3-5MB (message passing, attention)
  - **Total per forward:** ~10-12MB

**Cleanup:**
- ✓ Line 1018-1021: `z = model(...); z = z[:real_B]` (slicing doesn't copy)
- ✓ Line 1111: `del z`
- **ISSUE:** Autograd keeps full `model_in` in backward tape until backward() completes

---

#### 5. **Loss Computation** (Lines 1023-1049)

##### 5a. Cluster Label Construction (Lines 1025-1028)
```python
cluster_ids = [acc_to_cluster.get(...) for p in paths]  # list
label_map = {cid: i for i, cid in enumerate(set(cluster_ids))}  # dict
labels = torch.tensor([label_map[cid] for cid in cluster_ids], device=DEVICE)
```
**Allocation:**
- cluster_ids: 128 strings (Python list) = ~8KB
- label_map: dict of ~64 unique clusters = ~4KB
- labels: 128 × int64 = 1KB
- **Total:** ~13KB

**Cleanup:**
- ✗ **MISSING:** `cluster_ids`, `label_map` never deleted
  - Both are small Python objects (GC should handle), but if `extract_accession()` is expensive, this could leak
  - **RECOMMENDATION:** Add `del cluster_ids, label_map` after line 1028

---

##### 5b. TM-Score Auxiliary Loss (Lines 1031-1049)
```python
tm_matrix = build_tm_matrix(paths, tm_cache, z.device)  # Line 1031
loss = soft_supcon_loss(z, labels, tm_matrix, temperature=0.1)  # Line 1032
```
or
```python
tm_loss = tm_score_aux_loss_cached(z, paths, tm_cache)  # Line 1045
```

**Allocation:**
- `tm_matrix`: Build from paths
  - For 128 proteins: 128×128 float = 65KB (small)
  - But **inside soft_supcon_loss**, more allocations happen (see below)

**Cleanup:**
- ✗ **MISSING:** `tm_matrix` is created but never deleted
  - Small (~65KB), but Python object lifecycle unclear
  - **RECOMMENDATION:** Add explicit del after loss computation

---

##### 5c. Supervised SupCon Loss (Line 1034-1035 or 1032)
```python
loss = supervised_ntxent_loss(z, labels, temperature=0.1, hard_neg_beta=0.0)
```

**Inside `contrastive_engine.py:657-735` (supervised_ntxent_loss):**

```python
logits = z @ z.T  # Line ~680: 128×128 float = 65KB
logits_max = logits.max(dim=0, keepdim=True).values  # Line ~685: 1×128 = 512B
logits = logits - logits_max.detach()  # Line ~686: in-place subtract (no new alloc)
labels_eq = torch.eq(labels, labels.T).float().to(dev)  # Line ~661: 128×128 bool = 16KB
pos_weights = labels_eq.clone()  # Line ~664: 128×128 float = 65KB (COPY!)
```

**Total allocations in loss function:**
- logits: 65KB
- logits_max: 512B
- labels_eq: 16KB
- pos_weights: **65KB (unnecessary copy!)**
- neg_mask, pos_mask, etc.: ~16KB
- **Total:** ~166KB per step

**Cleanup:**
- ✗ **ISSUE:** Inside loss function, no explicit cleanup
  - PyTorch autograd graph holds all intermediate tensors until backward() completes
  - This is by design (needed for gradient computation)
  - **EXPECTED BEHAVIOR**
- ✓ Backward release: When `loss.backward()` (line 1079) completes, autograd tape is freed

---

#### 6. **Gradient Computation** (Line 1079)
```python
scaled_loss.backward()
```
**Allocation:**
- Backward pass creates gradient tensors for each parameter
- For 1.2M params: 1.2M × 4 bytes = 4.8MB (gradients)
- Stored in `.grad` buffers

**Cleanup:**
- ✓ Line 1100: `optimizer.zero_grad()` — clears `.grad`

---

#### 7. **Collapse Metrics** (Lines 1052-1053)
```python
emb_std, mean_cos = collapse_metrics(z)  # z is 128×128
```

**Inside `contrastive_engine.py:523-535`:**
```python
zn = F.normalize(z, p=2, dim=-1)  # normalized: 128×128 float = 65KB
sims = zn @ zn.t()  # pairwise similarity: 128×128 = 65KB
off = sims[~torch.eye(n, dtype=torch.bool, device=zn.device)]  # 128×127 = ~65KB
mean_cos = off.mean().item()  # scalar
```

**Total:** ~195KB per collapse check (every 10 steps)

**Cleanup:**
- ✗ **ISSUE:** Inside `collapse_metrics()`, no explicit cleanup
  - But: function is `with torch.no_grad()`, autograd doesn't track
  - Tensors should be freed at function return
  - **EXPECTED BEHAVIOR** (small enough not to accumulate)

---

#### 8. **Loss Scalar Extract** (Line 1081)
```python
lv = loss.item()  # extracts Python float
```
**Allocation:**
- 8 bytes (Python float)

**Cleanup:**
- ✓ Line 1083: `step_losses.append(lv)` — appends float, not tensor
- ✓ Line 1106: `step_losses.clear()` — clears list every 10 steps

---

#### 9. **Gradient Clipping** (Lines 1088-1097)
```python
params_to_clip = [p for p in model.parameters() if p.grad is not None]
torch.nn.utils.clip_grad_value_(params_to_clip, clip_value=1.0)
```

**Allocation:**
- params_to_clip: list of ~1.2M parameter references = ~100KB (list metadata)
- **No new tensor allocation** (clip is in-place on .grad)

**Cleanup:**
- ✓ Implicit: params_to_clip is local list, freed at scope exit

---

### Summary: Per-Step Memory Leaks

| Component | Allocation | Status | Fix |
|-----------|-----------|--------|-----|
| Batch loading | 2-3 MB | ✓ Cleaned (del batch) | — |
| Bucketing | ~1 KB | ⚠️ `features` kept | Could del features after model_in is freed |
| Forward pass | 10-12 MB | ✓ Autograd manages | — |
| Cluster labels | 13 KB | ✗ No del | Add del cluster_ids, label_map |
| TM-matrix | 65 KB | ✗ No del | Add del tm_matrix |
| SupCon loss | 166 KB | ✓ Autograd manages | — |
| Collapse metrics | 195 KB | ✓ Function local | — |
| **Per-step total** | **~200KB** | Mixed | Minor cleanup needed |

---

## Section 2: Validation Loop Memory (Lines 1138-1193)

### Key Difference: No Backward Pass

```python
with torch.no_grad():
    for v_step, v_batch in enumerate(tqdm(val_loader, ...)):
        vz = model(v_in)  # forward only
        val_loss_epoch += criterion(vz[:VB], vz[VB:]).item()
        ...
        del v_feat, v_in, vz, v_batch
```

**Allocations:**
- Same forward allocations as training (10-12MB per batch)
- **No gradient buffers** (no backward pass)
- **No autograd graph** (with torch.no_grad())

**Cleanup:**
- ✓ Line 1165, 1179: Explicit del for all intermediate tensors
- ✓ Lines 1184-1192: Periodic gc.collect() + torch.mps.empty_cache()
  - Every val_cleanup steps (default 25), or if soft_gb exceeded
  - **THIS IS CRITICAL:** Val loop originally had NO reclaim (27GB swap blowup before fix)

**Current status:** ✓ Fixed in Lines 1181-1188

---

## Section 3: End-of-Epoch Cleanup (Lines 1194-1280)

### 3a. ARI Computation (Lines 1204-1210)
```python
embs = np.concatenate(ari_embs, axis=0)  # 13967×128 float32 = 7.2MB
labels = [acc_to_cluster.get(...) for p in ari_paths]  # strings
```

**Cleanup:**
- ✗ **MISSING:** `embs`, `labels` never deleted explicitly
  - Held in scope until end of epoch
  - **IMPACT:** 7MB bloat per epoch (acceptable, one per epoch)

---

### 3b. K-means Clustering (Lines ~1215-1230)
```python
from sklearn.cluster import KMeans
best_ari = -1
for seed in range(n_seeds):
    kmeans = KMeans(n_clusters=..., random_state=seed)
    clusters = kmeans.fit_predict(embs)  # allocates kmeans object + clusters
```

**Allocations:**
- KMeans object per seed: ~1MB (centroids, labels, etc.)
- clusters array: 13967 × int = 56KB per seed

**Cleanup:**
- ✗ **MISSING:** `kmeans`, `clusters` never deleted
  - But: Loop replaces each iteration (old one GC'd)
  - **EXPECTED BEHAVIOR** (GC will clean)

---

### 3c. Checkpoint Saving (Lines 1274-1291)
```python
checkpoint_data = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),  # copy of all params
    'optimizer_state_dict': optimizer.state_dict(),  # copy of all optimizer states
    'scheduler_state_dict': scheduler.state_dict(),
    'best_loss': best_loss,
    'best_ari': best_ari,
    'model_config': model_config or {},
}
torch.save(checkpoint_data, last_path)
```

**Allocation:**
- state_dict() = shallow copy of parameter references + values
  - model: 1.2M × 4 bytes = 5MB
  - optimizer states (m, v for Adam): 2 × 5MB = 10MB
  - **Total:** ~15MB
- All held in memory during save

**Cleanup:**
- ✓ Line 1298: `del checkpoint_data`
- ✓ Lines 1300-1301: `gc.collect()` explicit final reclaim

---

## Section 4: Across-Epoch Accumulation

### Potential Leaks That Grow Across Epochs

| Variable | Scope | Lifecycle | Risk |
|----------|-------|-----------|------|
| `step_losses` | global in loop | Cleared every 10 steps (line 1106) | ✓ No accumulation |
| `ari_embs` | epoch-local | Reset per epoch (line 1142) | ✓ No accumulation |
| `tlog` (event buffer) | global | Flushed per epoch (line 1306) | ✓ No accumulation |
| Model parameters | global | Persistent (intended) | ✓ Intended |
| Optimizer states | global | Persistent (intended) | ✓ Intended |
| DataLoader workers | epoch-local | Cold-restarted on restart | ✓ Cleaned |

---

## Section 5: DataLoader Memory (train_contrastive.py Lines 808-849)

### 5a. Worker Buffers
```python
def _make_loader(dataset, *, batch_sampler=None, workers=num_workers, ...):
    return DataLoader(..., num_workers=4, prefetch_factor=2, persistent_workers=False)
```

**Allocation:**
- Worker processes: 4 CPU processes
- Each worker buffers ~2 batches (prefetch_factor=2)
- Per-worker: 2 × 3MB = 6MB
- Total workers: 4 × 6MB = 24MB

**Cleanup:**
- ✗ **ISSUE FOUND:** Workers leak memory across epochs
  - Workers are created per epoch with `make_train_loader()`
  - Each restart creates new workers (old ones should be GC'd)
  - But: **If old workers aren't fully released, memory accumulates**
  - **SYMPTOM:** This is exactly what caused restart spam at epoch 9
  - **FIX:** Cold-restart at line 979 calls `governor._reclaim()` after `del loader`
  - **BUT:** Need to verify workers are actually released

---

### 5b. Sampler Batch Keys Cache
```python
cache_path = os.path.join(CHECKPOINT_DIR, 'batch_keys_cache.pt')
keys = extract_batch_keys(train_files, cache_path)  # Line 827
train_sampler = HardNegativeBatchSampler(keys, batch_size, seed=42, max_residues=4000)
```

**Allocation:**
- keys: list of 53731 (size, helix_ratio, sheet_ratio) tuples = ~2MB
- Held in sampler.keys (reference, not copied)

**Cleanup:**
- ✓ Sampler is re-created per epoch (line 827)
- ✓ Old sampler GC'd

---

## CRITICAL ISSUES FOUND

### Issue 1: Missing Deletes (Low Risk)
```python
# Line 1027-1028: Create but never delete
label_map = {cid: i for i, cid in enumerate(set(cluster_ids))}
labels = torch.tensor([label_map[cid] for cid in cluster_ids], device=DEVICE)

# Line 1031: Create but never delete
tm_matrix = build_tm_matrix(paths, tm_cache, z.device)

# Line 1207: Create but never delete
embs = np.concatenate(ari_embs, axis=0)
```
**Impact:** ~100KB per step, 13KB per collapse check, 7MB per epoch (acceptable)  
**Severity:** Low (GC will clean, but explicit cleanup better)

**Fix:**
```python
# After line 1035
del label_map, labels

# After line 1032 or 1049
if tm_matrix is not None:
    del tm_matrix

# After line 1230 (after ARI computation)
del embs
```

---

### Issue 2: DataLoader Worker Release (HIGH RISK)
```python
# Line 972-979: Restart creates new loader, deletes old
loader = make_train_loader()
...
del loader
governor._reclaim()  # Calls gc.collect() + torch.mps.empty_cache()
```

**Problem:** Worker processes may hold MPS GPU memory that isn't released by `del loader`  
**Evidence:** Epoch 9 hit restart spam (memory didn't decrease even after reload)

**Fix:**
```python
# After del loader, before break (line 978)
loader = None
gc.collect()
if DEVICE.type == 'mps':
    torch.mps.empty_cache()
gc.collect()
```

---

### Issue 3: features Kept in Scope (LOW RISK)
```python
# Line 997: feature dict created
features = to_device(batch_data, DEVICE)

# Line 1015: bucketed version created
model_in, real_B = pad_to_buckets(features)

# Line 1111: only model_in deleted, not features
del features, model_in, z, batch
```

**Problem:** `features` is created fresh per step; should be freed immediately after pad_to_buckets  
**Risk:** Low (2-3MB, but adds up across 4000+ steps per epoch)

**Fix:** Better scope management
```python
if pad_buckets:
    model_in, real_B = pad_to_buckets(features)
    del features  # Free original after bucketing
else:
    model_in, real_B = features, ...
```

---

## RECOMMENDATIONS

### Priority 1: Add Worker Cleanup (Prevents Restart Spam)
```python
# In _restartable_batches(), after line 978:

del loader
gc.collect()
if DEVICE.type == 'mps':
    torch.mps.empty_cache()
gc.collect()
```

### Priority 2: Add Explicit Deletes (Cleanup Hygiene)
```python
# Line 1035 (after supervised loss):
if 'label_map' in locals():
    del label_map

# Line 1049 (after TM loss):
if tm_loss is not None and 'tm_matrix' in locals():
    del tm_matrix

# Line 1230 (after ARI):
del embs, labels
```

### Priority 3: Early Feature Release (Performance)
```python
# Line 1015-1017:
model_in, real_B = pad_to_buckets(features)
del features
```

### Priority 4: Reduce hard_gb (Reduces Pressure)
```bash
# Current
python train.py ... --mem-hard-gb 14.0

# Recommended
python train.py ... --mem-hard-gb 12.0
```

---

## Testing Plan

1. **Run with Issue 1 + Issue 2 fixes applied**
   ```bash
   python train.py ... --mem-hard-gb 12.0
   # Expected: 0 restarts per epoch (vs 1-2 before)
   ```

2. **Monitor memory over 5 epochs**
   - Should stay bounded at 10-12GB
   - No linear growth

3. **Verify no regression**
   - Per-step time should be same or better
   - Loss curve smooth

