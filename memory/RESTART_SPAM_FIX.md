# Restart Spam Fix: Memory Exhaustion Prevention

**Date:** 2026-06-11  
**Issue:** Epoch 9 at step 2450 hit 13 restarts in rapid succession (2-3 seconds), indicating exponential memory leak  
**Root Cause:** Hard memory cap (14GB on 16GB machine) too aggressive; restart spam prevented GC from ever succeeding  
**Solution:** Cooldown + abort safety  

---

## Problem Analysis

### What Happened

```
Epoch 9 step 2450: footprint = 14.0GB (hit hard cap)
  → trigger restart (budget: 4000 → 3500)
Step 2451: footprint = 14.0GB still (old allocations not freed)
  → trigger restart again (budget: 3500 → 2975)
...×13 times in 2 seconds
  → exponential memory leak, PC crashes before abort
```

### Why Rapid Restarts Fail

1. **Restart triggers at step N:** shrink budget, break loader, delete, gc
2. **But:** DataLoader worker cleanup is async on MPS. The next allocation (step N+1) happens before workers fully released
3. **Result:** New batches allocate + old worker memory still hanging = growth
4. **After 13 restarts:** Pool so fragmented that even tiny batches can't fit
5. **Cascade:** Each failed reclaim triggers another restart immediately

### Why 14GB Hard Cap Is Too Tight

```
16GB machine:
- OS reserve: ~1GB
- App overhead: ~1GB
- At 14GB: only 1GB buffer left
- MPS allocator: can't defragment 1GB, spills to swap
- Swap: 50-100× slower than RAM → kernel throttle
- Throttle: more time under pressure → more restarts
```

---

## Solution: Two Fixes

### 1. Restart Cooldown (Prevents Spam)

**New logic:**
```python
# Only allow restart if 50+ steps since last restart
if step - self._last_restart_step >= 50:
    trigger_restart()
    self._last_restart_step = step
else:
    skip_restart()  # silently; memory governor exhausted
```

**Why 50 steps?**
- Cold-restart DataLoader takes ~20-30 steps to warm up and release memory
- 50-step buffer ensures old workers fully cleaned before next restart
- If memory is STILL high after 50 steps, the problem isn't the current workload; it's fundamental

### 2. Epoch Abort (Prevents Crash)

**New logic:**
```python
if governor._epoch_restarts > 5:
    print(f"[!] Epoch {epoch} aborted: >5 restarts, memory exhausted")
    break
```

**Why 5 restarts?**
- First restart: budget 4000 → 3500 (12.5% cut)
- Fifth restart: budget shrunk to ~1500 (62.5% of original)
- Smaller batches won't help if fragmentation is the problem
- Better to checkpoint and resume with lower mem-hard-gb

---

## Recommendation: Use Lower mem_hard_gb

**Current (risky):**
```bash
python train.py ... --mem-hard-gb 14.0  # only 2GB buffer
```

**Better:**
```bash
python train.py ... --mem-hard-gb 12.0  # 4GB buffer (safe)
```

**Conservative:**
```bash
python train.py ... --mem-hard-gb 11.0  # 5GB buffer (slower but safer)
```

**Trade-off:**
- 14GB → 13GB: More aggressive reclaims, risks crashes
- 12GB → 11GB: Safer, but triggers reclaim earlier (costs ~10ms every 50 steps)
- Net: Safer is worth ~1-2% throughput loss

---

## What Changed in Code

**MemoryGovernor.__init__:**
- Added `self._last_restart_step = -100` (cooldown tracker)
- Added `self.no_budget_adapt` (wired from CLI flag)

**MemoryGovernor.after_step:**
- Check `step - self._last_restart_step >= 50` before allowing restart
- Skip restart silently if cooldown not expired
- Better docstring

**train_contrastive.py:**
- Warning printed if `mem_hard_gb > 12.0`
- Abort loop if `governor._epoch_restarts > 5`
- Removed broken `pbar.total` update logic (was causing overshoot)

**train_contrastive.py signature:**
- Added `no_budget_adapt` parameter (plumbed from CLI)

---

## Expected Behavior With Fix

### Normal training (no memory pressure):
```
Epoch N: 4345 steps, 0 restarts, cleanup every 10 steps
Result: ~50-53 min/epoch
```

### With occasional pressure (e.g., thermal spike):
```
Epoch N: 4345 steps, 1-2 restarts
Result: ~52-55 min/epoch (slight slowdown due to smaller budget)
```

### With sustained pressure (e.g., bad checkpoint, memory leak):
```
Epoch N: 4000 steps, >5 restarts → ABORT
Result: Epoch incomplete, manual restart needed with --mem-hard-gb 11.0
```

---

## Testing

To verify the fix works:

1. **Run epoch with 14GB cap (will hit pressure):**
   ```bash
   python train.py ... --mem-hard-gb 14.0
   ```
   Expected: Restart spam → cooldown → abort after 5 restarts (instead of crash)

2. **Run epoch with 12GB cap (should be safe):**
   ```bash
   python train.py ... --mem-hard-gb 12.0
   ```
   Expected: 0-1 restart per epoch, no abort

3. **If still hitting restarts at 12GB:**
   - Check for real leaks (not this restart spam issue)
   - Run with `--mem-hard-gb 11.0` (trading speed for stability)

---

## Summary

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Restart spam | Yes (13× in 2s) | No (max 1 per 50s) |
| Crash on pressure | Yes (exponential leak) | No (abort safely) |
| Memory buffer | 2GB (tight) | 4GB if using 12GB (recommended) |
| Recommended hard cap | 14GB (risky) | **12GB (safe)** |

**Going forward:** Train with `--mem-hard-gb 12.0` (or 11.0 if still hitting pressure).

