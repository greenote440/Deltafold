"""
Memory accounting and governance for the contrastive training loop.

This machine is a 16GB Apple Silicon box, and the training loop creeps toward the
swap cliff across an epoch (DataLoader worker buffers + a growing/fragmented MPS
allocator pool). This module provides three things:

  * `_phys_footprint_gb()` — the TRUE memory footprint on Apple Silicon, including
    MPS (GPU) driver allocations that psutil's RSS does not see.
  * `free_memory()`        — device-agnostic reclamation (gc + empty_cache).
  * `MemoryGovernor`       — an escalating policy that keeps the footprint under a
                             hard cap by reclaiming, cold-restarting workers, and
                             dynamically shrinking the residue budget.

See memory note `training-perf` for the broader context.
"""
import ctypes
import gc
import os

import psutil
import torch

from train import DEVICE

_PROC = psutil.Process(os.getpid())


# ── macOS phys_footprint via Mach task_info ─────────────────────────────────
# On Apple Silicon unified memory, MPS (GPU) allocations are managed by the
# driver in a separate pool that is NOT counted in psutil RSS. For example,
# after a real model forward pass RSS reports ~0.3 GB while Activity Monitor
# (and this API) shows ~1.5 GB — the gap is 100% MPS allocations.  phys_
# footprint = what Activity Monitor calls "Memory" and what triggers swap.
class _TaskVMInfo(ctypes.Structure):
    _fields_ = [
        ('virtual_size',                ctypes.c_uint64),
        ('region_count',                ctypes.c_int32),
        ('page_size',                   ctypes.c_int32),
        ('resident_size',               ctypes.c_uint64),
        ('resident_size_peak',          ctypes.c_uint64),
        ('device',                      ctypes.c_uint64),
        ('device_peak',                 ctypes.c_uint64),
        ('internal',                    ctypes.c_uint64),
        ('internal_peak',               ctypes.c_uint64),
        ('external',                    ctypes.c_uint64),
        ('external_peak',               ctypes.c_uint64),
        ('reusable',                    ctypes.c_uint64),
        ('reusable_peak',               ctypes.c_uint64),
        ('purgeable_volatile_pmap',     ctypes.c_uint64),
        ('purgeable_volatile_resident', ctypes.c_uint64),
        ('purgeable_volatile_virtual',  ctypes.c_uint64),
        ('compressed',                  ctypes.c_uint64),
        ('compressed_peak',             ctypes.c_uint64),
        ('compressed_lifetime',         ctypes.c_uint64),
        ('phys_footprint',              ctypes.c_uint64),
    ]

_TASK_VM_INFO = 22
try:
    _libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    _tvi_count = ctypes.c_uint32(ctypes.sizeof(_TaskVMInfo) // 4)
    _HAS_MACH = True
except Exception:
    _HAS_MACH = False


def _phys_footprint_gb():
    """Physical memory footprint (GiB) matching Activity Monitor's 'Memory' column.
    Includes both CPU-side RSS and MPS (GPU) driver allocations on Apple Silicon.
    Falls back to psutil RSS on non-macOS."""
    if _HAS_MACH:
        try:
            info = _TaskVMInfo()
            c = ctypes.c_uint32(_tvi_count.value)
            if _libc.task_info(_libc.mach_task_self(), _TASK_VM_INFO,
                               ctypes.byref(info), ctypes.byref(c)) == 0:
                return info.phys_footprint / (1024 ** 3)
        except Exception:
            pass
    return _PROC.memory_info().rss / (1024 ** 3)


def _report_memory(step, epoch):
    return _phys_footprint_gb()


def free_memory():
    """Device-agnostic memory reclamation. Collects Python reference cycles and
    releases the framework's cached device allocations so that peak RAM does not
    creep upward from epoch to epoch (the leftover prefetch/worker buffers of a
    finished DataLoader iterator and a fragmented MPS/CUDA cache are the usual
    culprits behind 'each epoch gets slower'). The double gc.collect() handles
    objects whose __del__ resurrects references on the first pass."""
    gc.collect()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


class MemoryGovernor:
    """Keeps the process RSS under a hard cap on this 16GB machine.

    The training loop leaks across an epoch (DataLoader worker buffers + a growing/
    fragmented MPS allocator pool), creeping to ~27GB by epoch 5 -> swap -> 40s/step.
    Swapping is far slower than rebuilding the workers, so this governor escalates:

      every `cleanup_every` steps        -> baseline gc + empty_cache (cheap throttle)
      RSS > soft_gb                       -> force gc + empty_cache NOW (off-schedule)
      RSS > hard_gb (after a reclaim try) -> signal a COLD RESTART of the DataLoader
                                             workers and SHRINK the residue budget so
                                             subsequent batches allocate less.

    The residue budget recovers slowly across clean epochs so throughput is not
    permanently sacrificed after a transient spike."""

    def __init__(self, soft_gb=11.0, hard_gb=14.0, sampler=None, cleanup_every=50,
                 min_residues=2000, shrink=0.85, grow=1.1):
        self.soft_gb = soft_gb
        self.hard_gb = hard_gb
        self.sampler = sampler
        self.cleanup_every = cleanup_every
        self.min_residues = min_residues
        self.shrink = shrink
        self.grow = grow
        self.base_residues = getattr(sampler, 'max_residues', None)
        self.last_fp = 0.0
        self.restarts = 0
        self.soft_hits = 0
        self._epoch_restarts = 0

    def _reclaim(self):
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def after_step(self, step):
        """Call once per training step. Returns 'restart' if the caller should cold-
        restart the DataLoader workers, else None."""
        enabled = self.hard_gb and self.hard_gb > 0
        # Baseline throttled cleanup (unconditional, matches the old behaviour).
        if DEVICE.type == 'mps' and self.cleanup_every > 0 and (step % self.cleanup_every == 0):
            self._reclaim()
        if not enabled:
            self.last_fp = _phys_footprint_gb()
            return None
        rss = _phys_footprint_gb()
        if rss > self.hard_gb:
            self._reclaim()
            rss = _phys_footprint_gb()
            if rss > self.hard_gb:
                self.last_fp = rss
                self.restarts += 1
                self._epoch_restarts += 1
                if self.sampler is not None and self.base_residues:
                    self.sampler.max_residues = max(
                        self.min_residues, int(self.sampler.max_residues * self.shrink))
                return 'restart'
        elif rss > self.soft_gb:
            self.soft_hits += 1
            self._reclaim()
            rss = _phys_footprint_gb()
        self.last_fp = rss
        return None

    def end_of_epoch(self):
        """Slowly grow the residue budget back toward its base after a clean epoch; reset
        per-epoch counters. Returns a short status string for logging."""
        grew = False
        if self.sampler is not None and self.base_residues:
            if self._epoch_restarts == 0 and self.sampler.max_residues < self.base_residues:
                self.sampler.max_residues = min(
                    self.base_residues, int(self.sampler.max_residues * self.grow))
                grew = True
        msg = (f"restarts={self._epoch_restarts} soft_hits={self.soft_hits} "
               f"budget={getattr(self.sampler, 'max_residues', None)}"
               + (" (grown)" if grew else ""))
        self._epoch_restarts = 0
        self.soft_hits = 0
        return msg
