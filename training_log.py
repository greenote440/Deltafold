"""
Consolidated training log for the contrastive run.

Replaces the four separate verbose CSVs (contrastive_losses, collapse_metrics,
ari_log, epoch_eval) with a single gzipped JSONL file written once per epoch.
Each record is one JSON line — trivial to read, token-efficient to benchmark.

  Per-epoch record ("epoch"):
    ep, steps, train_loss, val_loss, ari (kmeans-val), fp_gb (peak footprint),
    restarts, budget, eval {ari, nmi, tm_rho, recall, k_clusters, singletons}
    + optional loss_range [min,max] for a quick sanity check

  Per-step record ("step", every DELTAFOLD_LOG_EVERY grad steps):
    ep, s (step), loss, lr, fp (footprint GB), std/cos (collapse-health probe).
    Buffered during the epoch and flushed with the epoch record.

  Sparse event records ("restart", "collapse"):
    only written when something notable happens — keeps the file short.

Usage (read everything):
  python -c "import gzip,json; [print(json.loads(l)) for l in gzip.open('checkpoints/training_log.jsonl.gz')]"

Filter to epoch summaries only:
  python -c "import gzip,json; [print(l) for l in (json.loads(x) for x in gzip.open('checkpoints/training_log.jsonl.gz')) if l['t']=='epoch']"
"""
import gzip
import json


class TrainingLog:
    """Buffers log events during an epoch and appends them as a gzip stream at
    epoch-end.  Multi-stream gzip is transparent to Python's gzip.open reader."""

    def __init__(self, path):
        self.path = path
        self._buf = []   # (dict,) records collected this epoch

    def event(self, rec):
        """Queue a sparse event record (restart, collapse, etc.)."""
        self._buf.append(rec)

    def flush_epoch(self, epoch_rec):
        """Write the epoch summary + any buffered events in one gzip stream."""
        records = [epoch_rec] + self._buf
        with gzip.open(self.path, 'ab') as fz:
            for rec in records:
                fz.write((json.dumps(rec, separators=(',', ':')) + '\n').encode())
        self._buf.clear()

    @staticmethod
    def read(path):
        """Return a list of all records from training_log.jsonl.gz."""
        recs = []
        with gzip.open(path, 'rb') as fz:
            for line in fz:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs
