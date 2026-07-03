#!/usr/bin/env bash
# Overnight one-factor-at-a-time ablation sweep (plan_experimentation_v2 §5).
#
# Runs the InfoNCE BASE config on the corrected sub-base plus single-factor
# variants (axes B/C/D/E + the feature-shortcut control), each for $EPOCHS
# epochs, saving EVERY epoch checkpoint under checkpoints/sweep/<run>/ so the
# weights can be analysed tomorrow.
#
# Experimental hygiene: NO SupCon, NO TM-auxiliary loss (those would inject
# label/TM supervision and break the unsupervised ablation). The TM cache is
# passed for EVALUATION ONLY — with --tm-aux-weight 0 and no --soft-supcon it
# never enters the loss, it just lets each epoch log val TM-rho / recall.
#
# Each run is isolated via DELTAFOLD_CKPT_DIR so checkpoints / training_log /
# epoch_eval / batch_keys_cache don't collide between configs.
#
# 14 runs (9 single-factor + 5 hypothesis-driven combinations). At ~5.5 min/epoch
# on this machine: EPOCHS=8 -> ~10-11h, EPOCHS=6 -> ~8h, EPOCHS=5 -> ~6.5h.
# Pick EPOCHS to fit the night, or trim with RUNS_ONLY.
#
# Usage:   bash scripts/run_overnight_sweep.sh
#          EPOCHS=6 bash scripts/run_overnight_sweep.sh             # fit ~8h
#          RUNS_ONLY="base E_no_hardneg E_nohn_jitter1.0" bash scripts/run_overnight_sweep.sh
#          DRY_RUN=1 bash scripts/run_overnight_sweep.sh            # print commands only
set -uo pipefail

cd "$(dirname "$0")/.."                                   # repo root
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml_env 2>/dev/null || true

EPOCHS="${EPOCHS:-8}"
TM_CACHE="checkpoints/tm_score_cache.pt"
SWEEP_ROOT="checkpoints/sweep"
mkdir -p "$SWEEP_ROOT"

[ -f "$TM_CACHE" ] || { echo "Missing $TM_CACHE — build it first (build_tm_cache.py)."; exit 1; }

# --- Single-instance lock -------------------------------------------------
# The loop below is strictly sequential (each `python train.py` runs in the
# foreground; the next starts only after it exits). The ONLY way to get parallel
# trainings — which saturates RAM — is to launch this script more than once.
# This atomic mkdir lock makes a second launch refuse to start. Skipped for DRY_RUN.
if [ -z "${DRY_RUN:-}" ]; then
  LOCKDIR="$SWEEP_ROOT/.sweep.lock"
  if ! mkdir "$LOCKDIR" 2>/dev/null; then
    owner="$(cat "$LOCKDIR/pid" 2>/dev/null || echo '?')"
    if kill -0 "$owner" 2>/dev/null; then
      echo "A sweep is ALREADY running (PID $owner). Refusing to start a second"
      echo "(that is what saturated your RAM). Wait for it, or kill it and remove $LOCKDIR."
      exit 1
    fi
    echo "Stale lock from dead PID $owner — reclaiming."
    rm -rf "$LOCKDIR"; mkdir "$LOCKDIR"
  fi
  echo "$$" > "$LOCKDIR/pid"
  trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
fi

# Flags shared by every run (the §5 base config; --tm-cache is eval-only here).
COMMON=(--task contrastive --model asymmetric --split corrected --unsupervised
        --tm-cache "$TM_CACHE" --mem-hard-gb 12.0 --epochs "$EPOCHS")

# BASE factor settings (each variant changes exactly ONE of these):
#   RBF encoding · jitter 0.3 · hard-neg ON · no crop · all features · tau 0.1.
BASE="--dist-encoding rbf --jitter-sigma 0.3 --hard-neg-mining"

# name :: flags-that-REPLACE-the-base-factors.
# Do NOT put inline '# comments' after these entries — a '#' adjacent to the
# closing quote is not a comment and leaks into the flags. Legend:
#   --- single factor (plan §5, one axis at a time) ---
#   E_no_hardneg  axis E  hard-neg OFF (tests the false-negative hypothesis)
#   C_jitter*     axis C  stronger views (non-trivial positives)
#   C_crop        axis C  crop augmentation on
#   D_sinusoidal  axis D  sinusoidal vs RBF encoding
#   B_tau*        axis B  InfoNCE temperature
#   feat_geomonly control drop 3di/residue/positional — do features leak structure?
#   --- combinations (hypothesis: remove false negatives AND make positives
#       non-trivial -> the only way unsupervised training should beat the
#       feature baseline). All still unsupervised; no SupCon / no TM-aux. ---
#   E_nohn_jitter1.0       no hard-neg + strong jitter (lead hypothesis)
#   E_nohn_crop            no hard-neg + crop
#   E_nohn_strongviews     no hard-neg + jitter 1.0 + crop (max non-trivial views)
#   E_nohn_tau0.05         no hard-neg + low tau (more uniform spread / anti-collapse)
#   geomonly_nohn          geometry-only features + no hard-neg (forced to learn
#                          geometry, not sabotaged by false negatives)
RUNS=(
  "base              :: $BASE"
  "E_no_hardneg      :: --dist-encoding rbf --jitter-sigma 0.3"
  "C_jitter0.6       :: --dist-encoding rbf --jitter-sigma 0.6 --hard-neg-mining"
  "C_jitter1.0       :: --dist-encoding rbf --jitter-sigma 1.0 --hard-neg-mining"
  "C_crop            :: $BASE --crop-aug"
  "D_sinusoidal      :: --dist-encoding sinusoidal --jitter-sigma 0.3 --hard-neg-mining"
  "B_tau0.2          :: $BASE --temperature 0.2"
  "B_tau0.05         :: $BASE --temperature 0.05"
  "feat_geomonly     :: $BASE --no-3di --no-residue --no-positional"
  "E_nohn_jitter1.0  :: --dist-encoding rbf --jitter-sigma 1.0"
  "E_nohn_crop       :: --dist-encoding rbf --jitter-sigma 0.3 --crop-aug"
  "E_nohn_strongviews:: --dist-encoding rbf --jitter-sigma 1.0 --crop-aug"
  "E_nohn_tau0.05    :: --dist-encoding rbf --jitter-sigma 0.3 --temperature 0.05"
  "geomonly_nohn     :: --dist-encoding rbf --jitter-sigma 0.3 --no-3di --no-residue --no-positional"
)

echo "Sweep: ${#RUNS[@]} runs x $EPOCHS epochs -> $SWEEP_ROOT   (started $(date))"
SECONDS=0
for spec in "${RUNS[@]}"; do
  name="$(echo "${spec%%::*}" | xargs)"
  flags="$(echo "${spec##*::}" | xargs)"
  # Optional allow-list via RUNS_ONLY="name1 name2"
  if [ -n "${RUNS_ONLY:-}" ] && [[ " $RUNS_ONLY " != *" $name "* ]]; then continue; fi
  rundir="$SWEEP_ROOT/$name"
  mkdir -p "$rundir"
  echo
  echo "===== [$name]  ($(date '+%H:%M:%S'))  flags: $flags ====="
  if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY: DELTAFOLD_CKPT_DIR=$rundir python train.py ${COMMON[*]} $flags"
    continue
  fi
  # Record exactly what this run was, so analysis tomorrow needs no log-parsing.
  printf '{"name":"%s","epochs":%s,"started":"%s","common":"%s","flags":"%s"}\n' \
    "$name" "$EPOCHS" "$(date -Iseconds)" "${COMMON[*]}" "$flags" > "$rundir/config.json"
  DELTAFOLD_CKPT_DIR="$rundir" DELTAFOLD_LOG_EVERY="${DELTAFOLD_LOG_EVERY:-20}" \
    python train.py "${COMMON[@]}" $flags 2>&1 | tee "$rundir/run.log"
  echo "----- [$name] done | elapsed $((SECONDS/60))m -----"
done

echo
echo "============================ SWEEP SUMMARY ============================"
python scripts/analysis/summarize_sweep.py "$SWEEP_ROOT" || true
echo "Total sweep time: $((SECONDS/60))m. Per-epoch checkpoints: $SWEEP_ROOT/<run>/checkpoint_contrastive_asymmetric_epoch*.pth"
