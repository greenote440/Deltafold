#!/usr/bin/env bash
# rsync helper between this Mac and the deltafold box — routine code sync and pulling
# run artifacts, WITHOUT the git commit/push/pull dance (git = milestones only).
#
#   ./scripts/sync_to_box.sh push              # Mac -> box: CODE (mirror, safe)
#   ./scripts/sync_to_box.sh push-dry          # preview 'push' (no changes)
#   ./scripts/sync_to_box.sh pull-ckpt <run>   # box -> Mac: checkpoints/<run>/ (additive)
#   ./scripts/sync_to_box.sh pull <path>       # box -> Mac: any repo-relative path (additive)
#
# Why nothing important gets deleted:
#   * push uses --delete so the box mirrors the Mac, BUT the excludes protect .git,
#     .venv and everything in .gitignore (data/, checkpoints/, code_and_intermediate_data/,
#     ...). rsync never deletes excluded paths, so the box's data + checkpoints + venv
#     are untouched. Only tracked code is mirrored.
#   * pull* NEVER pass --delete: they only add/update files on the Mac, never remove —
#     so downloading box checkpoints can't wipe local files.
#
# Uses the 'Deltafold' ssh alias, so it rides the ControlMaster socket (no re-auth).
set -euo pipefail

BOX="${DELTAFOLD_BOX:-Deltafold:/home/pnardi/Deltafold}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
# EXPLICIT anchored excludes (a --filter=':- .gitignore' did NOT reliably protect these,
# so we do not depend on it). --delete never removes an excluded path, so the box's data,
# every checkpoint, the venv and the intermediate data are safe. Also blanket-exclude
# *.pt/*.pth so no stray model/embedding file is ever pushed or deleted.
CODE_EX=(
  --exclude='/.git/' --exclude='/.venv/' --exclude='__pycache__/' --exclude='.DS_Store'
  --exclude='/data/' --exclude='/checkpoints/' --exclude='/clusters/'
  --exclude='/code_and_intermediate_data/' --exclude='/documents/'
  --exclude='*.pt' --exclude='*.pth' --exclude='*.pdf'
)

cmd="${1:-}"; shift || true
case "$cmd" in
  push)      rsync -az  --delete "${CODE_EX[@]}" "$LOCAL/" "$BOX/" ;;
  push-dry)  rsync -azn --delete --itemize-changes "${CODE_EX[@]}" "$LOCAL/" "$BOX/" ;;
  pull-ckpt) run="${1:?usage: pull-ckpt <run_folder>}"
             mkdir -p "$LOCAL/checkpoints/$run"
             rsync -az --info=progress2 "$BOX/checkpoints/$run/" "$LOCAL/checkpoints/$run/" ;;
  pull)      p="${1:?usage: pull <repo-relative-path>}"
             rsync -azR --info=progress2 "${BOX%/}/./$p" "$LOCAL/" ;;
  *)         echo "usage: $(basename "$0") {push|push-dry|pull-ckpt <run>|pull <path>}"; exit 1 ;;
esac
echo "done: $cmd ${1:-}"
