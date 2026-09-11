#!/usr/bin/env bash
# Train a list of cells sequentially on one GPU. The trainer resumes from the newest checkpoint
# and exits quickly on a finished cell, so rerunning the list restarts an interrupted queue.
#
# Usage:
#   GPU=0 CELLS="imagenet256_maskgitT_di4c_M8 imagenet256_maskgitT_r2_M1" bash scripts/train/train_cells.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

CELLS=${CELLS:?set CELLS (space separated config names, without the .yaml suffix)}
GPU=${GPU:-0}
CFG_DIR=${CFG_DIR:-$CODE/configs}
LOG_DIR=${LOG_DIR:-$ROOT/logs/train}
mkdir -p "$LOG_DIR"
cd "$CODE"

for cell in $CELLS; do
  cfg=$CFG_DIR/$cell.yaml
  if [[ ! -f "$cfg" ]]; then
    echo "[gpu$GPU] $(date +%H:%M:%S) no config $cfg, skipping $cell" >&2
    continue
  fi
  echo "[gpu$GPU] $(date +%H:%M:%S) start $cell" >&2
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m training.trainer_distill \
    --config "$cfg" > "$LOG_DIR/$cell.train.log" 2>&1
  echo "[gpu$GPU] $(date +%H:%M:%S) done  $cell (rc=$?)" >&2
done
