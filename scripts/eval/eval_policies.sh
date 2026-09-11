#!/usr/bin/env bash
# Decoding-policy sweep (eval.eval_policies): every policy in one pass over a checkpoint,
# writing eval_policies_analytic.json per cell. Source of the policy-ablation tables.
#
# Usage:
#   RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 CELLS="lm1b_M1_K4_seed0:0050000" \
#     bash scripts/eval/eval_policies.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

RESULTS=${RESULTS:?set RESULTS (a results directory holding the cell subdirectories)}
CELLS=${CELLS:?set CELLS (space separated "cell" or "cell:step" specs)}
GPU=${GPU:-0}
NFE=${NFE:-1,2,4,8,16,32}
GEN_SAMPLES=${GEN_SAMPLES:-256}
SAMPLE_BATCH=${SAMPLE_BATCH:-32}
OUTNAME=${OUTNAME:-eval_policies_analytic.json}
LOG=${LOG:-$ROOT/logs/eval_policies_gpu$GPU.log}
export WANDB_MODE=disabled
mkdir -p "$(dirname "$LOG")"

cd "$CODE"
for spec in $CELLS; do
  cell=${spec%%:*}; step=${spec#*:}
  celldir=$RESULTS/$cell
  if [[ "$step" == "$spec" ]]; then
    CKPT=$(ls -1 "$celldir"/ckpt_*.pt 2>/dev/null | sort | tail -1)
  else
    CKPT=$celldir/ckpt_${step}.pt
  fi
  OUT=$celldir/$OUTNAME
  [[ -n "$CKPT" && -f "$CKPT" ]] || { echo "[policies $(date -Is)] no checkpoint for $cell, skipping" | tee -a "$LOG"; continue; }
  [[ -f "$OUT" ]] && { echo "[policies $(date -Is)] $cell already scored, skipping" | tee -a "$LOG"; continue; }
  echo "[policies $(date -Is)] $cell nfe=$NFE -> $OUT" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m eval.eval_policies \
    --ckpt "$CKPT" --out "$OUT" --nfe "$NFE" \
    --gen_samples "$GEN_SAMPLES" --sample_batch "$SAMPLE_BATCH" \
    --sampler analytic >> "$LOG" 2>&1
  echo "[policies $(date -Is)] $cell rc=$?" | tee -a "$LOG"
done
