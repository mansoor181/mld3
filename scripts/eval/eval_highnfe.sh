#!/usr/bin/env bash
# High-budget commit-k control: run the M=1 student for as many steps as the mixture policies
# spend in block passes (consensus at M=4, K=32 equals M=1 commit-k at K=64; best-of-4 equals
# K=126). The default grid brackets both.
#
# Usage:
#   RESULTS=$MLDF_RESULTS/lm1b_distill GPU=1 CELLS="lm1b_M1_K4_seed0:0050000" \
#     bash scripts/eval/eval_highnfe.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

RESULTS=${RESULTS:?set RESULTS}
CELLS=${CELLS:?set CELLS}
GPU=${GPU:-1}
NFE=${NFE:-48,64,96,128}
GEN_SAMPLES=${GEN_SAMPLES:-256}
SAMPLE_BATCH=${SAMPLE_BATCH:-64}
OUTNAME=${OUTNAME:-eval_highnfe_analytic.json}
LOG=${LOG:-$ROOT/logs/eval_highnfe_gpu$GPU.log}
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
  [[ -n "$CKPT" && -f "$CKPT" ]] || { echo "[highnfe $(date -Is)] no checkpoint for $cell, skipping" | tee -a "$LOG"; continue; }
  [[ -f "$OUT" ]] && { echo "[highnfe $(date -Is)] $cell already scored, skipping" | tee -a "$LOG"; continue; }
  echo "[highnfe $(date -Is)] $cell nfe=$NFE -> $OUT" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m eval.eval_text \
    --ckpt "$CKPT" --out "$OUT" --nfe "$NFE" \
    --gen_samples "$GEN_SAMPLES" --sample_batch "$SAMPLE_BATCH" --elbo_batches 8 \
    --sampler analytic --commit_k --no-best_of_m >> "$LOG" 2>&1
  echo "[highnfe $(date -Is)] $cell rc=$?" | tee -a "$LOG"
done
