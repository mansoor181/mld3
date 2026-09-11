#!/usr/bin/env bash
# Score finished checkpoints on the NFE grid under one decoding policy.
#
# Policies (section 6.3 of the paper): commit (commit-k, the default), cons (consensus commit,
# the headline policy), conssoft, bom (best-of-M, NELBO selector), bomev (best-of-M, running
# evidence, the reported selector). Mixture-only policies are skipped for M=1 cells, and an
# existing output file is left alone so an interrupted sweep can be rerun.
#
# Usage:
#   RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 POLICY=cons \
#     CELLS="lm1b_M4_K4_seed0:0050000 lm1b_M8_K4_seed0" bash scripts/eval/eval_cells.sh
#
# A cell without a :step suffix uses its newest checkpoint.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

RESULTS=${RESULTS:?set RESULTS (a results directory holding the cell subdirectories)}
CELLS=${CELLS:?set CELLS (space separated "cell" or "cell:step" specs)}
GPU=${GPU:-0}
POLICY=${POLICY:-commit}
NFE=${NFE:-1,2,3,4,5,6,7,8,16,32}
GEN_SAMPLES=${GEN_SAMPLES:-512}
# The NELBO pass materialises one [B, M, L, V] tensor, which does not fit beside a live trainer
# at the original batch of 64. Shrinking the batch and raising the batch count holds the scored
# sequence total constant, so only the wall clock changes and the reported number does not.
SAMPLE_BATCH=${SAMPLE_BATCH:-32}
GEN_PPL_BATCH=${GEN_PPL_BATCH:-8}
ELBO_BATCHES=${ELBO_BATCHES:-320}
TAG=${TAG:-}
EXTRA=${EXTRA:-}
LOG=${LOG:-$ROOT/logs/eval_cells_gpu$GPU.log}
export WANDB_MODE=disabled
mkdir -p "$(dirname "$LOG")"

case "$POLICY" in
  commit)   FLAGS="--commit_k";                                              NEEDS_MIXTURE=0 ;;
  cons)     FLAGS="--consensus";                                             NEEDS_MIXTURE=1 ;;
  conssoft) FLAGS="--consensus --consensus_soft";                            NEEDS_MIXTURE=1 ;;
  bom)      FLAGS="--no-commit_k --best_of_m";                               NEEDS_MIXTURE=1 ;;
  bomev)    FLAGS="--no-commit_k --best_of_m --best_of_m_selector evidence";  NEEDS_MIXTURE=1 ;;
  *) echo "unknown POLICY '$POLICY' (commit cons conssoft bom bomev)" >&2; exit 2 ;;
esac
# commit-k keeps the historical bare filename so the existing tables and collectors keep resolving.
SUFFIX=${TAG:+_$TAG}
case "$POLICY" in
  commit) OUTNAME="eval_nfegrid_analytic${SUFFIX}.json" ;;
  *)      OUTNAME="eval_nfegrid_analytic_${POLICY}${SUFFIX}.json" ;;
esac

cd "$CODE"
for spec in $CELLS; do
  cell=${spec%%:*}
  step=${spec#*:}
  celldir=$RESULTS/$cell
  if [[ "$step" == "$spec" ]]; then
    CKPT=$(ls -1 "$celldir"/ckpt_*.pt 2>/dev/null | sort | tail -1)
  else
    CKPT=$celldir/ckpt_${step}.pt
  fi
  OUT=$celldir/$OUTNAME
  if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[$POLICY $(date -Is)] no checkpoint for $cell, skipping" | tee -a "$LOG"; continue
  fi
  if [[ -f "$OUT" ]]; then
    echo "[$POLICY $(date -Is)] $cell already scored, skipping" | tee -a "$LOG"; continue
  fi
  # latent_M lives in the checkpoint config, so a factorized cell is detected without a config file.
  M=$("$PY" - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck.get("cfg") or {}
print(int(cfg.get("latent_M", 1) or 1))
PY
)
  M=${M:-1}
  if [[ "$NEEDS_MIXTURE" == 1 && "$M" -le 1 ]]; then
    echo "[$POLICY $(date -Is)] $cell is factorized (M=$M), policy needs a mixture, skipping" | tee -a "$LOG"
    continue
  fi
  echo "[$POLICY $(date -Is)] $cell M=$M nfe=$NFE -> $OUT" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m eval.eval_text \
    --ckpt "$CKPT" --out "$OUT" --nfe "$NFE" \
    --gen_samples "$GEN_SAMPLES" --sample_batch "$SAMPLE_BATCH" \
    --gen_ppl_batch "$GEN_PPL_BATCH" --elbo_batches "$ELBO_BATCHES" \
    --sampler analytic $FLAGS $EXTRA \
    >> "$LOG" 2>&1
  echo "[$POLICY $(date -Is)] $cell rc=$?" | tee -a "$LOG"
done
