#!/usr/bin/env bash
# Score a finished ImageNet-256 student on the full NFE grid under each decoding policy.
#
# Commit-k and consensus cost the same wall time here (the trunk emits all M components in one
# forward pass), about two hours per cell for the 10k-sample grid. Best-of-M runs R rollouts to
# completion and costs 12x, so it gets a quarter of the samples; the table caption says so.
#
# Usage:
#   GPU=1 CELLS="imagenet256_maskgitT_r2_M8" bash scripts/eval/eval_image_cells.sh
#   GPU=1 CELLS="a b" STEP=0014000 TAG=s14k POLICIES="commit cons" bash scripts/eval/eval_image_cells.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

RES=${RES:-$MLDF_IMAGE_ROOT/results/image_distill}
CELLS=${CELLS:?set CELLS (space separated cell directory names)}
GPU=${GPU:-1}
NFE=${NFE:-"1,2,3,4,5,6,7,8,16,32"}
STEP=${STEP:-0020000}
GEN_SAMPLES=${GEN_SAMPLES:-10000}
BOM_SAMPLES=${BOM_SAMPLES:-2500}
SB=${SB:-250}
CHUNK=${CHUNK:-64}
POLICIES=${POLICIES:-"commit cons bom"}
# A tag keeps an intermediate-checkpoint sweep from overwriting the final one. We score partway
# through training to watch the trend and to check the ordering early, and those files have to
# survive next to the 20k results rather than replace them.
TAG=${TAG:-}
SUF=${TAG:+_$TAG}
LOG=${LOG:-$ROOT/logs/image/eval_image.log}
export WANDB_MODE=disabled
mkdir -p "$(dirname "$LOG")"
cd "$CODE"

for cell in $CELLS; do
  CKPT=$RES/$cell/ckpt_${STEP}.pt
  if [[ ! -f "$CKPT" ]]; then
    echo "[img-eval $(date -Is)] missing $CKPT, skipping $cell" | tee -a "$LOG"
    continue
  fi
  # A factorized student has no components to search over, so the mixture policies reduce to
  # commit-k and we skip them rather than write three copies of the same table row. The mixture
  # size is read from the checkpoint itself, which stays correct however the configs are laid out.
  M=$("$PY" - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int((ck.get("cfg") or {}).get("latent_M", 1) or 1))
PY
)
  M=${M:-1}
  for pol in $POLICIES; do
    N=$GEN_SAMPLES
    case $pol in
      commit) FLAGS="--commit_k";;
      cons)   FLAGS="--consensus";;
      bom)    FLAGS="--best_of_m"; N=$BOM_SAMPLES;;
      *) echo "[img-eval] unknown policy $pol" | tee -a "$LOG"; continue;;
    esac
    if [[ "$pol" != commit && "$M" -le 1 ]]; then
      echo "[img-eval $(date -Is)] $cell is factorized (M=$M), skipping $pol" | tee -a "$LOG"
      continue
    fi
    OUT=$RES/$cell/eval_nfegrid_analytic_${pol}${SUF}.json
    if [[ -f "$OUT" ]]; then
      echo "[img-eval $(date -Is)] $cell $pol already scored, skipping" | tee -a "$LOG"
      continue
    fi
    # The trainer keeps only the newest checkpoints, so one can disappear between policies.
    if [[ ! -f "$CKPT" ]]; then
      echo "[img-eval $(date -Is)] $CKPT vanished (trainer rotated it), skipping $cell $pol" | tee -a "$LOG"
      continue
    fi
    echo "[img-eval $(date -Is)] $cell pol=$pol M=$M nfe=$NFE n=$N -> $OUT" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m eval.eval_text \
      --ckpt "$CKPT" --out "$OUT" --nfe "$NFE" \
      --gen_samples "$N" --sample_batch "$SB" --elbo_batches 40 \
      --sampler analytic $FLAGS --image_chunk "$CHUNK" \
      >> "$LOG" 2>&1
    rc=$?
    echo "[img-eval $(date -Is)] $cell pol=$pol rc=$rc" | tee -a "$LOG"
  done
done
