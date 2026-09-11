# Shared environment for the shell drivers. Source it, do not execute it:
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"   # from scripts/<group>/
#
# Machine-specific values (data roots, the interpreter, cache locations) belong in a
# git-ignored .env file at the repository root, one NAME=value per line. paths.py reads the
# same file, so the shell drivers and the library always agree. Every assignment below is a
# default: anything already exported by the caller or set in .env wins.

: "${CODE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${ROOT:=$(dirname "$CODE")}"

if [ -f "$CODE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$CODE/.env"
  set +a
fi

: "${PY:=${MLDF_PYTHON:-python3}}"

: "${MLDF_BASELINES:=$ROOT/baselines}"
: "${MLDF_TEXT_ROOT:=$ROOT/data/text}"
: "${MLDF_MOL_ROOT:=$ROOT/data/mols}"
: "${MLDF_DNA_ROOT:=$ROOT/data/dna}"
: "${MLDF_IMAGE_ROOT:=$ROOT/data/imagenet256}"
: "${MLDF_RESULTS:=$ROOT/results}"
: "${MLDF_TEACHERS:=$ROOT/teachers}"
export MLDF_BASELINES MLDF_TEXT_ROOT MLDF_MOL_ROOT MLDF_DNA_ROOT MLDF_IMAGE_ROOT MLDF_RESULTS MLDF_TEACHERS

# HuggingFace caches follow HF_HOME when the caller (or .env) sets one.
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
fi

# Keeps the long training runs from fragmenting GPU memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
