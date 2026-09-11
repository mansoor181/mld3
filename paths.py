"""Central path resolution for everything the pipeline reads or writes outside this tree.

All locations resolve from environment variables, falling back to directories next to the
repository. Machine-specific values belong in a `.env` file at the repository root (one
`NAME=value` per line, `#` comments allowed), which is git-ignored and loaded here before the
lookups run. `docs/data.md` describes what lives under each root and which script produces it.

Roots:
  MLDF_TEXT_ROOT      tokenized text corpora (LM1B, WikiText-103)
  MLDF_MOL_ROOT       molecule corpora (QM9, ZINC-250k preprocessed dumps)
  MLDF_DNA_ROOT       DNA corpus root (DeepSTARR)
  MLDF_IMAGE_ROOT     image arm: tokens, teacher, reference features, results
  MLDF_BASELINES      checkouts of the baseline repositories (mdlm, pairflow, di4c, redi)
  MLDF_RESULTS        run output directories
  MLDF_TEACHERS       teacher checkpoints trained with the baseline repositories
  MLDF_DI4C_RESULTS   evaluation output of the released Di4C pipeline, for the comparison figure
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_DIR.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


_load_dotenv(REPO_DIR / ".env")


def _root(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


TEXT_ROOT = _root("MLDF_TEXT_ROOT", str(PROJECT_DIR / "data" / "text"))
MOL_ROOT = _root("MLDF_MOL_ROOT", str(PROJECT_DIR / "data" / "mols"))
DNA_ROOT = _root("MLDF_DNA_ROOT", str(PROJECT_DIR / "data" / "dna"))
IMAGE_ROOT = _root("MLDF_IMAGE_ROOT", str(PROJECT_DIR / "data" / "imagenet256"))
BASELINES_DIR = _root("MLDF_BASELINES", str(PROJECT_DIR / "baselines"))
RESULTS_ROOT = _root("MLDF_RESULTS", str(PROJECT_DIR / "results"))
TEACHERS_ROOT = _root("MLDF_TEACHERS", str(PROJECT_DIR / "teachers"))
DI4C_RESULTS = os.environ.get("MLDF_DI4C_RESULTS", str(PROJECT_DIR / "results" / "di4c_released"))

# Source trees inside the baselines checkout (see docs/data.md for upstreams and commits).
MDLM_DIR = str(BASELINES_DIR / "mdlm")
PAIRFLOW_DIR = str(BASELINES_DIR / "pairflow")
MASKGIT_DIR = str(BASELINES_DIR / "di4c" / "maskgit-pytorch")
REDI_IMAGE_DIR = str(BASELINES_DIR / "redi" / "image")

# Weights & Biases; both empty by default so wandb falls back to the caller's own account.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mldf")

def config_vars() -> dict:
    """Roots a config file may reference as ${NAME}; training.common.load_config expands them."""
    return {
        "RESULTS": str(RESULTS_ROOT),
        "TEACHERS": str(TEACHERS_ROOT),
        "IMAGE_ROOT": str(IMAGE_ROOT),
        "BASELINES": str(BASELINES_DIR),
        "PROJECT": str(PROJECT_DIR),
    }
