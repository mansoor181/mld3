"""External paths, resolved from the environment with a `.env` file at the repository root.

Roots (all optional; each falls back to a directory beside the repository):
  MLD3_TEXT_ROOT    tokenized LM1B and WikiText-103 caches
  MLD3_MOL_ROOT     QM9 and ZINC-250k caches
  MLD3_DNA_ROOT     DeepSTARR cache
  MLD3_IMAGE_ROOT   ImageNet-256 tokens, reference features and results
  MLD3_BASELINES    checkouts of the teacher repositories (mdlm, pairflow, maskgit, redi)
  MLD3_RESULTS      run outputs
  MLD3_TEACHERS     teacher checkpoints
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_DIR.parent

env_file = REPO_DIR / ".env"
if env_file.is_file():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


def _root(name: str, *default: str) -> Path:
    return Path(os.environ.get(name, str(PROJECT_DIR.joinpath(*default))))


TEXT_ROOT = _root("MLD3_TEXT_ROOT", "data", "text")
MOL_ROOT = _root("MLD3_MOL_ROOT", "data", "mols")
DNA_ROOT = _root("MLD3_DNA_ROOT", "data", "dna")
IMAGE_ROOT = _root("MLD3_IMAGE_ROOT", "data", "imagenet256")
BASELINES_DIR = _root("MLD3_BASELINES", "baselines")
RESULTS_ROOT = _root("MLD3_RESULTS", "results")
TEACHERS_ROOT = _root("MLD3_TEACHERS", "teachers")

MDLM_DIR = str(BASELINES_DIR / "mdlm")
PAIRFLOW_DIR = str(BASELINES_DIR / "pairflow")
MASKGIT_DIR = str(BASELINES_DIR / "maskgit")
REDI_IMAGE_DIR = str(BASELINES_DIR / "redi" / "image")


def config_vars() -> dict:
    """Roots a config may reference as ${NAME}."""
    return {"RESULTS": str(RESULTS_ROOT), "TEACHERS": str(TEACHERS_ROOT),
            "IMAGE_ROOT": str(IMAGE_ROOT), "BASELINES": str(BASELINES_DIR)}
