"""Validity, uniqueness and novelty for QM9 and ZINC-250k.

Computed the way PairFlow computes them, so that the tables sit next to theirs: uniqueness and
novelty are percentages of the valid molecules, and novelty compares the raw generated SMILES
against a set of canonical SMILES. That comparison inflates novelty, since two strings can name
one molecule, so the canonicalised counts are reported alongside rather than instead.
Sequences arrive already stripped of the special markers by `data.decode`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

REFERENCES = {"qm9": "yairschiff/qm9", "zinc250k": "yairschiff/zinc250k"}


@lru_cache(maxsize=4)
def reference_smiles(dataset: str) -> frozenset[str]:
    import datasets as hfds
    return frozenset(hfds.load_dataset(REFERENCES[dataset], split="train")["canonical_smiles"])


@dataclass
class Trial:
    n: int
    valid: int
    unique: int
    novel: int
    unique_canonical: int
    novel_canonical: int


def score(smiles: list[str], dataset: str) -> Trial:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    valid = []
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s) if s else None
        except Exception:
            mol = None
        if mol is not None:
            valid.append(s)

    canonical = set()
    for s in valid:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canonical.add(Chem.MolToSmiles(mol))

    ref = reference_smiles(dataset)
    return Trial(n=len(smiles), valid=len(valid), unique=len(set(valid)),
                 novel=len(set(valid) - ref), unique_canonical=len(canonical),
                 novel_canonical=len(canonical - ref))


def evaluate(trials: list[list[str]], dataset: str) -> dict:
    """Aggregate independent sample batches into the reported percentages."""
    scored = [score(s, dataset) for s in trials]
    out = {"dataset": dataset, "n_trials": len(scored)}
    for key, denom in (("valid", "n"), ("unique", "valid"), ("novel", "valid"),
                       ("unique_canonical", "valid"), ("novel_canonical", "valid")):
        frac = np.array([getattr(t, key) / max(getattr(t, denom), 1) for t in scored])
        out[key] = float(frac.mean())
        out[f"{key}_std"] = float(frac.std())
    return out
