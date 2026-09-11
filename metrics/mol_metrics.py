"""Molecule metrics for the QM9 and ZINC-250k arms.

The three headline numbers, validity, uniqueness and novelty, are computed exactly the way
PairFlow's `main.py::_eval_qm9` computes them, down to the string surgery on the decoded
sequence and the comparison of raw generated SMILES against the reference set of canonical
SMILES. We keep that convention because the point of these tables is to sit next to
PairFlow's, and `tests/test_mol_metrics.py` pins the agreement on a fixed sample file.

The reference convention has a known weakness. A generated molecule is called novel when its
raw SMILES string is absent from a set of canonical SMILES, and two different strings can
name the same molecule, so novelty is inflated. We therefore also report a canonicalised
uniqueness and novelty alongside the reference numbers rather than in place of them.

The distributional panel, Frechet ChemNet distance together with the MOSES internal-diversity
and scaffold statistics, lives behind optional imports because neither `fcd_torch` nor
`molsets` is needed for the headline table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache

import datasets as hfds
import numpy as np

# The chemistry stack is an optional extra (see requirements.txt): rdkit backs every molecule
# metric, while fcd_torch and molsets back only the distributional panel. Guarding the imports
# keeps the text, DNA and image arms runnable without a chemistry install.
try:
    from rdkit import Chem, RDLogger
except ImportError:
    Chem = None

try:
    from fcd_torch import FCD
except ImportError:
    FCD = None

try:
    from moses.metrics import internal_diversity, ScafMetric, SNNMetric, FragMetric
    from moses.metrics.utils import mol_passes_filters
except ImportError:
    internal_diversity = None

_REFERENCE_SETS = {
    "qm9": ("yairschiff/qm9", "train", "canonical_smiles"),
    "zinc250k": ("yairschiff/zinc250k", "train", "canonical_smiles"),
}


def clean_sequences(seqs: list[str]) -> list[str]:
    """Strip the special markers the way the reference evaluation does.

    The reference appends `<eos>` before splitting so that a sample which never emitted one
    is still truncated at the end of the string rather than dropped.
    """
    out = [(s + "<eos>").split("<eos>")[0] for s in seqs]
    return [s.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "") for s in out]


@lru_cache(maxsize=4)
def reference_smiles(name: str) -> frozenset[str]:
    """The set of canonical SMILES the novelty count is measured against."""
    if name not in _REFERENCE_SETS:
        raise KeyError(f"no reference set for {name!r}; known={list(_REFERENCE_SETS)}")
    hf_id, split, col = _REFERENCE_SETS[name]
    return frozenset(hfds.load_dataset(hf_id, split=split)[col])


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


@dataclass
class TrialResult:
    n: int = 0
    valid: int = 0
    unique: int = 0
    novel: int = 0
    invalid: int = 0
    unique_canon: int = 0
    novel_canon: int = 0
    valid_smiles: list[str] = field(default_factory=list)


def score_trial(seqs: list[str], dataset: str, total_samples: int | None = None) -> TrialResult:
    """Score one batch of decoded sequences against the reference conventions."""
    if Chem is None:
        raise ImportError("molecule metrics need rdkit: pip install rdkit")
    RDLogger.DisableLog("rdApp.*")

    cleaned = clean_sequences(seqs)
    if total_samples is not None:
        cleaned = cleaned[:total_samples]
    valids, n_invalid = [], 0
    for s in cleaned:
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None or len(s) == 0:
            n_invalid += 1
        else:
            valids.append(s)

    ref = reference_smiles(dataset)
    canon = {c for c in (canonicalize(s) for s in valids) if c is not None}
    return TrialResult(
        n=len(cleaned), valid=len(valids), unique=len(set(valids)),
        novel=len(set(valids) - ref), invalid=n_invalid,
        unique_canon=len(canon), novel_canon=len(canon - ref),
        valid_smiles=valids,
    )


def aggregate(trials: list[TrialResult]) -> dict:
    """Average the per-trial counts and report percentages the reference way.

    Uniqueness and novelty are percentages of the valid molecules, not of the sample, and a
    trial with no valid molecule contributes zero rather than raising.
    """
    def pct(a: int, b: int) -> float:
        return 0.0 if b == 0 else a / b

    n = float(np.mean([t.n for t in trials]))
    out = {"n_trials": len(trials), "n_samples": n}
    for key in ("valid", "unique", "novel", "invalid", "unique_canon", "novel_canon"):
        vals = np.array([getattr(t, key) for t in trials], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    for key, denom in (("valid", "n"), ("invalid", "n"), ("unique", "valid"),
                       ("novel", "valid"), ("unique_canon", "valid"), ("novel_canon", "valid")):
        vals = np.array([pct(getattr(t, key), t.n if denom == "n" else t.valid)
                         for t in trials], dtype=float)
        out[f"{key}_pct"] = float(vals.mean())
        out[f"{key}_pct_std"] = float(vals.std())
    return out


def evaluate(seqs_per_trial: list[list[str]], dataset: str,
             total_samples: int | None = None) -> dict:
    """Score several independent sample batches and return the aggregated report."""
    trials = [score_trial(s, dataset, total_samples) for s in seqs_per_trial]
    report = aggregate(trials)
    report["dataset"] = dataset
    # The distributional panel is computed on the union of the trials, because the Frechet
    # distance and the nearest-neighbour similarity are unstable on a thousand molecules and
    # the caller should not have to parse the sequences a second time to get them.
    report["valid_smiles"] = [s for t in trials for s in t.valid_smiles]
    return report


# ---------------- distributional panel (optional dependencies) ----------------

def frechet_chemnet_distance(gen_smiles: list[str], ref_smiles: list[str],
                             device: str = "cuda", n_ref: int = 10000) -> float | None:
    """Frechet ChemNet distance between generated and reference molecules.

    Returns None when `fcd_torch` is not installed, because the headline table does not
    depend on it and we would rather report a gap than fail the whole evaluation.
    """
    if FCD is None:
        print("[mol-metrics] fcd_torch not installed; skipping FCD")
        return None
    rng = np.random.default_rng(0)
    ref = list(ref_smiles)
    if len(ref) > n_ref:
        ref = [ref[i] for i in rng.choice(len(ref), n_ref, replace=False)]
    return float(FCD(device=device, n_jobs=1)(ref=ref, gen=list(gen_smiles)))


def moses_panel(gen_smiles: list[str], ref_smiles: list[str]) -> dict:
    """Internal diversity and scaffold statistics from the MOSES benchmark suite."""
    if internal_diversity is None:
        print("[mol-metrics] molsets not installed; skipping the MOSES panel")
        return {}
    out = {"intdiv1": float(internal_diversity(gen_smiles, p=1)),
           "intdiv2": float(internal_diversity(gen_smiles, p=2))}
    for name, cls in (("scaf", ScafMetric), ("snn", SNNMetric), ("frag", FragMetric)):
        try:
            out[name] = float(cls()(gen=gen_smiles, ref=list(ref_smiles)))
        except Exception as exc:
            print(f"[mol-metrics] {name} failed: {exc}")
    # The filter pass rate is the medicinal-chemistry screen of the MOSES panel, and it is a
    # plain fraction rather than a metric object, so it does not go through the loop above.
    try:
        passed = [mol_passes_filters(s) for s in gen_smiles]
        out["filters"] = float(np.mean(passed)) if passed else 0.0
    except Exception as exc:
        print(f"[mol-metrics] filters failed: {exc}")
    return out


def distributional_report(gen_smiles: list[str], dataset: str,
                          device: str = "cuda", n_ref: int = 10000) -> dict:
    ref = list(reference_smiles(dataset))
    out = {"fcd": frechet_chemnet_distance(gen_smiles, ref, device=device, n_ref=n_ref)}
    out.update(moses_panel(gen_smiles, ref))
    return out


__all__ = ["clean_sequences", "reference_smiles", "canonicalize", "TrialResult",
           "score_trial", "aggregate", "evaluate", "frechet_chemnet_distance",
           "moses_panel", "distributional_report", "asdict"]
