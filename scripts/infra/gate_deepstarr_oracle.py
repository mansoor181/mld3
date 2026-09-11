"""Hard gate on the DeepSTARR arm: does the PyTorch oracle port still predict enhancer activity?

The whole DNA arm rests on a released convolutional regressor whose weights were trained
elsewhere. If the port does not reproduce the published held-out correlation then every
Frechet Biological Distance and every top-decile fraction downstream of it is meaningless, so
we check the correlation before spending any training compute.

The published held-out Pearson correlations for DeepSTARR are about 0.68 on the developmental
head and about 0.74 on the housekeeping head. We treat those values as a floor rather than as
a target, because a port that lost or scrambled weights can only score below them, and the
correlation a reimplementation measures on the full held-out chromosome is routinely a little
above the figure the paper quotes. The gate therefore passes when each head reaches its
published correlation less a small allowance.

Usage:
    python scripts/gate_deepstarr_oracle.py [--n 5000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from data import dna as dnadata            # noqa: E402
from metrics import dna_metrics as DM      # noqa: E402

PUBLISHED = {"dev": 0.68, "hk": 0.74}
TOL = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/deepstarr/oracle_gate.json")
    args = ap.parse_args()

    ds, meta = dnadata.load_dna("deepstarr", "test", with_labels=True)
    n = min(args.n, len(ds))
    idx = np.arange(n)
    ids = ds.x[idx].numpy()
    labels = ds.y[idx].numpy()
    seqs = dnadata.decode("deepstarr", ids, alphabet=meta["alphabet"])
    print(f"[gate] scoring {n} held-out sequences of length {meta['seq_len']}")

    report = DM.calibrate(seqs, labels, device=args.device)
    report["published"] = PUBLISHED
    report["allowance"] = TOL
    ok = True
    for name, target in PUBLISHED.items():
        got = report[f"pearson_{name}"]
        good = got >= target - TOL
        ok = ok and good
        print(f"[gate] pearson_{name} = {got:.4f}  (floor {target - TOL:.2f}, "
              f"published {target:.2f})  {'OK' if good else 'FAIL'}")
    report["pass"] = ok

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[gate] wrote {out}")
    print("[gate] PASS" if ok else "[gate] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
