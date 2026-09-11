"""Collect the analytic-sampler re-scores into the row bodies the manuscript tables want.

Every eval json is a {"meta": ..., "metrics": [...]} pair whose metrics field is a list of
records rather than a dict, so a small reader is easier to trust than an ad-hoc jq. Given a
results root and a list of cell names, this prints one line per cell holding the generative
perplexity at each budget on the ten-point grid, the unigram entropy, and the measured
transition information, which is exactly the column order of tab:main.

    python scripts/collect_analytic.py --root results/lm1b_distill \
        --cell lm1b_M1_K4_seed0 --cell lm1b_M4_K4_seed0 --file eval_nfegrid_analytic.json

Pass --latex to get the row bodies instead of the aligned plain text.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

GRID = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]


def read(path: Path):
    d = json.loads(path.read_text())
    by = {}
    scal = {}
    for m in d.get("metrics", []):
        if m.get("nfe") is None:
            scal[m["metric"]] = m["value"]
        else:
            by.setdefault(m["metric"], {})[int(m["nfe"])] = m["value"]
    return d.get("meta", {}), by, scal


def fmt(v, nd=0):
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--file", default="eval_nfegrid_analytic.json")
    ap.add_argument("--grid", default=",".join(str(g) for g in GRID))
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    grid = [int(x) for x in args.grid.split(",")]
    root = Path(args.root)

    if not args.latex:
        head = "cell".ljust(34) + "".join(f"{g:>7}" for g in grid) + "     H1     I    uKL@4"
        print(head)
        print("-" * len(head))
    for cell in args.cell:
        p = root / cell / args.file
        if not p.exists():
            print(f"{cell:<34} (missing {args.file})")
            continue
        meta, by, scal = read(p)
        gp = by.get("gen_ppl", {})
        h1 = scal.get("unigram_entropy", by.get("unigram_entropy", {}).get(4))
        mi = scal.get("mi_k_transition")
        ukl = by.get("unigram_kl", {}).get(4)
        if args.latex:
            cols = " & ".join(f"${gp[g]:.0f}$" if g in gp else r"\TBD" for g in grid)
            print(f"% {cell} step={meta.get('step')} sampler={meta.get('sampler')} "
                  f"bom={meta.get('best_of_m')}")
            print(f"{cols} & ${h1:.2f}$ & ${mi:.2f}$ \\\\" if h1 is not None and mi is not None
                  else f"{cols} \\\\")
        else:
            row = "".join(f"{gp[g]:>7.0f}" if g in gp else "      -" for g in grid)
            print(f"{cell:<34}{row}  {fmt(h1, 2):>6} {fmt(mi, 2):>5} {fmt(ukl, 2):>7}")


if __name__ == "__main__":
    main()
