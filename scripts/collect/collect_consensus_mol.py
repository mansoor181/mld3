import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
#!/usr/bin/env python3
"""Collect the consensus-commit molecule evaluations into one table per corpus.

Each cell was scored under three single-chain policies on a common NFE grid and written to
`eval_policies_analytic.json` next to its checkpoint. This script prints validity against NFE
with one row per (cell, policy), then the quality panel that the harness only computes at
NFE 8, so consensus commit can be read against commit-k inside the same file.
"""
import argparse
import collections
import json
import os

RES = str(paths.RESULTS_ROOT)
ARMS = [
    ("v2", "v2_nontext_sub4", "{c}_mdlmT_r2_{m}", ["M4", "M8"]),
    ("v2", "v2_nontext_sub4", "{c}_mdlmT_di4c_{m}", ["M4", "M8"]),
    ("v1", "mol_distill_v1", "{c}_mdlmT_{m}_K4_seed0", ["M4", "M8"]),
    ("v1", "mol_distill_v1", "{c}_mdlmT_di4c_{m}_K4_seed0", ["M4", "M8"]),
]
POLICIES = ["commit-k", "consensus commit", "consensus commit (soft)"]
SHORT = {"commit-k": "commit-k", "consensus commit": "consensus",
         "consensus commit (soft)": "consensus-s"}
NFES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    g = collections.defaultdict(dict)
    for r in d["policies"]:
        g[r["policy"]][r["nfe"]] = r
    return d.get("meta", {}), g


def cell_rows(corpus):
    rows = []
    for arm, sub, pat, ms in ARMS:
        for m in ms:
            name = pat.format(c=corpus, m=m)
            got = load(os.path.join(RES, sub, name, "eval_policies_analytic.json"))
            if got is None:
                continue
            label = ("di4c " if "di4c" in name else "MLDF ") + m
            rows.append((arm, label, got[1]))
    return rows


def fmt(v, w=6, p=3):
    return "-".rjust(w) if v is None else ("%*.*f" % (w, p, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="qm9,zinc250k")
    args = ap.parse_args()
    for corpus in args.corpora.split(","):
        rows = cell_rows(corpus)
        if not rows:
            continue
        print("\n=== %s : validity (fraction) against NFE, analytic sampler" % corpus)
        print("%-4s %-9s %-12s %s %8s" % ("arm", "cell", "policy",
                                          " ".join("%6d" % n for n in NFES), "blk/step"))
        for arm, label, g in rows:
            for pol in POLICIES:
                if pol not in g:
                    continue
                vals = [g[pol].get(n) for n in NFES]
                per = [r["blocks_per_sample"] / max(n, 1)
                       for n, r in ((n, g[pol][n]) for n in NFES if n in g[pol])]
                print("%-4s %-9s %-12s %s %8.0f" % (
                    arm, label, SHORT[pol],
                    " ".join(fmt(None if v is None else v["valid_pct"]) for v in vals),
                    per[0] if per else 0))
        print("\n=== %s : quality at NFE 8" % corpus)
        cols = ["fcd", "scaf", "snn", "frag", "filters", "intdiv1"]
        print("%-4s %-9s %-12s %8s %8s %s" % ("arm", "cell", "policy", "uniq", "novel",
                                              " ".join("%8s" % c for c in cols)))
        for arm, label, g in rows:
            for pol in POLICIES:
                r = g.get(pol, {}).get(8)
                if r is None:
                    continue
                print("%-4s %-9s %-12s %s %s %s" % (
                    arm, label, SHORT[pol],
                    fmt(r.get("unique_canon_pct"), 8),
                    fmt(r.get("novel_canon_pct"), 8),
                    " ".join(fmt(r.get(c), 8) for c in cols)))


if __name__ == "__main__":
    main()
