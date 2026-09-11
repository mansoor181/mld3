import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

RES = str(paths.RESULTS_ROOT)
#!/usr/bin/env python3
"""Collect the analytic molecule evaluations into one comparison table per corpus.

Every molecule student was originally decoded with the ancestral sampler while both teachers were
decoded analytically, so the published tables compared two different decoders. This script reads
the re-scored `eval_analytic.json` files, falls back to the old ancestral `eval.json` when a cell
has not been re-scored yet, and prints validity against NFE together with the distribution
metrics that the harness only computes at NFE 8.
"""
import argparse
import collections
import json
import os

TEACH = RES + "/mol_teacher_eval/{c}_eval.json"
ARMS = [
    ("v1", RES + "/mol_distill_v1", "{c}_mdlmT_{m}_K4_seed0",
     ["M1", "M4", "M8", "di4c_M4", "di4c_M8"]),
    ("v2-R2", RES + "/v2_nontext_sub4", "{c}_mdlmT_r2_{m}",
     ["M1", "M4", "M8"]),
    ("v2-C2", RES + "/v2_nontext", "{c}_mdlmT_c2_{m}",
     ["M1", "M4", "M8"]),
]
NFES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    g = collections.defaultdict(dict)
    for m in d["metrics"]:
        g[m["metric"]][m["nfe"]] = m["value"]
    return d.get("meta", {}), g


def cell_rows(corpus):
    rows = []
    t = load(TEACH.format(c=corpus))
    if t:
        rows.append(("teacher", "MDLM", t[0], t[1]))
    for arm, root, pat, ms in ARMS:
        for m in ms:
            d = os.path.join(root, pat.format(c=corpus, m=m))
            got = load(os.path.join(d, "eval_analytic.json"))
            if got is None:
                got = load(os.path.join(d, "eval.json"))
            if got is None:
                continue
            rows.append((arm, m, got[0], got[1]))
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
        print("\n=== %s : validity (%%) against NFE, analytic sampler unless marked anc" % corpus)
        print("%-8s %-9s %-4s %s" % ("arm", "cell", "smp", " ".join("%6d" % n for n in NFES)))
        for arm, m, meta, g in rows:
            smp = "anc" if meta.get("sampler") == "ancestral" else "ana"
            vals = [g["valid_pct"].get(n) for n in NFES]
            print("%-8s %-9s %-4s %s" % (arm, m, smp,
                                         " ".join(fmt(v) for v in vals)))
        print("\n=== %s : quality at NFE 8 and likelihood" % corpus)
        cols = ["fcd", "scaf", "snn", "frag", "filters", "intdiv1"]
        print("%-8s %-9s %-4s %8s %8s %s" % ("arm", "cell", "smp", "nll", "uniq@8",
                                             " ".join("%8s" % c for c in cols)))
        for arm, m, meta, g in rows:
            smp = "anc" if meta.get("sampler") == "ancestral" else "ana"
            nll = g.get("nll_per_tok", {}).get(None)
            uq = g.get("unique_canon_pct", {}).get(8)
            print("%-8s %-9s %-4s %s %s %s" % (
                arm, m, smp, fmt(nll, 8, 4), fmt(uq, 8),
                " ".join(fmt(g.get(c, {}).get(8), 8) for c in cols)))


if __name__ == "__main__":
    main()
