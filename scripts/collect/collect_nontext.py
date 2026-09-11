"""Gather the molecule and DNA arms into one table.

The two arms write a `train_log.json` for the training curve and an `eval.json` for the NFE
sweep, one pair per cell, under `results/mol_distill` and `results/dna_distill`. This script
reads both and prints a status line for every cell together with the numbers the paper tables
need, so that a single command answers what has finished, what is still running, and what the
mixture is buying.

Usage:
    python scripts/collect_nontext.py                    # status plus the headline table
    python scripts/collect_nontext.py --nfe 8            # pick a column of the NFE sweep
    python scripts/collect_nontext.py --csv out.csv      # also write the long form
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = paths.PROJECT_DIR
DIRS = [ROOT / "results" / "mol_distill", ROOT / "results" / "dna_distill"]
TEACHERS = ROOT / "results" / "mol_teacher"
GRID = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]
# The manuscript tables carry the same dense grid out to 32, which is where every student row
# has gone flat. Only the working document keeps the 64, 128 and 256 columns, and the
# paragraphs beside the manuscript tables quote that tail in prose instead.
PAPER_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
MOL_KEYS = ["valid_pct", "unique_pct", "novel_pct"]
DNA_KEYS = ["fbd", "w1_dev", "w1_hk", "pi90_dev"]
# A cell with a latent wider than one can be decoded under more than one policy, and each
# policy writes its own sweep beside the default one. The order here is the order the policy
# panel prints its rows in.
POLICIES = {"commit": ("eval.json", "commit-$k$", "commit-k"),
            "bom": ("eval_bom.json", "best-of-$M$", "best-of-M"),
            "routed": ("eval_routed.json", "routed", "routed")}


def parse_cell(name: str) -> dict:
    """Split a cell directory name into the fields the tables are keyed by."""
    corpus = name.split("_mdlmT_")[0]
    objective = "di4c" if "_di4c_" in name else "exact"
    m = re.search(r"_M(\d+)_", name)
    k = re.search(r"_K(\d+)_", name)
    s = re.search(r"_seed(\d+)", name)
    return {"corpus": corpus, "objective": objective,
            "M": int(m.group(1)) if m else None,
            "K": int(k.group(1)) if k else None,
            "seed": int(s.group(1)) if s else None}


def last_train_entry(cell_dir: Path) -> dict | None:
    path = cell_dir / "train_log.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    rows = blob if isinstance(blob, list) else blob.get("logs", [])
    return rows[-1] if rows else None


def eval_at(cell_dir: Path, nfe: int, policy: str = "commit") -> dict:
    """The metrics recorded at one NFE under one decoding policy, as a plain mapping."""
    path = cell_dir / POLICIES[policy][0]
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    out = {}
    for row in blob.get("metrics", []):
        if row.get("nfe") == nfe:
            out[row["metric"]] = row["value"]
    return out


def queue_state(cell: str) -> str:
    for state in ("running", "failed", "done"):
        if list((ROOT / "queue").glob(f"gpu*/{state}/*_{cell}.job")):
            return state
    if list((ROOT / "queue").glob(f"gpu*/*_{cell}.job")):
        return "queued"
    return "unknown"


LABEL = {("teacher", None): "teacher (MDLM, ours)",
         ("exact", 1): "MLDF, $M{=}1$ (control)",
         ("exact", 4): "MLDF, $M{=}4$",
         ("exact", 8): "MLDF, $M{=}8$",
         ("di4c", 4): "Di4C, $M{=}4$",
         ("di4c", 8): "Di4C, $M{=}8$"}
BLOCKS = {None: 12, 1: 12, 4: 24, 8: 40}
# The order the rows appear in within one corpus block of every table.
ORDER = [("teacher", None), ("di4c", 4), ("di4c", 8),
         ("exact", 1), ("exact", 4), ("exact", 8)]
# The policy panel drops the teacher and the factorized control, which have no policy to
# choose, and it leads with the exact mixture because that is the row the panel argues about.
POL_ORDER = [("exact", 4), ("exact", 8), ("di4c", 4), ("di4c", 8)]


def tex(v, p=2, scale=1.0):
    return r"\TBD" if not isinstance(v, (int, float)) else f"${v * scale:.{p}f}$"


def emit_latex(rows: list[dict]) -> None:
    """Print the row bodies of the three tables in the manuscript.

    The rows are pasted into `writeup/mldf.tex` rather than included from a file, because the
    tables carry hand-written reference rows and shading that we do not want a script to own.
    """
    by = {(r["corpus"], r["objective"], r["M"]): r for r in rows}

    print("% ---- tab:mol, valid out of 1024 over the dense NFE grid ----")
    for corpus, title in (("qm9", "(a) QM9"), ("zinc250k", "(b) ZINC-250k")):
        print(f"% {title}")
        for obj, m in ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            counts = " & ".join(tex(r["nfe_valid"].get(n), 1, 1024.0)
                                for n in PAPER_GRID)
            dec = "ancestral" if obj == "teacher" else "commit-$k$"
            mi = "n/a" if obj == "teacher" else ("$0.00$" if m == 1 else tex(r["MI"], 2))
            print(f"{LABEL[(obj, m)]} & {dec} & ${BLOCKS[m]}$ & {counts} & {mi} \\\\")

    print("\n% ---- tab:moldist, distributional panel at NFE 8 ----")
    for corpus, tag in (("qm9", "QM9"), ("zinc250k", "ZINC")):
        for obj, m in ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            cols = " & ".join([tex(r.get("fcd"), 2), tex(r.get("scaf"), 3),
                               tex(r.get("snn"), 3), tex(r.get("intdiv1"), 3),
                               tex(r.get("filters"), 3)])
            print(f"{tag} & {LABEL[(obj, m)]} & ${BLOCKS[m]}$ & {cols} \\\\")

    print("\n% ---- tab:dna, FBD over the dense NFE grid, then W1 and pi90 ----")
    for obj, m in ORDER:
        r = by.get(("deepstarr", obj, m))
        if r is None:
            continue
        fbd = " & ".join(tex(r["nfe_fbd"].get(n), 2) for n in PAPER_GRID)
        dec = "ancestral" if obj == "teacher" else "commit-$k$"
        mi = "n/a" if obj == "teacher" else ("$0.00$" if m == 1 else tex(r["MI"], 2))
        print(f"{LABEL[(obj, m)]} & {dec} & ${BLOCKS[m]}$ & {fbd} & "
              f"{tex(r.get('w1_dev'), 3)} & {tex(r.get('w1_hk'), 3)} & "
              f"{tex(r.get('pi90_dev'), 3)} & {mi} \\\\")


def emit_policy_latex(rows: list[dict]) -> None:
    """Print the row bodies of the decoding-policy panel.

    Only a cell whose latent is wider than one has a policy to choose, because commit-$k$,
    best-of-$M$ and routed decoding all collapse to the same chain at $M=1$. We price each row
    in transformer blocks the way the cost model does, which charges $C(M) = 8 + 4M$ for a call
    that evaluates every component and charges best-of-$M$ another factor of $M$ on top,
    because it runs $M$ complete rollouts before it selects.
    """
    by = {(r["corpus"], r["objective"], r["M"]): r for r in rows}

    print("% ---- tab:molpol, valid out of 1024 by decoding policy ----")
    for corpus, title in (("qm9", "(a) QM9"), ("zinc250k", "(b) ZINC-250k")):
        print(f"% {title}")
        for obj, m in POL_ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            for policy, (_, tex_name, _) in POLICIES.items():
                curve = r.get("pol_valid", {}).get(policy, {})
                if not any(isinstance(v, (int, float)) for v in curve.values()):
                    continue
                blocks = BLOCKS[m] * (m if policy == "bom" else 1)
                counts = " & ".join(tex(curve.get(n), 1, 1024.0) for n in PAPER_GRID)
                name = tex_name.replace("$M$", f"${m}$")
                print(f"{LABEL[(obj, m)]} & {name} & ${blocks}$ & {counts} \\\\")


def emit_markdown(rows: list[dict]) -> None:
    """Print the row bodies of the working tables in `writeup/mldf.md`.

    The working document carries the whole budget grid rather than the five columns the
    manuscript has room for, because the shape of the curve between 1 and 8 evaluations is
    what the discussion around the table argues about.
    """
    by = {(r["corpus"], r["objective"], r["M"]): r for r in rows}

    def md(v, p=1, scale=1024.0):
        return "TBD" if not isinstance(v, (int, float)) else f"{v * scale:.{p}f}"

    print("<!-- 11.4 molecules: valid out of 1024 -->")
    print("| model | decoding | blocks | " +
          " | ".join(f"NFE={n}" for n in GRID) + " |")
    print("| --- " * (3 + len(GRID)) + "|")
    for corpus, tag in (("qm9", "QM9"), ("zinc250k", "ZINC")):
        for obj, m in ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            dec = "ancestral" if obj == "teacher" else "commit-k"
            label = LABEL[(obj, m)].replace("$M{=}", "M=").replace("$", "")
            cells = " | ".join(md(r["nfe_valid"].get(n)) for n in GRID)
            print(f"| {tag} {label} | {dec} | {BLOCKS[m]} | {cells} |")

    print("\n<!-- 11.4 decoding policies: valid out of 1024 -->")
    print("| model | decoding | blocks | " +
          " | ".join(f"NFE={n}" for n in GRID) + " |")
    print("| --- " * (3 + len(GRID)) + "|")
    for corpus, tag in (("qm9", "QM9"), ("zinc250k", "ZINC")):
        for obj, m in POL_ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            label = LABEL[(obj, m)].replace("$M{=}", "M=").replace("$", "")
            for policy, (_, _, md_name) in POLICIES.items():
                curve = r.get("pol_valid", {}).get(policy, {})
                if not any(isinstance(v, (int, float)) for v in curve.values()):
                    continue
                blocks = BLOCKS[m] * (m if policy == "bom" else 1)
                cells = " | ".join(md(curve.get(n)) for n in GRID)
                name = md_name.replace("best-of-M", f"best-of-{m}")
                print(f"| {tag} {label} | {name} | {blocks} | {cells} |")

    print("\n<!-- 11.4 distributional panel at NFE 8 -->")
    print("| model | FCD | scaffold sim | SNN | int. div. | filters | mi_kx |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for corpus, tag in (("qm9", "QM9"), ("zinc250k", "ZINC")):
        for obj, m in ORDER:
            r = by.get((corpus, obj, m))
            if r is None:
                continue
            label = LABEL[(obj, m)].replace("$M{=}", "M=").replace("$", "")
            mi = "n/a" if obj == "teacher" else ("0.00" if m == 1 else md(r["MI"], 2, 1.0))
            cols = " | ".join([md(r.get("fcd"), 2, 1.0), md(r.get("scaf"), 3, 1.0),
                               md(r.get("snn"), 3, 1.0), md(r.get("intdiv1"), 3, 1.0),
                               md(r.get("filters"), 3, 1.0)])
            print(f"| {tag} {label} | {cols} | {mi} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--latex", action="store_true",
                    help="also print the row bodies of the manuscript tables")
    ap.add_argument("--md", action="store_true",
                    help="also print the row bodies of the working tables in mldf.md")
    args = ap.parse_args()

    rows = []
    for base in DIRS:
        for cell_dir in sorted(base.glob("*/")):
            cell = cell_dir.name
            if cell.startswith("_smoke"):
                continue
            row = parse_cell(cell)
            row["cell"] = cell
            row["domain"] = "dna" if base.name.startswith("dna") else "mol"
            last = last_train_entry(cell_dir) or {}
            row["step"] = last.get("step")
            row["nll"] = last.get("nll")
            row["val_nll"] = last.get("val_nll_exact", last.get("val_nll"))
            row["MI"] = last.get("mi_est")
            row["Hrouter"] = last.get("router_entropy")
            ckpts = sorted(cell_dir.glob("ckpt_*.pt"))
            row["ckpt"] = ckpts[-1].name if ckpts else None
            row["train"] = queue_state(cell)
            row["eval"] = queue_state(f"eval_{cell}")
            row.update(eval_at(cell_dir, args.nfe))
            # The tables read one metric across the whole budget grid, so we keep those two
            # curves beside the single-budget columns.
            row["nfe_valid"] = {n: eval_at(cell_dir, n).get("valid_pct") for n in GRID}
            row["nfe_fbd"] = {n: eval_at(cell_dir, n).get("fbd") for n in GRID}
            row["pol_valid"] = {p: {n: eval_at(cell_dir, n, p).get("valid_pct") for n in GRID}
                                for p in POLICIES}
            row["pol_fbd"] = {p: {n: eval_at(cell_dir, n, p).get("fbd") for n in GRID}
                              for p in POLICIES}
            rows.append(row)

    # Every table opens with the teacher row, which is produced by the same sweep but lives
    # under results/mol_teacher and carries no training log of ours.
    for cell_dir in sorted(TEACHERS.glob("*_mdlm")):
        if not (cell_dir / "eval.json").exists():
            continue
        row = {"corpus": cell_dir.name[: -len("_mdlm")], "objective": "teacher",
               "M": None, "K": None, "seed": None, "cell": cell_dir.name,
               "domain": "dna" if cell_dir.name.startswith("deepstarr") else "mol",
               "step": None, "nll": None, "val_nll": None, "MI": None, "Hrouter": None,
               "ckpt": "last.ckpt", "train": "done",
               "eval": queue_state(f"eval_teacher_{cell_dir.name[: -len('_mdlm')]}")}
        row.update(eval_at(cell_dir, args.nfe))
        row["nfe_valid"] = {n: eval_at(cell_dir, n).get("valid_pct") for n in GRID}
        row["nfe_fbd"] = {n: eval_at(cell_dir, n).get("fbd") for n in GRID}
        row["pol_valid"] = {"commit": row["nfe_valid"]}
        row["pol_fbd"] = {"commit": row["nfe_fbd"]}
        rows.append(row)

    if not rows:
        print("[collect] nothing under results/{mol,dna}_distill yet")
        return

    def fmt(v, w, p=4):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

    print(f"{'cell':<42} {'train':<8} {'eval':<8} {'step':>7} {'nll':>9} "
          f"{'val_nll':>9} {'MI':>7} {'Hrout':>7}")
    print("-" * 102)
    for r in rows:
        print(f"{r['cell']:<42} {r['train']:<8} {r['eval']:<8} "
              f"{r['step'] if r['step'] is not None else '-':>7} "
              f"{fmt(r['nll'], 9)} {fmt(r['val_nll'], 9)} "
              f"{fmt(r['MI'], 7)} {fmt(r['Hrouter'], 7, 3)}")

    keys = MOL_KEYS if any(r["domain"] == "mol" for r in rows) else DNA_KEYS
    have = [r for r in rows if any(k in r for k in keys + DNA_KEYS)]
    if have:
        print(f"\nsampling metrics at NFE={args.nfe}")
        cols = [k for k in MOL_KEYS + DNA_KEYS if any(k in r for r in have)]
        print(f"{'cell':<42} " + " ".join(f"{c:>10}" for c in cols))
        print("-" * (42 + 11 * len(cols)))
        for r in have:
            cells = " ".join(f"{r[c]:>10.4f}" if c in r else f"{'-':>10}" for c in cols)
            print(f"{r['cell']:<42} {cells}")
    else:
        print(f"\nno eval.json has an NFE={args.nfe} entry yet")

    if args.latex:
        print()
        emit_latex(rows)
        print()
        emit_policy_latex(rows)

    if args.md:
        print()
        emit_markdown(rows)

    if args.csv:
        for r in rows:
            for key in ("nfe_valid", "nfe_fbd", "pol_valid", "pol_fbd"):
                r.pop(key, None)
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[collect] wrote {args.csv}")


if __name__ == "__main__":
    main()
