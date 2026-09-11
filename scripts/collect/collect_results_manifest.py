#!/usr/bin/env python3
"""Gather every eval JSON, local and remote, into one tidy table.

Arms that trained on other machines are pulled with --pull, which rsyncs their JSONs into
results/_collected/ and then walks the local roots together with that mirror. Remotes come from
the MLDF_REMOTE_RESULTS environment variable as space-separated host:path pairs, and extra local
roots from MLDF_EXTRA_RESULTS, so no machine name is baked in here. Without --pull the script
walks whatever is already on disk, which lets a notebook re-run offline.

The output is results/_collected/manifest.parquet, one row per (file, metric, nfe) with the arm,
policy and provenance already decoded, plus manifest.csv for anything that cannot read parquet.

    python3 scripts/collect/collect_results_manifest.py --pull
"""
import argparse
import json
import os
import re
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths  # noqa: E402

COLLECT = str(paths.RESULTS_ROOT / "_collected")

# Local roots to walk. The image arm keeps its results beside its data, so both are listed.
LOCAL_ROOTS = [(str(paths.RESULTS_ROOT), "local"), (str(paths.IMAGE_ROOT / "results"), "local")]
for extra in os.environ.get("MLDF_EXTRA_RESULTS", "").split():
    LOCAL_ROOTS.append((extra, "local"))

# Remote result roots as host:path pairs, e.g. "gpuhost:/data/results otherhost:/scratch/results".
REMOTES = []
for spec in os.environ.get("MLDF_REMOTE_RESULTS", "").split():
    host, _, root = spec.partition(":")
    if host and root:
        REMOTES.append((host, root))

# Which host trained each corpus, for the provenance column; unlisted corpora count as local.
CORPUS_HOST = {}
for spec in os.environ.get("MLDF_CORPUS_HOSTS", "").split():
    ds, _, host = spec.partition(":")
    if ds and host:
        CORPUS_HOST[ds] = host

# Some corpora were run twice, once under the earlier grid-only supervision and once under the
# dense time sampling. The two must never be averaged together, so the group directory is decoded
# into an explicit supervision column.
SUPERVISION = {
    "wt103_distill": "grid", "wt103_distill_v2": "dense",
    "lm1b_distill": "grid", "mol_distill_v1": "grid",
    "image_distill": "dense", "v2_nontext": "dense", "v2_nontext_sub4": "dense",
    "v2_nontext_r2sub8": "dense", "v2_ablate": "dense", "dna_distill": "grid",
}

DOMAIN = {
    "lm1b": "text", "wikitext103": "text",
    "qm9": "molecule", "zinc250k": "molecule",
    "deepstarr": "dna", "imagenet256": "image",
}

# The headline quality metric per corpus, all lower-is-better.
PRIMARY = {
    "lm1b": "gen_ppl", "wikitext103": "gen_ppl",
    "qm9": "valid_pct", "zinc250k": "valid_pct",
    "deepstarr": "fbd", "imagenet256": "fid",
}
# True when a larger value is a better model. Molecule validity is the only headline metric of
# the six that runs the other way, and the Frechet distances are only computed at 8 steps.
PRIMARY_HIGHER_BETTER = {"valid_pct": True}

# The diversity metric the gate of Section 12.4 reads, per corpus.
DIVERSITY = {
    "lm1b": "unigram_entropy", "wikitext103": "unigram_entropy",
    "qm9": "intdiv1", "zinc250k": "intdiv1",
    "deepstarr": "unigram_entropy", "imagenet256": "recall",
}


def pull():
    os.makedirs(COLLECT, exist_ok=True)
    for host, root in REMOTES:
        dest = os.path.join(COLLECT, host + root.replace("/", "_"))
        os.makedirs(dest, exist_ok=True)
        # Only the JSONs. A checkpoint is gigabytes and none of it is needed here.
        cmd = [
            "rsync", "-rq", "--prune-empty-dirs",
            "--include=*/", "--include=eval*.json", "--exclude=*",
            f"{host}:{root}/", dest + "/",
        ]
        print("[collect] " + " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)


def arm_of(cell, latent_M, group=""):
    """Decode which arm a cell directory belongs to.

    The naming is not uniform across corpora, since the arms were added over several months, so
    we match on the substrings that actually appear rather than assuming one scheme.
    """
    c = (cell + " " + group).lower()
    if "teacher" in c or "anchor" in c:
        return "teacher"
    if "di4c" in c:
        return "Di4C"
    if "surr" in c:
        return "surrogate"
    if latent_M is not None:
        return f"MLDF M={latent_M}"
    return "MLDF"


def policy_of(meta, fname):
    """Decode the decoding policy from the meta flags, falling back on the file name.

    Older files predate the consensus flags, so their meta carries only commit_k / best_of_m.
    """
    if meta.get("consensus"):
        return "consensus soft" if meta.get("consensus_soft") else "consensus"
    if meta.get("best_of_m"):
        return "best-of-M"
    if meta.get("commit_k"):
        return "commit-k"
    f = fname.lower()
    if "cons_soft" in f or "consensus_soft" in f:
        return "consensus soft"
    if "cons" in f:
        return "consensus"
    if "bom" in f or "best_of" in f:
        return "best-of-M"
    return "commit-k"


STEP_IN_NAME = re.compile(r"(?:_s|_mid_|_step)0*(\d+)")


def walk(roots):
    rows = []
    for root, default_host in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not (fn.startswith("eval") and fn.endswith(".json")):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    d = json.load(open(path))
                except Exception:
                    continue
                # Two schemas exist. The evaluator writes {"meta", "metrics":[{metric,nfe,value}]},
                # while the image teacher anchor predates it and writes a flat
                # {"dataset", "nfe": {"8": {metric: value}}}. We normalise the second into the
                # first rather than special-casing the anchor everywhere downstream.
                if "metrics" not in d and isinstance(d.get("nfe"), dict):
                    meta = {k: d.get(k) for k in ("dataset", "sampler")}
                    meta["step"] = None
                    meta["gen_samples"] = d.get("n_samples")
                    flat = []
                    for k, sub in d["nfe"].items():
                        for mname, mval in sub.items():
                            flat.append({"metric": mname, "nfe": int(k), "value": mval})
                    d = {"meta": meta, "metrics": flat}
                    lbl = json.load(open(path)).get("label_mode")
                    if lbl:
                        cellsuffix = f" ({lbl})"
                    else:
                        cellsuffix = ""
                else:
                    cellsuffix = ""
                meta = d.get("meta", {})
                ds = meta.get("dataset")
                if not ds or "metrics" not in d:
                    continue
                cell = os.path.basename(dirpath) + cellsuffix
                group = os.path.basename(os.path.dirname(dirpath))
                M = meta.get("latent_M")
                # A step recorded in meta wins over one parsed from the file name, because a
                # mid-training file is named for the checkpoint it scored and meta agrees.
                step = meta.get("step")
                if step is None:
                    m = STEP_IN_NAME.search(fn)
                    step = int(m.group(1)) if m else None
                base = dict(
                    corpus=ds,
                    domain=DOMAIN.get(ds, meta.get("domain", "?")),
                    host=CORPUS_HOST.get(ds, default_host),
                    group=group,
                    cell=cell,
                    file=fn,
                    path=path,
                    supervision=SUPERVISION.get(group, "dense" if "v2" in group else "grid"),
                    arm=arm_of(cell, M, group),
                    latent_M=M if M is not None else (1 if "teacher" not in cell else None),
                    policy=policy_of(meta, fn),
                    sampler=meta.get("sampler"),
                    step=step,
                    seq_len=meta.get("seq_len"),
                    gen_samples=meta.get("gen_samples"),
                    n_params=meta.get("n_params"),
                )
                for x in d["metrics"]:
                    rows.append(dict(base, metric=x["metric"], nfe=x.get("nfe"),
                                     value=x.get("value")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true",
                    help="rsync the remote JSONs before walking")
    args = ap.parse_args()
    if args.pull:
        pull()
    roots = list(LOCAL_ROOTS)
    for host, root in REMOTES:
        p = os.path.join(COLLECT, host + root.replace("/", "_"))
        if os.path.isdir(p):
            roots.append((p, host))
    df = walk(roots)
    if df.empty:
        sys.exit("[collect] no eval JSONs found")
    os.makedirs(COLLECT, exist_ok=True)
    df.to_parquet(os.path.join(COLLECT, "manifest.parquet"), index=False)
    df.to_csv(os.path.join(COLLECT, "manifest.csv"), index=False)
    print(f"[collect] {len(df):,} rows from {df['path'].nunique():,} files")
    print(df.groupby(["domain", "corpus"])["path"].nunique().to_string())
    print("\nby arm:")
    print(df.groupby(["corpus", "arm"])["path"].nunique().to_string())


if __name__ == "__main__":
    main()
