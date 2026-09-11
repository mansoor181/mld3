"""Build the DeepSTARR regulatory-DNA cache for the latent-kernel flow.

Downloads the Drosophila STARR-seq enhancer-activity corpus of de Almeida et al. (2022)
from the HuggingFace mirror `GenerTeam/DeepSTARR-enhancer-activity`, encodes each 249 bp
sequence over a nucleotide alphabet, and writes `{train,valid,test}.pt` plus a `meta.json`
next to them.

The alphabet is built from the data rather than assumed, because an unexpected ambiguity
code silently folded into `A` would corrupt every downstream number. The absorbing MASK is
appended once past the last nucleotide, which keeps it inside the model vocabulary in the
same way the SMILES tokenizers do, so the trainer never extends the vocabulary itself.

Usage:
    python scripts/prep/prep_deepstarr.py [--out $MLDF_DNA_ROOT/deepstarr]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import datasets as hfds

HF_ID = "GenerTeam/DeepSTARR-enhancer-activity"
SEQ_LEN = 249
LABEL_COLS = ("Dev_log2_enrichment", "Hk_log2_enrichment")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.DNA_ROOT / "deepstarr"))
    ap.add_argument("--hf_id", default=HF_ID)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dsd = hfds.load_dataset(args.hf_id)
    print(f"[prep] splits: { {k: len(v) for k, v in dsd.items()} }")

    # Build the alphabet from a sample of the training split, then verify it covers everything.
    counter: Counter[str] = Counter()
    probe = dsd["train"].select(range(min(20000, len(dsd["train"]))))
    for s in probe["sequence"]:
        counter.update(s)
    alphabet = sorted(counter)
    print(f"[prep] observed characters: {dict(counter)}")
    if alphabet != ["A", "C", "G", "T"]:
        print(f"[prep] WARNING: alphabet is {alphabet}, not the plain four nucleotides")

    stoi = {c: i for i, c in enumerate(alphabet)}
    mask_id = len(alphabet)
    vocab_size = len(alphabet) + 1
    print(f"[prep] alphabet={alphabet} mask_id={mask_id} vocab_size={vocab_size}")

    lut = np.full(256, -1, dtype=np.int64)
    for c, i in stoi.items():
        lut[ord(c)] = i

    written = {}
    for split, key in (("train", "train"), ("valid", "validation"), ("test", "test")):
        d = dsd[key]
        seqs = d["sequence"]
        n = len(seqs)
        x = np.empty((n, SEQ_LEN), dtype=np.int64)
        for i, s in enumerate(seqs):
            if len(s) != SEQ_LEN:
                raise ValueError(f"{key}[{i}] has length {len(s)}, expected {SEQ_LEN}")
            row = lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
            if (row < 0).any():
                bad = {s[j] for j in np.nonzero(row < 0)[0]}
                raise ValueError(f"{key}[{i}] holds characters outside the alphabet: {bad}")
            x[i] = row
        y = np.stack([np.asarray(d[c], dtype=np.float32) for c in LABEL_COLS], axis=1)
        path = out / f"{split}.pt"
        torch.save({"x_clean": torch.from_numpy(x).to(torch.uint8),
                    "y": torch.from_numpy(y)}, path)
        written[split] = n
        print(f"[prep] wrote {path}  x={x.shape}  y={y.shape}")

    meta = {
        "hf_id": args.hf_id,
        "seq_len": SEQ_LEN,
        "alphabet": alphabet,
        "vocab_size": vocab_size,
        "mask_id": mask_id,
        "label_cols": list(LABEL_COLS),
        "rows": written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep] wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
