"""Build the ImageNet-256 VQ-token cache for the latent-kernel flow.

Flattens the `Jiayi-Pan/imagenet-maskgit-256-vqgan` parquet dump, in which every row is a
16x16 grid of VQGAN f16 codebook indices over a codebook of 1024 together with its ImageNet
class, into the same `{split}.pt` plus `meta.json` layout that `prep_deepstarr.py` writes, so
that `data/images.py` can be a near copy of `data/dna.py`.

The grid is raveled to a length-256 sequence in row-major order. Our trunk has no notion of
two dimensions and learns position from data, hence the ordering only has to be the same one
the MaskGIT teacher uses, which it is, since `MaskTransformer.forward` does exactly
`img_token.view(b, -1)` at `Network/transformer.py:194`.

The absorbing MASK sits inside the model vocabulary at 1024, just past the last codebook
entry, matching the teacher's own visual mask token. We assert that no stored id reaches 1024,
because a corpus that already contained a mask or a BOS id would train a student to reproduce
it and nothing downstream would notice.

Row groups are converted one at a time through the Arrow buffers rather than through
`to_pylist`, because a 640k-row shard is 164 million integers and materialising that as nested
Python lists costs several gigabytes for no benefit.

Usage:
    python3.10 scripts/prep_imagenet256.py
    python3.10 scripts/prep_imagenet256.py --limit 4096   # small cache for a smoke test
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

HF_ID = "Jiayi-Pan/imagenet-maskgit-256-vqgan"
GRID, CODEBOOK, N_CLASSES = 16, 1024, 1000
SEQ_LEN = GRID * GRID
MASK_ID = CODEBOOK
VOCAB_SIZE = CODEBOOK + 1
ROOT = paths.IMAGE_ROOT


def read_split(pattern: str, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (x [N,256] int16, y [N] int64) for every parquet shard matching the pattern."""

    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet matching {pattern}; run scripts/fetch_image_arm.py")
    xs, ys, seen = [], [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            tbl = pf.read_row_group(g, columns=["token", "label"])
            # A ChunkedArray of list<list<int16>>. combine_chunks gives one ListArray, and the
            # two flattens strip the outer row dimension and then the 16 rows of the grid,
            # leaving a contiguous int16 buffer we can view as [n, 256] for free.
            tok = tbl["token"].combine_chunks()
            flat = np.asarray(tok.flatten().flatten())
            n = len(tbl)
            if flat.size != n * SEQ_LEN:
                raise ValueError(f"{f} group {g} holds {flat.size} ids for {n} rows, "
                                 f"expected {n * SEQ_LEN}, so a row is not {GRID}x{GRID}")
            xs.append(flat.reshape(n, SEQ_LEN).astype(np.int16, copy=False))
            ys.append(np.asarray(tbl["label"]).astype(np.int64, copy=False))
            seen += n
            if limit and seen >= limit:
                break
        if limit and seen >= limit:
            break
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if limit:
        x, y = x[:limit], y[:limit]
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "imagenet256" / "_hf"))
    ap.add_argument("--out", default=str(ROOT / "imagenet256"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, stem in (("train", "train"), ("val", "val")):
        x, y = read_split(os.path.join(args.src, "**", f"{stem}-*.parquet"), args.limit)
        lo, hi = int(x.min()), int(x.max())
        if lo < 0 or hi >= CODEBOOK:
            raise ValueError(f"{split} ids span [{lo}, {hi}], outside the codebook [0, "
                             f"{CODEBOOK - 1}]. An id at {MASK_ID} would collide with MASK.")
        cls = int(y.min()), int(y.max())
        if cls[0] < 0 or cls[1] >= N_CLASSES:
            raise ValueError(f"{split} labels span {cls}, outside [0, {N_CLASSES - 1}]")
        path = out / f"{split}.pt"
        torch.save({"x_clean": torch.from_numpy(x), "y": torch.from_numpy(y)}, path)
        written[split] = int(x.shape[0])
        print(f"[prep] wrote {path}  x={tuple(x.shape)} int16  ids=[{lo},{hi}]  "
              f"classes={len(np.unique(y))}  {path.stat().st_size / 2**20:.0f} MB")

    meta = {
        "hf_id": HF_ID,
        "seq_len": SEQ_LEN,
        "grid": GRID,
        "codebook_size": CODEBOOK,
        "vocab_size": VOCAB_SIZE,
        "mask_id": MASK_ID,
        "n_classes": N_CLASSES,
        "tokenizer": "vqgan_f16_1024",
        "rows": written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep] wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
