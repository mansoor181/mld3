"""Re-express the DeepSTARR cache in the form PairFlow's trainer expects.

We train the DNA teacher with the same fork and the same recipe as the two molecule teachers,
because that is what makes the teacher row of the three arms comparable and what lets the warm
start in models/warm_start.py copy the teacher block for block. PairFlow's loader is generic. It
calls `datasets.load_from_disk` on `<dataset_path>/{train,valid}` and reads an `x_clean` column
of token ids beside an `attention_mask`, so the only work is a format change.

The trainer also needs a tokenizer, from which it takes the vocabulary size and the mask id and
nothing else. We write a five-nucleotide word-level tokenizer with the mask as the sixth entry,
which reproduces the ids our own cache already uses.

Usage:
    python scripts/prep_deepstarr_pairflow.py
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
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import datasets as hfds

SRC = str(paths.DNA_ROOT / "deepstarr")
DST = str(paths.MOL_ROOT / "deepstarr" / "parsed" / "base")
TOKDIR = str(paths.DNA_ROOT / "deepstarr" / "tokenizer")


def write_tokenizer(alphabet: list[str], out: Path) -> None:
    """A word-level tokenizer over the nucleotides with the mask appended last."""

    vocab = {c: i for i, c in enumerate(alphabet)}
    vocab["[MASK]"] = len(alphabet)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="N"))
    # The sequences arrive as bare strings of nucleotides, so a character split is the
    # pre-tokenization we want.
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    out.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, mask_token="[MASK]",
                                  unk_token="N", model_max_length=249)
    tok.save_pretrained(str(out))
    # The mask has to live inside the base vocabulary rather than among the added tokens,
    # because PairFlow reads `tokenizer.vocab_size`, which excludes added tokens.
    check = PreTrainedTokenizerFast.from_pretrained(str(out))
    if check.vocab_size != len(alphabet) + 1 or check.mask_token_id != len(alphabet):
        raise SystemExit(f"tokenizer mismatch: vocab_size={check.vocab_size} "
                         f"mask_token_id={check.mask_token_id}")
    print(f"[prep] wrote {out} (vocab_size={check.vocab_size}, "
          f"mask_token_id={check.mask_token_id})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--tokenizer_dir", default=TOKDIR)
    args = ap.parse_args()


    src = Path(args.src)
    meta = json.loads((src / "meta.json").read_text())
    write_tokenizer(meta["alphabet"], Path(args.tokenizer_dir))

    dst = Path(args.dst)
    for split, name in (("train", "train"), ("valid", "valid")):
        blob = torch.load(src / f"{split}.pt")
        x = blob["x_clean"].to(torch.int64).numpy()
        n, length = x.shape
        table = {"x_clean": list(x),
                 "attention_mask": list(np.ones((n, length), dtype=np.int8))}
        ds = hfds.Dataset.from_dict(table)
        # PairFlow's caches carry the torch format on disk, and its trainer relies on it,
        # because `nll` reads `x0.shape` straight off the collated batch.
        ds.set_format("torch", columns=["x_clean", "attention_mask"])
        ds.save_to_disk(str(dst / name))
        print(f"[prep] wrote {dst / name}  n={n} length={length}")

    print(f"[prep] alphabet={meta['alphabet']} mask_id={meta['mask_id']} "
          f"vocab_size={meta['vocab_size']}")


if __name__ == "__main__":
    main()
