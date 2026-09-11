"""Text dataloaders for the latent-kernel flow (LM1B, text8, wikitext103, openwebtext).

Reads the same pre-tokenized HuggingFace `save_to_disk` Arrow caches the baselines use
(`$MLDF_TEXT_ROOT/<ds>/<ds>_<split>_bs<L>_{wrapped,unwrapped}.dat`), so the student trains
on the identical tokenization as MDLM and ReDi, which makes the comparison same-tokenizer and fair. Each row is a
length-L `input_ids` sequence; we expose it as a `[L]` int64 tensor plus a `meta` dict carrying
`vocab_size` (data vocab, before the +1 MASK the trainer appends), `seq_len`, and the `tokenizer`
name so the eval harness can decode samples for generative PPL.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import datasets as hfds


from paths import TEXT_ROOT

# name -> (subdir file stem template, block size L, tokenizer id, data vocab size or None=scan)
#   {split} is filled with the HF split token used when the cache was built.
_REGISTRY = {
    # NOTE lm1b is UNWRAPPED (one padded sentence per row): ~76.5% of tokens are [PAD]=0.
    # `pad_id` is surfaced in meta so pad-aware losses can exclude those positions the way
    # MDLM's `_loss` multiplies by attention_mask. Wrapped caches have no pads (pad_id=-1).
    "lm1b": dict(stem="lm1b_{split}_bs128_unwrapped", L=128,
                 tokenizer="bert-base-uncased", vocab=30522, pad_id=0,
                 splits={"train": "train", "val": "test", "test": "test"}),
    # Sample-level distillation corpora generated from LKF-M8 (scripts/gen_bestof_corpus.py):
    #   lm1b_bo8m8 = best-of-8 selected @NFE=32 (the quality lane, teacher gen-PPL 105).
    #   lm1b_ckm8  = commit-k @NFE=128 self-samples (no cross-component selection; control).
    # Same tokenizer/vocab/pad as unwrapped lm1b so a single-pass student trains identically.
    "lm1b_bo8m8": dict(stem="lm1b_bo8m8_{split}_bs128_wrapped", L=128,
                       tokenizer="bert-base-uncased", vocab=30522, pad_id=0,
                       splits={"train": "train", "val": "test", "test": "test"}),
    "lm1b_ckm8": dict(stem="lm1b_ckm8_{split}_bs128_wrapped", L=128,
                      tokenizer="bert-base-uncased", vocab=30522, pad_id=0,
                      splits={"train": "train", "val": "test", "test": "test"}),
    "text8": dict(stem="text8_{split}_bs256_wrapped", L=256,
                  tokenizer="char", vocab=None,  # scanned (char-level, ~27-35)
                  splits={"train": "train", "val": "test", "test": "test"}),
    "wikitext103": dict(stem="wikitext103_{split}_bs128_wrapped", L=128,
                        tokenizer="gpt2", vocab=50257,
                        splits={"train": "train", "val": "validation", "test": "validation"}),
    "openwebtext": dict(stem="openwebtext-train_{split}_bs128_wrapped", L=128,
                        tokenizer="gpt2", vocab=50257,
                        splits={"train": "train", "val": "valid", "test": "valid"}),
}


def is_text_dataset(name: str) -> bool:
    return name in _REGISTRY


class HFTextDataset(Dataset):
    """Wrap a pre-tokenized HF Arrow dataset, yielding `[L]` int64 input_ids."""

    def __init__(self, dpath: Path) -> None:
        self.ds = hfds.load_from_disk(str(dpath)).with_format("numpy", columns=["input_ids"])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.ds[i]["input_ids"].astype(np.int64))


def _scan_vocab(ds: HFTextDataset, n: int = 20000) -> int:
    m = 0
    for i in range(min(n, len(ds))):
        m = max(m, int(ds[i].max()))
    return m + 1


def load_text(name: str, split: str = "train",
              root: Path = TEXT_ROOT) -> tuple[HFTextDataset, dict]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown text dataset {name!r}; known={list(_REGISTRY)}")
    spec = _REGISTRY[name]
    hf_split = spec["splits"][split]
    stem = spec["stem"].format(split=hf_split)
    dpath = Path(root) / name / f"{stem}.dat"
    if not dpath.exists():
        raise FileNotFoundError(f"missing text cache {dpath} (build it via the baseline data prep)")
    ds = HFTextDataset(dpath)
    vocab = spec["vocab"] if spec["vocab"] is not None else _scan_vocab(ds)
    meta = {
        "vocab_size": int(vocab),
        "seq_len": int(spec["L"]),
        "tokenizer": spec["tokenizer"],
        "dir": str(dpath),
        "is_text": True,
        "pad_id": int(spec.get("pad_id", -1)),
    }
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2, root: Path = TEXT_ROOT) -> tuple[DataLoader, dict]:
    ds, meta = load_text(name, split, root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return loader, meta


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def decode(name: str, ids: np.ndarray) -> list[str]:
    """Decode `[B, L]` (or `[L]`) token ids back to text for generative-PPL scoring."""
    spec = _REGISTRY[name]
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None]
    tok_name = spec["tokenizer"]
    if tok_name == "char":
        # text8 char map: ids are ' '=?, a-z; reconstruct via the standard text8 alphabet.
        # (ids were built by the baseline char tokenizer; index 0 reserved.) Best-effort:
        alphabet = " abcdefghijklmnopqrstuvwxyz"
        out = []
        for row in ids:
            out.append("".join(alphabet[c] if 0 <= c < len(alphabet) else "" for c in row))
        return out
    tok = AutoTokenizer.from_pretrained(tok_name)
    specials = set(tok.all_special_ids)
    out = []
    for row in ids:
        keep = [int(c) for c in row if int(c) not in specials]
        out.append(tok.decode(keep, skip_special_tokens=True).strip())
    return out
