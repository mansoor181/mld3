"""Regulatory-DNA dataloader for the latent-kernel flow (DeepSTARR enhancers).

Reads the cache that `scripts/prep_deepstarr.py` writes, in which every row is a length-249
nucleotide sequence and every label is the pair of developmental and housekeeping log2
enrichments. The absorbing MASK sits inside the model vocabulary at the index just past the
last nucleotide, so the trainer takes `meta["mask_id"]` as given and does not extend the
vocabulary.

Sequences carry no padding, so `meta["pad_id"]` is -1 and every position is modelled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import DNA_ROOT

_REGISTRY = {
    "deepstarr": dict(subdir="deepstarr",
                      splits={"train": "train", "val": "valid", "test": "test"}),
}


def is_dna_dataset(name: str) -> bool:
    return name in _REGISTRY


class DNADataset(Dataset):
    """Yield `[L]` int64 nucleotide ids, optionally alongside the `[2]` activity label."""

    def __init__(self, path: Path, with_labels: bool = False) -> None:
        blob = torch.load(str(path), map_location="cpu")
        self.x = torch.as_tensor(blob["x_clean"]).to(torch.int64)
        self.y = torch.as_tensor(blob["y"]).float() if "y" in blob else None
        self.with_labels = with_labels and self.y is not None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i):
        if self.with_labels:
            return self.x[i], self.y[i]
        return self.x[i]


def load_dna(name: str, split: str = "train", root: Path = DNA_ROOT,
             with_labels: bool = False) -> tuple[DNADataset, dict]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown DNA dataset {name!r}; known={list(_REGISTRY)}")
    spec = _REGISTRY[name]
    base = Path(root) / spec["subdir"]
    path = base / f"{spec['splits'][split]}.pt"
    mpath = base / "meta.json"
    if not path.exists() or not mpath.exists():
        raise FileNotFoundError(
            f"missing DNA cache {path}; build it with `python scripts/prep_deepstarr.py`")
    prep = json.loads(mpath.read_text())
    ds = DNADataset(path, with_labels=with_labels)
    L = int(ds.x.shape[1])
    if L != int(prep["seq_len"]):
        raise ValueError(f"{name} cache has L={L}, meta.json says {prep['seq_len']}")
    meta = {
        "vocab_size": int(prep["vocab_size"]),
        "seq_len": L,
        "tokenizer": "nucleotide",
        "dir": str(path),
        "is_text": False,
        "domain": "dna",
        "mask_id": int(prep["mask_id"]),
        "pad_id": -1,
        "alphabet": list(prep["alphabet"]),
        "label_cols": list(prep["label_cols"]),
    }
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2, root: Path = DNA_ROOT) -> tuple[DataLoader, dict]:
    ds, meta = load_dna(name, split, root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return loader, meta


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def decode(name: str, ids: np.ndarray, alphabet: list[str] | None = None) -> list[str]:
    """Decode `[B, L]` (or `[L]`) nucleotide ids to strings.

    A position still holding the absorbing MASK decodes to `N`, which keeps the string at its
    full length so the oracle can score it and so a partially decoded sample is visibly
    distinguishable from a complete one.
    """
    if alphabet is None:
        _, meta = load_dna(name, "val")
        alphabet = meta["alphabet"]
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None]
    table = list(alphabet)
    return ["".join(table[c] if 0 <= c < len(table) else "N" for c in row) for row in ids]
