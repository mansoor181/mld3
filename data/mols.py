"""Molecule dataloaders for the latent-kernel flow (QM9, ZINC-250k).

Reads the exact preprocessed caches that the PairFlow repo builds through
`data/preprocessed/<corpus>/download_dataset.py`, so a student trains on the identical
tokenization as the MDLM/DCD/ReDi rows we compare against. Each row is a length-L
`input_ids` sequence of SMILES tokens.

Two properties of these corpora differ from the text ones and both matter downstream.
The absorbing MASK token is a real member of the vocabulary (id 2) rather than an index
appended past the end, and the data itself never uses it, so `meta["mask_id"]` is surfaced
and the trainer must not extend the vocabulary. Roughly half of every sequence is padding,
and PairFlow trains its teachers with `compute_loss_on_pad_tokens=True`, so we model the
padding rather than exclude it and `meta["pad_id"]` stays -1 for loss purposes. The raw
pad id is carried separately as `meta["pad_token_id"]` for decoding.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from paths import MOL_ROOT

# name -> (subdir under MOL_ROOT, block size L, HF tokenizer id, data vocab size)
_REGISTRY = {
    "qm9": dict(subdir="qm9", L=32, tokenizer="yairschiff/qm9-tokenizer", vocab=40,
                splits={"train": "train", "val": "valid", "test": "valid"}),
    "zinc250k": dict(subdir="zinc-250k", L=74, tokenizer="yairschiff/zinc250k-tokenizer",
                     vocab=72,
                     splits={"train": "train", "val": "valid", "test": "valid"}),
}

# Shared special-token layout of both `yairschiff` SMILES tokenizers (verified by probing
# the live tokenizers, not read off the paper).
BOS_ID, EOS_ID, MASK_ID, PAD_ID, UNK_ID = 0, 1, 2, 3, 4


def is_mol_dataset(name: str) -> bool:
    return name in _REGISTRY


class MolTensorDataset(Dataset):
    """Wrap PairFlow's `{split}.pt` tensor dump, yielding `[L]` int64 input_ids."""

    def __init__(self, path: Path) -> None:
        blob = torch.load(str(path), map_location="cpu")
        x = blob["x_clean"] if isinstance(blob, dict) else blob
        # Stored as int8 to save disk; widen once here rather than per __getitem__.
        self.x = torch.as_tensor(x).to(torch.int64)
        if self.x.ndim != 2:
            raise ValueError(f"expected [N, L] in {path}, got {tuple(self.x.shape)}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


def load_mol(name: str, split: str = "train",
             root: Path = MOL_ROOT) -> tuple[MolTensorDataset, dict]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown molecule dataset {name!r}; known={list(_REGISTRY)}")
    spec = _REGISTRY[name]
    fname = spec["splits"][split]
    path = Path(root) / spec["subdir"] / "parsed" / f"{fname}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing molecule cache {path}; build it with "
            f"`python data/preprocessed/{spec['subdir']}/download_dataset.py` in the PairFlow repo")
    ds = MolTensorDataset(path)
    L = int(ds.x.shape[1])
    if L != spec["L"]:
        raise ValueError(f"{name} cache has L={L}, registry says {spec['L']}")
    hi = int(ds.x.max())
    if hi >= spec["vocab"]:
        raise ValueError(f"{name} cache holds token id {hi} but vocab is {spec['vocab']}")
    if bool((ds.x == MASK_ID).any()):
        raise ValueError(
            f"{name} cache contains the MASK id {MASK_ID}; the absorbing state would be ambiguous")
    meta = {
        "vocab_size": int(spec["vocab"]),
        "seq_len": L,
        "tokenizer": spec["tokenizer"],
        "dir": str(path),
        "is_text": False,
        "domain": "mol",
        # The mask lives inside the vocabulary, so the trainer must NOT append one.
        "mask_id": MASK_ID,
        # PairFlow scores the loss on padding, so we do not exclude it.
        "pad_id": -1,
        "pad_token_id": PAD_ID,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
    }
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2, root: Path = MOL_ROOT) -> tuple[DataLoader, dict]:
    ds, meta = load_mol(name, split, root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return loader, meta


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


_TOK_CACHE: dict[str, object] = {}


def get_tokenizer(name: str):
    spec = _REGISTRY[name]
    tid = spec["tokenizer"]
    if tid not in _TOK_CACHE:
        _TOK_CACHE[tid] = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
    return _TOK_CACHE[tid]


def decode(name: str, ids: np.ndarray) -> list[str]:
    """Decode `[B, L]` (or `[L]`) token ids to SMILES strings.

    This reproduces PairFlow's `main.py::_eval_qm9` exactly, which appends an end-of-sequence
    marker, cuts at the first one and then strips the remaining special markers. Cutting
    before stripping matters, because a sample whose first token is already the end marker
    must decode to the empty string and be counted invalid rather than silently skipped.
    """
    tok = get_tokenizer(name)
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None]
    raw = tok.batch_decode(torch.as_tensor(ids))
    out = []
    for seq in raw:
        seq = (seq + "<eos>").split("<eos>")[0]
        for marker in ("<bos>", "<eos>", "<pad>"):
            seq = seq.replace(marker, "")
        out.append(seq)
    return out
