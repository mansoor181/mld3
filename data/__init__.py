"""Corpus loaders.

Every corpus resolves to `[L]` int64 token rows plus a `meta` dict. Text caches are the
pre-tokenized HuggingFace dumps the baselines build, so the comparison uses one tokenization;
the other three corpora are `.pt` tensor dumps written by `scripts/`.

The absorbing symbol is appended past the end of the vocabulary for text, and is already a
reserved member of the vocabulary for molecules, DNA and images. `meta["mask_id"]` and
`meta["model_vocab"]` resolve that difference once, here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import DNA_ROOT, IMAGE_ROOT, MOL_ROOT, TEXT_ROOT

DATASETS = {
    "lm1b": dict(domain="text", root=TEXT_ROOT, subdir="lm1b", seq_len=128, vocab=30522,
                 stem="lm1b_{split}_bs128_unwrapped", tokenizer="bert-base-uncased",
                 splits={"train": "train", "val": "test"}),
    "wikitext103": dict(domain="text", root=TEXT_ROOT, subdir="wikitext103", seq_len=128,
                        vocab=50257, stem="wikitext103_{split}_bs128_wrapped", tokenizer="gpt2",
                        splits={"train": "train", "val": "validation"}),
    "qm9": dict(domain="mol", root=MOL_ROOT, subdir="qm9/parsed", seq_len=32, vocab=40,
                mask_id=2, tokenizer="yairschiff/qm9-tokenizer",
                splits={"train": "train", "val": "valid"}),
    "zinc250k": dict(domain="mol", root=MOL_ROOT, subdir="zinc-250k/parsed", seq_len=74,
                     vocab=72, mask_id=2, tokenizer="yairschiff/zinc250k-tokenizer",
                     splits={"train": "train", "val": "valid"}),
    "deepstarr": dict(domain="dna", root=DNA_ROOT, subdir="deepstarr", seq_len=249, vocab=6,
                      mask_id=5, tokenizer="nucleotide",
                      splits={"train": "train", "val": "valid"}),
    "imagenet256": dict(domain="image", root=IMAGE_ROOT, subdir="imagenet256", seq_len=256,
                        vocab=1025, mask_id=1024, tokenizer="vqgan",
                        splits={"train": "train", "val": "val"}),
}


class ArrowDataset(Dataset):
    def __init__(self, path: Path) -> None:
        import datasets as hfds
        self.ds = hfds.load_from_disk(str(path)).with_format("numpy", columns=["input_ids"])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.ds[i]["input_ids"].astype(np.int64))


class TensorDataset(Dataset):
    """A `.pt` dump holding `x_clean` as [N, L]. Kept narrow on disk and widened per item."""

    def __init__(self, path: Path) -> None:
        self.x = torch.load(str(path), map_location="cpu")["x_clean"]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i].to(torch.int64)


def load(name: str, split: str = "train") -> tuple[Dataset, dict]:
    spec = DATASETS[name]
    base = Path(spec["root"]) / spec["subdir"]
    hf_split = spec["splits"][split]
    if spec["domain"] == "text":
        path = base / f"{spec['stem'].format(split=hf_split)}.dat"
        ds = ArrowDataset(path) if path.exists() else None
    else:
        path = base / f"{hf_split}.pt"
        ds = TensorDataset(path) if path.exists() else None
    if ds is None:
        raise FileNotFoundError(f"missing {name} cache at {path}; see README for how to build it")

    mask_id = spec.get("mask_id")
    meta = {
        "domain": spec["domain"],
        "seq_len": spec["seq_len"],
        "vocab_size": spec["vocab"],
        "tokenizer": spec["tokenizer"],
        # Text appends the absorbing symbol; the other corpora reserve one inside the vocabulary.
        "mask_id": spec["vocab"] if mask_id is None else mask_id,
        "model_vocab": spec["vocab"] + 1 if mask_id is None else spec["vocab"],
    }
    side = base / "meta.json"
    if side.exists():
        meta.update({k: v for k, v in json.loads(side.read_text()).items()
                     if k in ("grid", "n_classes", "alphabet", "codebook_size")})
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2) -> tuple[DataLoader, dict]:
    ds, meta = load(name, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
    return loader, meta


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


_tokenizers: dict[str, object] = {}


def _tokenizer(name: str):
    tid = DATASETS[name]["tokenizer"]
    if tid not in _tokenizers:
        from transformers import AutoTokenizer
        _tokenizers[tid] = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
    return _tokenizers[tid]


def decode(name: str, ids: np.ndarray, meta: dict | None = None):
    """Token ids to strings, or to a token grid for images."""
    ids = np.atleast_2d(np.asarray(ids))
    domain = DATASETS[name]["domain"]

    if domain == "image":
        grid = (meta or load(name, "val")[1])["grid"]
        return ids.reshape(-1, grid, grid)

    if domain == "dna":
        table = (meta or load(name, "val")[1])["alphabet"]
        return ["".join(table[c] if c < len(table) else "N" for c in row) for row in ids]

    tok = _tokenizer(name)
    if domain == "mol":
        # PairFlow cuts at the first end marker before stripping the rest, so a sample that
        # starts with one decodes to the empty string and counts as invalid.
        out = []
        for seq in tok.batch_decode(torch.as_tensor(ids)):
            seq = (seq + "<eos>").split("<eos>")[0]
            for marker in ("<bos>", "<eos>", "<pad>"):
                seq = seq.replace(marker, "")
            out.append(seq)
        return out

    specials = set(tok.all_special_ids)
    return [tok.decode([int(c) for c in row if int(c) not in specials],
                       skip_special_tokens=True).strip() for row in ids]
