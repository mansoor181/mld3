"""VQ-token image dataloader for the latent-kernel flow (ImageNet-256 as VQGAN f16 codes).

Reads the cache that `scripts/prep_imagenet256.py` writes, in which every row is a 16x16 grid
of VQGAN codebook indices raveled to a length-256 sequence and every label is an ImageNet class
index. The absorbing MASK sits inside the model vocabulary at 1024, just past the last codebook
entry, which is the same index the MaskGIT teacher uses for its own visual mask token, so the
trainer takes `meta["mask_id"]` as given and does not extend the vocabulary.

Every position carries a real code, hence `meta["pad_id"]` is -1 and the whole grid is modelled.

The label is kept as int64 rather than cast to float as at `data/dna.py:40`, because a class
index is a categorical condition and not a regression target. Nothing in the first pass of the
image arm reads it, since we train and decode unconditionally, and it is stored so that the
class-marginal teacher variant needs no second pass over the corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import IMAGE_ROOT

_REGISTRY = {
    "imagenet256": dict(subdir="imagenet256",
                        splits={"train": "train", "val": "val", "test": "val"}),
}


def is_image_dataset(name: str) -> bool:
    return name in _REGISTRY


class ImageTokenDataset(Dataset):
    """Yield `[L]` int64 codebook ids, optionally alongside the scalar class label."""

    def __init__(self, path: Path, with_labels: bool = False) -> None:
        blob = torch.load(str(path), map_location="cpu")
        # The cache stores int16 to halve the 635 MB train blob. Casting the whole tensor to
        # int64 here would quadruple it to 2.5 GB of resident memory for no gain, so the cast
        # happens per item instead and the embedding layer sees int64 exactly as it expects.
        self.x = torch.as_tensor(blob["x_clean"])
        self.y = torch.as_tensor(blob["y"]).to(torch.int64) if "y" in blob else None
        self.with_labels = with_labels and self.y is not None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i):
        x = self.x[i].to(torch.int64)
        if self.with_labels:
            return x, self.y[i]
        return x


def load_images(name: str, split: str = "train", root: Path = IMAGE_ROOT,
                with_labels: bool = False) -> tuple[ImageTokenDataset, dict]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown image dataset {name!r}; known={list(_REGISTRY)}")
    spec = _REGISTRY[name]
    base = Path(root) / spec["subdir"]
    path = base / f"{spec['splits'][split]}.pt"
    mpath = base / "meta.json"
    if not path.exists() or not mpath.exists():
        raise FileNotFoundError(
            f"missing image cache {path}; build it with `python scripts/prep_imagenet256.py`")
    prep = json.loads(mpath.read_text())
    ds = ImageTokenDataset(path, with_labels=with_labels)
    L = int(ds.x.shape[1])
    if L != int(prep["seq_len"]):
        raise ValueError(f"{name} cache has L={L}, meta.json says {prep['seq_len']}")
    meta = {
        "vocab_size": int(prep["vocab_size"]),
        "seq_len": L,
        "tokenizer": str(prep["tokenizer"]),
        "dir": str(path),
        "is_text": False,
        "domain": "image",
        "mask_id": int(prep["mask_id"]),
        "pad_id": -1,
        "codebook_size": int(prep["codebook_size"]),
        "n_classes": int(prep["n_classes"]),
        "grid": int(prep["grid"]),
    }
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2, root: Path = IMAGE_ROOT) -> tuple[DataLoader, dict]:
    ds, meta = load_images(name, split, root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return loader, meta


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def decode(name: str, ids: np.ndarray, grid: int | None = None) -> np.ndarray:
    """Reshape `[B, L]` (or `[L]`) codebook ids into `[B, grid, grid]` token grids.

    Decoding stops here rather than continuing to pixels, so that the trainer never has to
    import `taming` and never has to hold a 958 MB VQGAN. Turning a grid into an image lives in
    `metrics/image_metrics.py`, which is only reached from the evaluator.

    A position still holding the absorbing MASK is left at its own id rather than remapped,
    because the VQGAN has no entry for it and a partially decoded sample must fail loudly at
    the point of decoding instead of quietly becoming codebook entry zero.
    """
    if grid is None:
        _, meta = load_images(name, "val")
        grid = meta["grid"]
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None]
    if ids.shape[1] != grid * grid:
        raise ValueError(f"expected {grid * grid} ids per row, got {ids.shape[1]}")
    return ids.reshape(-1, grid, grid)
