"""Dataset loaders for the four corpus families the experiments use.

Each submodule exposes the same surface: `is_<family>_dataset(name)`, `make_loader(name, split, batch_size, ...)`, `load_<family>(...)`, and a `decode` helper for evaluation. training/trainer_distill.py and eval/eval_text.py dispatch on the dataset name across text.py (LM1B, WikiText-103), mols.py (QM9, ZINC-250k), dna.py (DeepSTARR), and images.py (ImageNet-256 VQ tokens).
"""
from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import DataLoader


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    """Cycle a DataLoader forever (the trainers count optimizer steps, not epochs)."""
    while True:
        for batch in loader:
            yield batch
