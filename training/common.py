"""Shared training utilities: seeding, device selection, config loading, vocabulary/MASK resolution, and the EMA tracker.

These five symbols are consumed by training/trainer_distill.py (the only trainer in this tree) and by the tests. They were extracted from the retired generic trainer so that the distillation trainer carries no dead code path with it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import yaml

from paths import config_vars


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expand(node, variables: dict):
    """Recursively substitute ${NAME} references in every string of a parsed config."""
    if isinstance(node, dict):
        return {k: _expand(v, variables) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v, variables) for v in node]
    if isinstance(node, str) and "${" in node:
        for name, value in variables.items():
            node = node.replace("${" + name + "}", value)
        if "${" in node:
            raise ValueError(f"config references an unknown path root: {node}")
    return node


def load_config(path: str) -> dict:
    """Load a cell config, resolving the ${NAME} path roots defined in paths.config_vars()."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _expand(cfg, config_vars())


def maybe_extend_vocab_for_mask(vocab_size: int, use_mask_prior: bool,
                                mask_id: int | None = None) -> tuple[int, int]:
    """Resolve the model vocabulary and the absorbing MASK index.

    Text corpora carry no MASK of their own, so the absorbing path appends one past the end
    of the data vocabulary. The molecule and DNA corpora already reserve a MASK inside their
    vocabulary, and passing `mask_id` explicitly selects that one and leaves the vocabulary
    size untouched.
    """
    if mask_id is not None and int(mask_id) >= 0:
        mask_id = int(mask_id)
        if not use_mask_prior:
            raise ValueError("an explicit mask_id requires use_mask_prior=True")
        if not 0 <= mask_id < vocab_size:
            raise ValueError(
                f"explicit mask_id {mask_id} lies outside the data vocabulary of size {vocab_size}")
        return vocab_size, mask_id
    if use_mask_prior:
        # MASK is the last token id
        return vocab_size + 1, vocab_size
    return vocab_size, -1


class EMA:
    """Exponential moving average of model parameters, MDLM-equivalent.

    Same decay warmup as MDLM's ExponentialMovingAverage(use_num_updates=True):
        d_n = min(decay, (1 + n) / (10 + n)),  shadow -= (1 - d_n) * (shadow - param)
    Shadow is keyed by parameter NAME so it survives checkpoint round-trips robustly.
    Checkpoints store only `state_dict()` output (plain tensors keyed by name), so moving
    this class between modules does not invalidate saved checkpoints.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].sub_((1.0 - d) * (self.shadow[n] - p))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow = state["shadow"]
