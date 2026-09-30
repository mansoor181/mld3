"""The MLD3 mixture kernel.

    p_theta(x_s | x_t) = sum_k w_k(x_t, t) prod_i P_{i,k}(x_s^i | x_t, t)

A finite component index makes the step likelihood a closed-form sum of M terms, so the
mixing weights sit inside the logarithm and receive a gradient.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trunk import Trunk, TrunkConfig


def mask_forward_noise(x_1: torch.Tensor, t: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Keep each clean token with probability kappa_t = t, else replace it with the mask."""
    keep = torch.rand_like(x_1, dtype=torch.float) < t[:, None]
    return torch.where(keep, x_1, torch.full_like(x_1, mask_id))


def mask_two_time_transition(x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor,
                             s: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Reveal each masked position to its clean token with probability (s - t) / (1 - t)."""
    p_reveal = ((s - t) / (1.0 - t).clamp_min(1e-8)).clamp(0.0, 1.0)
    reveal = torch.rand_like(x_t, dtype=torch.float) < p_reveal[:, None]
    return torch.where(x_t == mask_id,
                       torch.where(reveal, x_1, torch.full_like(x_1, mask_id)), x_t)


@dataclass
class MLD3Config(TrunkConfig):
    mask_id: int = 0
    router_entropy_coef: float = 0.1


class MLD3(nn.Module):
    def __init__(self, cfg: MLD3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)

    def mixture_logprob(self, x_t: torch.Tensor, x_s: torch.Tensor,
                        t: torch.Tensor, s: torch.Tensor):
        """log sum_k w_k prod_i P_{i,k}(x_s^i), summed over the positions masked in x_t.

        Returns (log_mix [B], log_w [B, M], per_component [B, M]).
        """
        logits, router_logits = self.trunk(x_t, t, s)
        logp = F.log_softmax(logits, dim=-1)
        idx = x_s[:, None, :, None].expand(-1, self.cfg.latent_M, -1, 1)
        gathered = torch.gather(logp, -1, idx).squeeze(-1)          # [B, M, L]
        # Revealed positions transition deterministically and contribute log 1 = 0.
        masked = (x_t == self.cfg.mask_id).float()[:, None]
        per_component = (gathered * masked).sum(-1)                 # [B, M]
        log_w = F.log_softmax(router_logits, dim=-1)
        return torch.logsumexp(log_w + per_component, dim=-1), log_w, per_component
