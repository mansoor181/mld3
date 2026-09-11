"""Latent-Kernel Discrete Flow, the mixture-of-factorized flow-map kernel.

Ψ^θ_{t→s}(x_s | x_t) = Σ_k w^θ_k(x_t, t, s) · Π_i P^θ_i(x_s^i | x_t, k, t, s)

Key operations:
  * Training: exact-mixture NLL of `x_s` given `x_t`; the target `(x_t, x_s)` pair is
    generated from a factorized conditional path given `x_1 ∼ data`, then `x_1` is marginalized
    out by the construction (we never see `x_1` at inference, so the model must learn to model
    the correlation created by marginalization).
  * Inference sampling: draw k ∼ router(x_t), then per-position tokens from P^θ_i(·|x_t,k).

Masking path (default):
  * Absorbing prior: x_0 = MASK; x_t is obtained by keeping x_1 with prob κ_t and MASK with
    prob 1-κ_t, IID per position.
  * Two-time transition `p_{t→s}(·|x_t, x_1)` for masking: each masked-in-x_t position is
    revealed to x_1^i with probability (κ_s - κ_t)/(1 - κ_t), stays MASK else. Kept-in-x_t
    positions stay their token (identity transition).
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trunk import LatentKernelTrunk, TrunkConfig


# ------------------ scheduler + noising ------------------

def linear_kappa(t: torch.Tensor) -> torch.Tensor:
    """Linear scheduler κ_t = t. Data at t=1, noise at t=0."""
    return t


def mask_forward_noise(x_1: torch.Tensor, t: torch.Tensor, mask_id: int,
                       kappa_fn=linear_kappa) -> torch.Tensor:
    """Sample x_t ∼ p_t(·|x_1) for masking path: each token kept w.p. κ_t, else MASK."""
    kappa = kappa_fn(t)                                  # [B]
    keep = (torch.rand_like(x_1, dtype=torch.float) < kappa[:, None])
    return torch.where(keep, x_1, torch.full_like(x_1, mask_id))


def mask_two_time_transition(x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                             mask_id: int, kappa_fn=linear_kappa) -> torch.Tensor:
    """Sample x_s ∼ p_{t→s}(·|x_t, x_1) for masking path (0 ≤ t < s ≤ 1).

    For each position i:
      * if x_t^i is a real token (not MASK): x_s^i = x_t^i (identity).
      * if x_t^i is MASK: reveal to x_1^i w.p. p_rev = (κ_s − κ_t) / (1 − κ_t), else stay MASK.
    """
    kt = kappa_fn(t)
    ks = kappa_fn(s)
    denom = (1.0 - kt).clamp_min(1e-8)
    p_rev = ((ks - kt) / denom).clamp(0.0, 1.0)          # [B]
    reveal = (torch.rand_like(x_t, dtype=torch.float) < p_rev[:, None])
    x_s = torch.where(x_t == mask_id,
                      torch.where(reveal, x_1, torch.full_like(x_1, mask_id)),
                      x_t)
    return x_s


# ------------------ model wrapper ------------------

@dataclass
class LKFConfig(TrunkConfig):
    mask_id: int = 0                       # override in caller: usually vocab_size (extended)
    use_mask_prior: bool = True
    # Training objective:
    #   "two_time": exact-mixture NLL of the two-time transition x_s | x_t.
    #   "x0":       MDLM-equivalent absorbing-ELBO on the clean tokens x_1 (SUBS zero-masking
    #                + loglinear ELBO weight). At M=1 this reproduces MDLM's per-token loss.
    objective: str = "two_time"
    time_eps: float = 1e-3                  # loglinear schedule eps (matches MDLM LogLinearNoise)
    schedule: str = "loglinear"             # x0-mode noise schedule; only "loglinear" supported
    # PAD exclusion (x0 objective only). If >= 0, positions where x_1 == pad_id are excluded
    # from the loss and its normalizer, exactly like MDLM's `_loss` multiplying by
    # attention_mask and dividing by attention_mask.sum(). -1 disables (legacy behavior:
    # the lm1b unwrapped cache is ~76.5% [PAD], so -1 trains mostly on predicting PAD).
    pad_id: int = -1
    # Antithetic (low-discrepancy) t-sampling for the x0 objective, exactly MDLM's
    # `_sample_t` with training.antithetic_sampling=True: u_i = (rand_i/B + i/B) mod 1
    # stratifies tau across the batch, reducing ELBO-weight variance. False = legacy uniform.
    antithetic_t: bool = False
    # Two-time schedule: min gap for training
    dt_min: float = 0.02
    dt_max: float = 1.0
    # Anti-collapse
    router_entropy_coef: float = 0.01
    router_load_balance_coef: float = 0.01
    # Semigroup / consistency training
    semigroup_coef: float = 0.0            # weight on the semigroup consistency loss
    semigroup_dt_min: float = 0.02         # minimum gap on (t,s) and (s,u) for consistency


class LatentKernelFlow(nn.Module):
    """Mixture-of-factorized flow map with exact per-step latent marginalization."""

    def __init__(self, cfg: LKFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Extend vocab by one for the [MASK] token if using absorbing prior.
        trunk_cfg = TrunkConfig(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_mult=cfg.mlp_mult,
            latent_M=cfg.latent_M,
            latent_last_L=cfg.latent_last_L,
            time_embed_dim=cfg.time_embed_dim,
            tie_embeddings=cfg.tie_embeddings,
            trunk_arch=getattr(cfg, "trunk_arch", "lkf"),
            time_eps=cfg.time_eps,
            schedule=cfg.schedule,
            dropout=getattr(cfg, "dropout", 0.0),
            cond_dim=getattr(cfg, "cond_dim", 0),
            cond_extra_silu=getattr(cfg, "cond_extra_silu", True),
            time_conditioning=getattr(cfg, "time_conditioning", True),
        )
        self.trunk = LatentKernelTrunk(trunk_cfg)

    # ---- exact-mixture NLL loss ----

    def mixture_logprob(
        self, x_t: torch.Tensor, x_s: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """log Σ_k w_k · Π_i P_i(x_s^i | x_t, k, t, s), per example.

        Returns:
            log_mix: [B], the mixture log-probability of x_s given x_t
            router_logp: [B, M], the log router weights, used for entropy regularization
            per_latent_logp: [B, M], the per-component log probability, used for MI enumeration
        """
        logits, router_logits = self.trunk(x_t, t, s)     # [B,M,L,V], [B,M]
        logp_tok = F.log_softmax(logits, dim=-1)          # [B,M,L,V]
        # Gather log-probs at x_s per position:
        # x_s: [B, L] → [B, 1, L, 1] → gather over V
        idx = x_s[:, None, :, None].expand(-1, self.cfg.latent_M, -1, 1)
        gathered = torch.gather(logp_tok, dim=-1, index=idx).squeeze(-1)  # [B, M, L]

        # For masking prior, kept-position (x_t not MASK) transitions are deterministic;
        # we should NOT include them in the training loss (they contribute log 1 = 0 exactly).
        if self.cfg.use_mask_prior:
            mask_positions = (x_t == self.cfg.mask_id).float()            # [B, L]
            per_latent_logp = (gathered * mask_positions[:, None]).sum(-1)  # [B, M]
        else:
            per_latent_logp = gathered.sum(-1)                              # [B, M]

        log_w = F.log_softmax(router_logits, dim=-1)                        # [B, M]
        log_mix = torch.logsumexp(log_w + per_latent_logp, dim=-1)          # [B]
        return log_mix, log_w, per_latent_logp
