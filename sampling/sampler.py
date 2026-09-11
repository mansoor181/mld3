"""K-step samplers for the latent-kernel flow (masking path).

sample_lkf(model, K, batch_size, seq_len, device):
    starts from an all-MASK x_0 (t=0) and iteratively applies the flow-map kernel
    Ψ^θ_{t→s} with t = i/K, s = (i+1)/K for i = 0..K-1.
    At each step: forward pass → draw k ∼ w(x_t), then per-position tokens from P_i(·|x_t,k).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from sampling import policies as P


@torch.no_grad()
def sample_lkf(model, K: int, batch_size: int, seq_len: int, device: torch.device,
               freeze_k: int | None = None, shuffle_k: bool = False,
               commit_k: bool = False) -> torch.Tensor:
    """Sample from the latent-kernel flow in K steps.

    Args:
        model: LatentKernelFlow
        K: number of function evaluations
        batch_size: number of samples to draw
        seq_len: sequence length
        device: cuda/cpu
        freeze_k: if not None, force k = freeze_k at every step (diagnostic ablation)
        shuffle_k: if True, permute the drawn k across the batch (diagnostic ablation)
        commit_k: if True, draw k ~ w(x_0) once at step 0 and hold it fixed for the
            whole trajectory. Matches the sequence-level marginalization used at
            training time; this is the recommended M>1 default.

    Returns:
        x_1_hat: [B, L] int64
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    committed_k = None   # commit_k: k drawn once from the router at step 0, then held fixed

    for i in range(K):
        t = torch.full((batch_size,), i / K, device=device)
        s = torch.full((batch_size,), (i + 1) / K, device=device)
        logits, router_logits = model.trunk(x, t, s)            # [B,M,L,V], [B,M]

        # draw k
        if freeze_k is not None:
            k_ids = torch.full((batch_size,), freeze_k, device=device, dtype=torch.long)
        elif commit_k and committed_k is not None:
            k_ids = committed_k
        else:
            w = F.softmax(router_logits, dim=-1)
            k_ids = torch.multinomial(w, num_samples=1).squeeze(-1)
            if commit_k:
                committed_k = k_ids
        if shuffle_k:
            k_ids = k_ids[torch.randperm(batch_size, device=device)]

        # gather per-latent logits
        gath = logits[torch.arange(batch_size, device=device), k_ids]   # [B, L, V]

        # For masking path, only sample at currently MASK positions
        mask_positions = (x == mask_id)
        # sample tokens at mask positions
        probs = F.softmax(gath, dim=-1)                                  # [B, L, V]
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(batch_size, seq_len)
        x = torch.where(mask_positions, sampled, x)

    # After K steps if anything is still MASK, fill greedily from the last step's logits
    still_masked = (x == mask_id)
    if still_masked.any():
        # use logits at k=k_ids from the last step
        gath = logits[torch.arange(batch_size, device=device), k_ids]
        x = torch.where(still_masked, gath.argmax(dim=-1), x)
    return x


@torch.no_grad()
def sample_lkf_analytic(model, K: int, batch_size: int, seq_len: int, device: torch.device,
                        freeze_k: int | None = None, shuffle_k: bool = False,
                        commit_k: bool = False) -> torch.Tensor:
    """Analytic-reverse sampler (MDLM-equivalent) for the masking path.

    Unlike `sample_lkf` (which samples x_s directly from the two-time kernel Ψ_{t→s}, so the
    per-step reveal rate is whatever the network learned), this sampler enforces the exact
    absorbing-schedule reveal rate and only uses the network to predict the clean-data
    posterior. It mirrors MDLM's `_ddpm_update` so that the M=1 latent-kernel flow decodes by
    the *same algorithm* as the factorized diffusion baseline (fair sampler-parity comparison).

    Procedure (i = 0..K-1, t = i/K, s = (i+1)/K):
      * query the kernel at (t, s=1): at s=1 the reveal prob (κ_s-κ_t)/(1-κ_t) = 1, so the
        per-latent head is exactly the x_1 (clean-data) posterior p_x0 = softmax(logits_k).
      * mask/move chance mc_τ = 1 - κ_τ = 1 - τ. Analytic reverse posterior at a masked pos:
          q(token j) ∝ p_x0[j] · (mc_t - mc_s),   q(MASK) ∝ mc_s
        (the MASK column of p_x0 is zeroed + renormalized first). Sample only at MASK
        positions; already-revealed positions are copied through (identity).

    At M=1 this is numerically identical to MDLM's ddpm_cache update.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    ones = torch.ones(batch_size, device=device)
    arange = torch.arange(batch_size, device=device)
    p_x0 = None
    committed_k = None   # commit_k: k drawn once from the router at step 0, then held fixed

    for i in range(K):
        t = torch.full((batch_size,), i / K, device=device)
        # query the clean-data posterior at s = 1 (full reveal head)
        logits, router_logits = model.trunk(x, t, ones)          # [B,M,L,V], [B,M]

        # draw k ~ w(x_t, t, s=1)
        if freeze_k is not None:
            k_ids = torch.full((batch_size,), freeze_k, device=device, dtype=torch.long)
        elif commit_k and committed_k is not None:
            k_ids = committed_k
        else:
            w = F.softmax(router_logits, dim=-1)
            k_ids = torch.multinomial(w, num_samples=1).squeeze(-1)
            if commit_k:
                committed_k = k_ids
        if shuffle_k:
            k_ids = k_ids[torch.randperm(batch_size, device=device)]

        gath = logits[arange, k_ids]                              # [B, L, V]
        p_x0 = F.softmax(gath, dim=-1)                            # [B, L, V]
        # SUBS parameterization: the clean posterior never emits MASK
        p_x0 = p_x0.clone()
        p_x0[..., mask_id] = 0.0
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # analytic absorbing-reverse posterior over (t -> s). The mask/move chance mc_tau is the
        # probability a position is still MASK at grid time tau. LKF's trunk time t_lkf = i/K is
        # the fraction KEPT, so the MDLM-time is tau = 1 - t_lkf, and:
        #   loglinear:  mc = (1 - eps) * tau = (1 - eps) * (1 - i/K)   (mirrors MDLM move_chance)
        #   linear:     mc = 1 - i/K
        # For loglinear the (1 - eps) factor is common to q(token) and q(MASK) below, so it
        # cancels in the multinomial normalization -> numerically identical to MDLM ddpm_cache.
        schedule = getattr(cfg, "schedule", "linear")
        if schedule == "loglinear":
            ome = 1.0 - getattr(cfg, "time_eps", 1e-3)
            mc_t = ome * (1.0 - i / K)
            mc_s = ome * (1.0 - (i + 1) / K)
        else:
            mc_t = (1.0 - i / K)
            mc_s = (1.0 - (i + 1) / K)
        q = p_x0 * (mc_t - mc_s)                                  # [B, L, V]
        q[..., mask_id] = mc_s
        sampled = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).view(batch_size, seq_len)

        mask_positions = (x == mask_id)
        x = torch.where(mask_positions, sampled, x)

    # fill any residual MASK greedily from the last clean posterior
    still_masked = (x == mask_id)
    if still_masked.any() and p_x0 is not None:
        x = torch.where(still_masked, p_x0.argmax(dim=-1), x)
    return x


@torch.no_grad()
def best_of_m_sample(model, K: int, batch_size: int, seq_len: int, device: torch.device,
                     M: int | None = None, score_draws: int = 4,
                     sampler: str = "ancestral",
                     return_scores: bool = False,
                     selector: str = "nelbo"):
    """Best-of-M rollout selector for the latent-kernel flow (headline M>1 policy).

    For each output slot we run M parallel freeze_k=m rollouts (m = 0..M-1) and keep the one a
    selector prefers. Two selectors are available and they see the identical candidates.

    With selector="nelbo" we score each candidate with the model's own sequence-level mixture
    NELBO (the same one used for the likelihood cross-check) under `score_draws` MC draws of
    (t, mask) that are SHARED across candidates for the same slot (paired-comparison variance
    reduction), and keep the argmin. Every draw needs the marginal denoiser, so the selector
    adds M*score_draws full calls on top of the pool.

    With selector="evidence" we keep the candidate whose frozen component assigned the highest
    total log-probability to the tokens it revealed along its own trajectory. The trunk already
    produces that quantity during generation, so this selector is free and the pool is the whole
    price. It is delegated to sampling.policies.rollout_pool, which is the same code path the
    decoding-policy sweep uses.

    Both are judge-free, since the model scores its own candidates, and embarrassingly parallel.

    Args:
        model: LatentKernelFlow with latent_M >= 1
        K, batch_size, seq_len, device: as in sample_lkf
        M: number of candidates per slot (defaults to model.cfg.latent_M)
        score_draws: number of shared MC (t, mask) draws used by the nelbo selector
        sampler: "ancestral" (Ψ_{t→s} rollout) or "analytic" (MDLM-reverse)
        return_scores: if True also return the [B, M] score matrix for logging
        selector: "nelbo" or "evidence"

    Returns:
        x_best: [B, L] int64 (winning rollout per slot)
        scores: [B, M] float if return_scores else omitted, lower is better for "nelbo" and
            higher is better for "evidence"
    """
    cfg = model.cfg
    if M is None:
        M = cfg.latent_M
    mask_id = cfg.mask_id
    fn = sample_lkf_analytic if sampler == "analytic" else sample_lkf

    if selector == "evidence":
        cands, ev, _, _, _ = P.rollout_pool(model, K, batch_size, seq_len, device,
                                            comps=list(range(M)),
                                            analytic=(sampler == "analytic"))
        x_best = P.pick(cands, ev.argmax(0))
        return (x_best, ev.transpose(0, 1)) if return_scores else x_best

    # M parallel freeze_k=m rollouts, stacked. The component index wraps, which matters only for
    # the M=1 paired control, where a pool of R candidates has to come from the single component
    # the trunk owns and the diversity within the pool is the sampling noise alone. For every
    # mixture cell R <= latent_M holds and the wrap is the identity.
    cands = []
    for m in range(M):
        x_m = fn(model, K=K, batch_size=batch_size, seq_len=seq_len, device=device,
                 freeze_k=m % cfg.latent_M)
        cands.append(x_m)
    cands = torch.stack(cands, dim=0)                                    # [M, B, L]

    # score each candidate with the k-marginalized denoiser at s=1, using shared
    # (t, mask) draws for paired comparison across the M candidates of each slot
    ones = torch.ones(batch_size, device=device)
    scores = torch.zeros(M, batch_size, device=device)
    for _ in range(score_draws):
        t = torch.rand(batch_size, device=device).clamp(1e-3, 1 - 1e-3)
        keep = torch.rand(batch_size, seq_len, device=device) < t[:, None]
        weight = (1.0 / (1.0 - t)).clamp_max(1e4)                        # [B]
        for m in range(M):
            x1 = cands[m]                                                # [B, L]
            x_t = torch.where(keep, x1, torch.full_like(x1, mask_id))
            logits, router_logits = model.trunk(x_t, t, ones)            # [B,M,L,V], [B,M]
            w = F.softmax(router_logits, dim=-1)                         # [B,M]
            p = F.softmax(logits, dim=-1)                                # [B,M,L,V]
            p_marg = (w[:, :, None, None] * p).sum(1)                    # [B,L,V]
            lp = torch.log(p_marg.clamp_min(1e-30))
            true_lp = lp.gather(-1, x1[:, :, None]).squeeze(-1)          # [B,L]
            masked = (x_t == mask_id).float()                            # [B,L]
            seq_nll = (weight[:, None] * (-true_lp) * masked).sum(1)     # [B]
            scores[m] += seq_nll / seq_len                               # per-token
    scores = scores / max(score_draws, 1)                                # [M, B]

    best = scores.argmin(dim=0)                                          # [B]
    x_best = cands[best, torch.arange(batch_size, device=device)]        # [B, L]
    if return_scores:
        return x_best, scores.transpose(0, 1)                            # [B, M]
    return x_best
