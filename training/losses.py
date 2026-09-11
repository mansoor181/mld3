"""The distillation objective and its dense-supervision time sampling.

distill_step_loss is the exact-mixture trajectory KL. The supervision knobs widen it from the
K-grid single-term objective to dense two-time sampling with a marginalized x0 KL; each knob is
inert at its default, and tests/test_supervision.py covers every one of them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.latent_kernel import LatentKernelFlow, mask_forward_noise
from training.teachers import teacher_rollout

# ---------- distillation loss ----------

_NEG_FLOOR = -1e4

# The dense-supervision knobs, at the values that reproduce the plain single-term objective.
# Anything that reads a config must fall back to these so an untouched run stays reproducible.
SUPERVISION_DEFAULTS = {
    "time_sampling": "grid",        # grid | mixed
    "offgrid_frac": 0.0,            # P(a step draws off the K grid) when mixed
    "horizon": "fixed",             # fixed | random
    "horizon_range": (0.03125, 1.0),
    "substep_mode": "fixed",        # fixed | scaled
    "x0_kl_coef": 0.0,              # weight on the dense teacher-posterior KL
    "x0_kl_rows": 1.0,              # fraction of the micro-batch the KL is evaluated on
    "n_targets": 1,                 # teacher targets drawn per (x_t, t)
}


def read_supervision_cfg(dcfg: dict) -> dict:
    """Pull the supervision knobs out of a distill config block, validating the enumerated ones."""
    sup = dict(SUPERVISION_DEFAULTS)
    for key in sup:
        if key in dcfg:
            sup[key] = dcfg[key]
    sup["offgrid_frac"] = float(sup["offgrid_frac"])
    sup["x0_kl_coef"] = float(sup["x0_kl_coef"])
    sup["x0_kl_rows"] = float(sup["x0_kl_rows"])
    sup["n_targets"] = int(sup["n_targets"])
    sup["horizon_range"] = tuple(float(x) for x in sup["horizon_range"])
    for key, allowed in (("time_sampling", ("grid", "mixed")),
                         ("horizon", ("fixed", "random")),
                         ("substep_mode", ("fixed", "scaled"))):
        if sup[key] not in allowed:
            raise SystemExit(f"distill.{key} must be one of {allowed}, got {sup[key]!r}")
    if sup["n_targets"] < 1:
        raise SystemExit("distill.n_targets must be at least 1")
    if not 0.0 <= sup["offgrid_frac"] <= 1.0:
        raise SystemExit("distill.offgrid_frac must lie in [0, 1]")
    if not 0.0 < sup["x0_kl_rows"] <= 1.0:
        raise SystemExit("distill.x0_kl_rows must lie in (0, 1]")
    lo, hi = sup["horizon_range"]
    if not 0.0 < lo <= hi <= 1.0:
        raise SystemExit("distill.horizon_range must satisfy 0 < lo <= hi <= 1")
    return sup


def sample_two_times(B: int, K: int, T_per_step: int, device, sup: dict):
    """Draw the (t, s) pair and the matching number of teacher sub-steps for one batch.

    With ``time_sampling='grid'`` this reproduces the original draw exactly, including the
    order in which it consumes the RNG, which is what lets an unmodified config reproduce
    the plain grid objective bit for bit.

    The off-grid branch decides on-grid against off-grid once per batch rather than once per
    row. A per-row decision would leave the rows of one batch wanting different sub-step
    counts, and teacher_rollout takes a single scalar count, so a per-row decision would cost
    a second rollout every step. Deciding per batch keeps one rollout per step and still
    covers the whole time axis over a run.

    Returns (t, s, T, frac_ongrid, horizon) with t and s of shape [B].
    """
    if sup["time_sampling"] == "grid":
        j = torch.randint(0, K, (B,), device=device)
        return j.float() / K, (j + 1).float() / K, T_per_step, 1.0, 1.0 / K

    on_grid = float(torch.rand((), device=device)) >= sup["offgrid_frac"]
    if on_grid:
        j = torch.randint(0, K, (B,), device=device)
        return j.float() / K, (j + 1).float() / K, T_per_step, 1.0, 1.0 / K

    if sup["horizon"] == "random":
        lo, hi = sup["horizon_range"]
        u = float(torch.rand((), device=device))
        h = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    else:
        h = 1.0 / K
    t = torch.rand(B, device=device) * (1.0 - h)
    s = (t + h).clamp(max=1.0)
    if sup["substep_mode"] == "scaled":
        T = int(min(32, max(1, round(T_per_step * h * K))))
    else:
        T = T_per_step
    return t, s, T, 0.0, h


def _drop_mask_column(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Renormalise a token distribution onto the real vocabulary, out of place.

    The clean-token posterior cannot place mass on the absorbing symbol, and an in-place write
    would break autograd on the student's branch, so we floor the column with an out-of-place
    index_fill rather than a clone followed by an assignment.
    """
    col = torch.tensor([mask_id], device=logits.device)
    return F.log_softmax(logits.index_fill(-1, col, _NEG_FLOOR), dim=-1)


def teacher_x0_logprob(teacher, x_t: torch.Tensor, t: torch.Tensor, mask_id: int,
                       sel: torch.Tensor | None = None) -> torch.Tensor:
    """Teacher's clean-token posterior log p(x_0 | x_t, t), with the absorbing column removed.

    A boolean sel of shape [B, L] restricts the result to the selected positions and returns
    [N, V] instead of [B, L, V].
    """
    ones = torch.ones(x_t.shape[0], device=x_t.device)
    logits, _ = teacher.trunk(x_t, t, ones)          # [B, M_t, L, V]
    logits = logits[:, 0]
    if sel is not None:
        logits = logits[sel]                         # [N, V]
    return _drop_mask_column(logits, mask_id)


def student_x0_logprob_marginal(student: LatentKernelFlow, x_t: torch.Tensor,
                                t: torch.Tensor, mask_id: int,
                                sel: torch.Tensor | None = None) -> torch.Tensor:
    """Student's k-MARGINALIZED clean-token posterior, log sum_k w_k p_k(x_0 | x_t, t).

    The marginal is the only correct target for the auxiliary KL of C2. Matching each
    component to the teacher individually would pull the components onto one factorized
    distribution and destroy the transition information the mixture exists to carry, whereas
    constraining only their weighted average leaves them free to specialize. It is also the
    exact quantity eval_text.py::elbo_ppl reports, so the training term and the diagnostic
    measure the same thing, and it is the same choice Di4C makes in its own data_loss_indep.

    We accumulate the mixture one component at a time. Softmaxing all M branches at once would
    hold four [B, M, L, V] tensors alive, which on the WikiText-103 shape is 26 GB and forces
    the micro-batch down to 16, and halving the micro-batch would then be scored against Di4C
    as if the term were twice as expensive as it is. A boolean sel of shape [B, L] restricts
    the whole computation to the selected positions, which are the only ones the KL reads.
    """
    ones = torch.ones(x_t.shape[0], device=x_t.device)
    logits, router_logits = student.trunk(x_t, t, ones)      # [B, M, L, V], [B, M]
    log_w = F.log_softmax(router_logits, dim=-1)             # [B, M]
    rows = sel.nonzero(as_tuple=True)[0] if sel is not None else None
    acc = None
    for m in range(logits.shape[1]):
        branch = logits[:, m]                                # [B, L, V]
        w = log_w[:, m]                                      # [B]
        if sel is not None:
            branch, w = branch[sel], w[rows][:, None]        # [N, V], [N, 1]
        else:
            w = w[:, None, None]                             # [B, 1, 1]
        term = _drop_mask_column(branch, mask_id) + w
        acc = term if acc is None else torch.logaddexp(acc, term)
    return acc


def distill_step_loss(student: LatentKernelFlow, teacher: LatentKernelFlow,
                      x_1: torch.Tensor, K: int, T_per_step: int,
                      objective_variant: str = "exact",
                      sup: dict | None = None) -> dict[str, torch.Tensor]:
    """Trajectory-matching KL: student's log p(x_s | x_t) on a teacher-rolled x_s.

    Sampling:
      j ~ U{0, ..., K-1};  t = j/K,  s = (j+1)/K  (LKF kappa times).
      x_t = teacher.forward_noise(x_1, t)  (mask each position w.p. 1-t, iid).
      x_s ~ teacher_rollout(x_t, t -> s) with T_per_step analytic sub-steps.

    objective_variant:
      "exact"     -- loss = -mean_b log Sigma_k w_k P_k(x_s|x_t)  (exact-mixture NLL)
      "mc_sample" -- Di4C-style implicit-mixture ablation: sample one latent index
                     k ~ Categorical(w(x_t)) per batch item, use only that component's
                     transition NLL:  loss = -mean_b log P_k(x_s|x_t).
                     Router receives gradient only via regularization + through the
                     importance-weighted sampling (self-normalized IS with N=1 leaves
                     no direct data-likelihood signal on the router). This mirrors
                     Di4C's "no explicit marginalization" objective.

    sup carries the three supervision changes that bring this objective level with Di4C's.
    All of them are inert at their defaults, so a config that sets none of them reproduces
    the original loss and consumes the RNG in the original order.
      time_sampling  -- 'grid' keeps the four fixed (t, s) pairs, 'mixed' covers the axis.
      x0_kl_coef     -- weight on the dense teacher-posterior KL, 0 disables the term and
                        skips both forwards it would need.
      n_targets      -- number of teacher targets drawn per (x_t, t) to cut the variance of
                        the single-draw transition score.
    """
    sup = sup or SUPERVISION_DEFAULTS
    B, L = x_1.shape
    device = x_1.device
    mask_id = student.cfg.mask_id
    t, s, T_roll, frac_ongrid, horizon = sample_two_times(B, K, T_per_step, device, sup)
    # Both models must agree on mask_id (enforced by caller).
    x_t = mask_forward_noise(x_1, t, mask_id)

    n_targets = int(sup["n_targets"])
    x_s = teacher_rollout(teacher, x_t, t, s, T_roll)

    log_mix, log_w, per_latent_logp = student.mixture_logprob(x_t, x_s, t, s)
    if objective_variant == "mc_sample":
        # Sample one component per batch item from the router distribution and use
        # only that component's log-prob as the transition NLL. Detach the router
        # for the sampler (avoid double-counting router grad via sampling probs).
        with torch.no_grad():
            probs = log_w.detach().exp()
            idx = torch.distributions.Categorical(probs=probs).sample()  # [B]
        picked = per_latent_logp.gather(-1, idx[:, None]).squeeze(-1)     # [B]
        nll = -picked.mean()
    elif objective_variant == "exact":
        nll = -log_mix.mean()
    else:
        raise ValueError(f"unknown objective_variant: {objective_variant}")

    # C3. Extra teacher targets out of the same (x_t, t). The single-draw score is the
    # highest-variance signal in this objective, because one draw of a K-step rollout has to
    # stand in for a distribution over roughly L/K jointly unmasked positions. The rollout
    # states diverge after the first sub-step, so the draws cannot share teacher forwards and
    # each one costs a further rollout. Pair n_targets > 1 with a smaller substeps to hold
    # teacher compute fixed.
    for _ in range(n_targets - 1):
        x_s_i = teacher_rollout(teacher, x_t, t, s, T_roll)
        log_mix_i, _, per_latent_i = student.mixture_logprob(x_t, x_s_i, t, s)
        if objective_variant == "mc_sample":
            with torch.no_grad():
                idx_i = torch.distributions.Categorical(probs=log_w.detach().exp()).sample()
            nll = nll - per_latent_i.gather(-1, idx_i[:, None]).squeeze(-1).mean()
        else:
            nll = nll - log_mix_i.mean()
    if n_targets > 1:
        nll = nll / n_targets

    # Router regularization: same form as loss()/loss_x0() in latent_kernel.py, so a mixture
    # student is not penalized for exploring its components early. Weights come from student cfg.
    w = log_w.exp()
    marginal = w.mean(dim=0)
    H_marginal = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    H_cond = -(w * log_w).sum(dim=-1).mean()
    reg = (- student.cfg.router_entropy_coef * H_marginal
           + student.cfg.router_load_balance_coef * H_cond)

    # C2. Dense teacher-posterior KL on the k-marginalized student. The transition score above
    # supervises one sampled destination per step, whereas this term supervises the full clean
    # -token distribution at every masked position, which is what keeps the denoiser anchored to
    # the teacher at times the transition score visits rarely or never.
    #
    # We accumulate it as a sum over the masked positions of a sequence and a mean over
    # sequences, which is exactly how nll above is accumulated. Both then carry the same units
    # and x0_kl_coef reads as the relative weight of one nat of teacher-posterior KL against
    # one nat of transition score. Normalising the KL per token instead would leave it roughly
    # L times smaller than the score, and a coefficient of 1 would be very nearly no term at
    # all. Because both sides scale with the sequence length, the coefficient still means the
    # same thing across corpora of different length.
    x0_kl = torch.zeros((), device=device)
    if sup["x0_kl_coef"] > 0.0:
        # The extra student forward this term needs costs about as much as the whole rest of
        # the step, because it runs all M branches through the vocabulary head. Evaluating it
        # on a fraction of the rows buys the supervision at a fraction of the price, and since
        # the loader already shuffles, the leading rows are a random subset. The KL is dense
        # over the vocabulary at every masked position it does see, which is why a quarter of
        # the rows still carries far more signal than the single sampled destination does.
        n_rows = max(1, int(round(B * sup["x0_kl_rows"])))
        x_k, t_k = (x_t, t) if n_rows == B else (x_t[:n_rows], t[:n_rows])
        sel = x_k == mask_id                                       # [n_rows, L]
        if int(sel.sum()) > 0:
            with torch.no_grad():
                log_pt = teacher_x0_logprob(teacher, x_k, t_k, mask_id, sel)   # [N, V]
                p_t = log_pt.exp()
            log_ps = student_x0_logprob_marginal(student, x_k, t_k, mask_id, sel)
            x0_kl = (p_t * (log_pt - log_ps)).sum() / n_rows

    loss = nll + sup["x0_kl_coef"] * x0_kl + reg

    # Diagnostics (same shape as latent_kernel.py's diag block).
    with torch.no_grad():
        posterior = F.softmax(log_w + per_latent_logp, dim=-1)
        H_post = -(posterior * posterior.clamp_min(1e-30).log()).sum(-1)
        H_prior = -(w * log_w).sum(-1)
        mi_est = (H_prior - H_post).mean()
        post_neg_ent = (-H_post).mean()
        mask_frac = (x_t == mask_id).float().mean()

    with torch.no_grad():
        nll_exact = (-log_mix).mean().detach()

    return {
        "loss": loss,
        "nll": nll.detach(),
        "nll_exact": nll_exact,   # exact-mixture NLL (identical to nll under variant=exact);
                                   # under variant=mc_sample this is a diagnostic for
                                   # head-to-head comparison vs the exact-mixture student.
        "router_entropy": H_marginal.detach(),
        "load_balance": H_cond.detach(),
        "mi_est": mi_est.detach(),
        "post_neg_ent": post_neg_ent.detach(),
        "semigroup": torch.zeros((), device=device),
        "mask_frac": mask_frac.detach(),
        # sup instrumentation. frac_ongrid and mean_horizon tell us at a glance which time
        # sampler a run actually used, which is the one thing a stale config would hide.
        "x0_kl": x0_kl.detach(),
        "frac_ongrid": torch.tensor(frac_ongrid, device=device),
        "mean_horizon": torch.tensor(horizon, device=device),
        "mean_t": t.mean().detach(),
    }
