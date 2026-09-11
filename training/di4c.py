"""The Di4C baseline objective, reimplemented inside our trunk.

The comparison isolates supervision: the Di4C student carries the same mixture head as ours and
differs only in this loss. tests/test_di4c_parity.py checks the implementation term by term
against the authors' released code.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.latent_kernel import LatentKernelFlow, mask_forward_noise
from training.losses import _NEG_FLOOR

# ---------- Di4C baseline objective ----------
#
# A faithful reimplementation of the Di4C loss (baselines/di4c/sdtt/src/sdtt/core/distill/
# multi_round_sdtt.py::di4c_loss, lines 428-567) against this trunk, so a Di4C arm can be
# trained with exactly the same teacher, data, trunk, optimizer, EMA and step budget as the
# exact-mixture students and the only axis separating the two is the training objective.
#
# Conventions. Our time is kappa, the fraction of positions kept, while the reference uses a
# noise time with t=1 fully masked, so t_sdtt = 1 - t_kappa and the reference's r = 1 - t_sdtt
# is our t_kappa. The sigmoid weighting therefore needs no conversion and reads
# sigmoid(20 * (0.5 - t_kappa)), which is close to one when the sequence is nearly all masked
# and close to zero once most positions are revealed.
#
# One intentional deviation. The reference sums over every position because MDLM's carry-over
# forces already revealed positions to contribute log 1 = 0. Our trunk applies SUBS
# zero-masking without carry-over, so we restrict every position sum to the masked positions,
# which is the same restriction mixture_logprob already applies in latent_kernel.py.


def di4c_terms(lp: torch.Tensor,
               p0t_teacher: torch.Tensor, log_p0t_teacher: torch.Tensor,
               lp_u: torch.Tensor, x_1: torch.Tensor, x_0: torch.Tensor,
               mk: torch.Tensor, t_kappa: torch.Tensor,
               alpha_t: str = "sigmoid", alpha_const: float = 0.1,
               distil_delta: float = 0.01,
               use_cv: bool = True) -> dict[str, torch.Tensor]:
    """Every named term of the reference di4c_loss, as a pure function of pinned inputs.

    Keeping this free of sampling and of module state is what lets the parity test drive it
    with the same tensors the reference sees and compare term by term.

    Args:
        lp: [B, N, L, V] student log p(x0 | x_t) for each of the N latent draws, with grad.
        p0t_teacher, log_p0t_teacher: [B, L, V] teacher x0 posterior at x_t, no grad.
        lp_u: [B, N, L, V] student log p(x0 | x_s) at the teacher-advanced states, no grad.
        x_1: [B, L] clean data.
        x_0: [B, N, L] consistency targets sampled from lp_u.
        mk: [B, L] float indicator of the positions that are masked in x_t.
        t_kappa: [B] our kappa time.
    """
    B, N, L, V = lp.shape
    logN = float(np.log(N))
    mkn = mk[:, None, :]                                             # [B, 1, L]

    # Datapoint loss. dll_ is the reference's (B, D, batch_size) gather, transposed here to
    # put the latent axis second so it lines up with the rest of our [B, N, ...] layout.
    idx1 = x_1[:, None, :, None].expand(B, N, L, 1)
    dll_ = torch.gather(lp, -1, idx1).squeeze(-1)                    # [B, N, L]
    dll = (dll_ * mkn).sum(-1)                                       # [B, N]
    data_loss_pointwise = -(torch.logsumexp(dll, dim=1) - logN)      # [B]
    data_loss_nll = torch.logsumexp(dll_, dim=1) - logN              # [B, L]
    log_p0t_student = torch.logsumexp(lp, dim=1) - logN              # [B, L, V]
    if use_cv:
        data_loss_cor = data_loss_pointwise + (data_loss_nll * mk).sum(-1)
        data_loss_indep = (p0t_teacher * (log_p0t_teacher - log_p0t_student)
                           * mk[..., None]).sum(dim=(-1, -2))        # [B]
    else:
        data_loss_cor = data_loss_pointwise
        data_loss_indep = torch.zeros_like(data_loss_pointwise)

    if alpha_t == "sigmoid":
        time_coeff = torch.sigmoid(20.0 * (0.5 - t_kappa))
    elif alpha_t == "linear":
        time_coeff = 1.0 - t_kappa
    else:
        time_coeff = torch.ones_like(t_kappa)
    time_coeff = time_coeff * alpha_const
    data_loss = time_coeff * data_loss_cor + data_loss_indep

    # Pure-distillation branch, taken near the fully masked end. The reference switches the
    # whole batch on t.flatten()[0]; we decide per example, which agrees with the reference
    # whenever the batch shares one time and is otherwise a strict generalisation.
    use_distil = ((1.0 - t_kappa) < distil_delta)                    # [B] bool
    distil_loss = ((p0t_teacher[:, None] * (log_p0t_teacher[:, None] - lp)
                    * mk[:, None, :, None]).sum(dim=(-1, -2)).sum(dim=1) / N)   # [B]

    # Consistency loss.
    p0u_student = lp_u.exp().mean(dim=1)                             # [B, L, V]
    log_p0u_student = torch.logsumexp(lp_u, dim=1) - logN            # [B, L, V]
    if use_cv:
        consis_loss_indep = (p0u_student * (log_p0u_student - log_p0t_student)
                             * mk[..., None]).sum(dim=(-1, -2))      # [B]
    else:
        consis_loss_indep = torch.zeros_like(data_loss_pointwise)
    consis_loss_cor = torch.zeros_like(data_loss_pointwise)
    for i in range(N):
        idx0 = x_0[:, i][:, None, :, None].expand(B, N, L, 1)
        cll_ = torch.gather(lp, -1, idx0).squeeze(-1)                # [B, N, L]
        cll = (cll_ * mkn).sum(-1)                                   # [B, N]
        cll = torch.logsumexp(cll, dim=1) - logN
        if use_cv:
            cll = cll - ((torch.logsumexp(cll_, dim=1) - logN) * mk).sum(-1)
        consis_loss_cor = consis_loss_cor - cll / N
    consis_loss = consis_loss_cor + consis_loss_indep

    # The reference zeroes consis_loss on the distillation branch and zeroes distil_loss
    # otherwise, so exactly one of the two is active per example.
    tail = torch.where(use_distil, distil_loss, consis_loss)
    loss = (data_loss + tail).mean()
    return {
        "loss": loss,
        "data_loss": data_loss,
        "data_loss_pointwise": data_loss_pointwise,
        "data_loss_nll": data_loss_nll,
        "data_loss_cor": data_loss_cor,
        "data_loss_indep": data_loss_indep,
        "consis_loss": consis_loss,
        "consis_loss_cor": consis_loss_cor,
        "consis_loss_indep": consis_loss_indep,
        "distil_loss": distil_loss,
        "time_coeff": time_coeff,
        "distil_branch_frac": use_distil.float().mean(),
    }


@torch.no_grad()
def teacher_step_n(teacher, x_t: torch.Tensor, t_kappa: torch.Tensor,
                   s_kappa: torch.Tensor, n: int) -> torch.Tensor:
    """Draw n independent one-step teacher transitions out of the same state x_t.

    This is teacher_analytic_step with the single multinomial draw replaced by n draws from
    the same transition weights. The teacher is factorized, so its transition distribution
    depends only on (x_t, t, s) and the n draws are exactly n independent samples of the
    reference's _ddpm_update applied to n replicas of the row. Sharing the forward pass this
    way turns n teacher evaluations into one, which is what keeps the Di4C arm's compute in
    line with the exact-mixture arms.
    """
    cfg = teacher.cfg
    mask_id = cfg.mask_id
    B, L = x_t.shape
    ones = torch.ones(B, device=x_t.device)
    logits, router_logits = teacher.trunk(x_t, t_kappa, ones)        # [B,M,L,V], [B,M]
    w = F.softmax(router_logits, dim=-1)
    k_ids = torch.multinomial(w, num_samples=1).squeeze(-1)
    gath = logits[torch.arange(B, device=x_t.device), k_ids]         # [B, L, V]
    p_x0 = F.softmax(gath, dim=-1).clone()
    p_x0[..., mask_id] = 0.0
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    if getattr(cfg, "schedule", "linear") == "loglinear":
        ome = 1.0 - getattr(cfg, "time_eps", 1e-3)
        mc_t, mc_s = ome * (1.0 - t_kappa), ome * (1.0 - s_kappa)
    else:
        mc_t, mc_s = 1.0 - t_kappa, 1.0 - s_kappa
    q = p_x0 * (mc_t - mc_s).clamp_min(0.0)[:, None, None]
    q[..., mask_id] = mc_s[:, None]
    sampled = torch.multinomial(q.reshape(-1, q.shape[-1]), n, replacement=True)
    sampled = sampled.view(B, L, n).permute(0, 2, 1)                 # [B, n, L]
    return torch.where((x_t == mask_id)[:, None, :], sampled, x_t[:, None, :])


def _student_x0_logprob(student, x_t: torch.Tensor, t_kappa: torch.Tensor,
                        k: torch.Tensor, share_stack: bool) -> torch.Tensor:
    """Student log p(x0 | x_t) at time t_kappa under n latent components per row.

    x_t is [B, L] with a shared state across the n draws when share_stack is set, and
    [B, n, L] otherwise. k is [B, n]. Returns [B, n, L, V] with the absorbing column removed.
    When the state is shared we run the component-independent blocks once on B rows and
    broadcast, which is exact because the latent only enters the last latent_last_L blocks.
    """
    mask_id = student.cfg.mask_id
    B, n = k.shape
    ones_n = torch.ones(B * n, device=k.device)
    if share_stack:
        h, _ = student.trunk.shared_stack(x_t, t_kappa, torch.ones_like(t_kappa),
                                          return_router=False)       # [B, L, D]
        L, D = h.shape[1], h.shape[2]
        h = h.unsqueeze(1).expand(B, n, L, D).reshape(B * n, L, D)
    else:
        L = x_t.shape[-1]
        h, _ = student.trunk.shared_stack(x_t.reshape(B * n, L),
                                          t_kappa.repeat_interleave(n),
                                          ones_n, return_router=False)
    t_rep = t_kappa.repeat_interleave(n)
    logits = student.trunk.latent_stack(h, t_rep, ones_n, k_select=k.reshape(-1))
    logits = logits.reshape(B, n, L, -1)
    # A finite floor rather than -inf. The KL terms form differences of two such log-probs
    # weighted by a probability that is exactly zero on the mask column, and -inf would turn
    # that into 0 * nan. At -1e4 in float32 the softmax weight on the mask column underflows
    # to exactly zero, which is the behaviour we want, while the difference stays finite.
    logits[..., mask_id] = _NEG_FLOOR
    return F.log_softmax(logits, dim=-1)


def di4c_step_loss(student: LatentKernelFlow, teacher, x_1: torch.Tensor,
                   dc: dict) -> dict[str, torch.Tensor]:
    """One Di4C training step: sample a time, advance the teacher one step, score the terms.

    Unlike the exact-mixture objective this does not use the K-step boundary grid. Di4C draws
    a continuous time and advances the teacher by a single small step, and its consistency
    term is what is meant to make the student step-agnostic, so restricting it to the four
    student boundaries would handicap the method rather than make the comparison fairer. We
    keep the reference's own time sampling and step size and hold everything outside the loss
    fixed instead.
    """
    B, L = x_1.shape
    device = x_1.device
    mask_id = student.cfg.mask_id
    N = int(dc["latent_bsize"])
    T = int(dc["T"])

    # Noise time, uniform on (0, 1] and snapped to the 1/T grid. This is where we depart from
    # the reference _sample_t_like (multi_round_sdtt.py:409), which draws one time for the whole
    # batch and stratifies it, sending 30% of draws below distil_delta and the rest above. With
    # T = 1024 and distil_delta = 0.01 that puts 29.2% of reference steps in the pure
    # distillation branch against 0.98% of ours, and it also makes the branch a per-step choice
    # there and a per-example choice here. The deviation moves gradient budget toward the
    # consistency term rather than away from it, so it does not handicap the arm we are
    # measuring, and our estimator of the (distil_delta, 1) region has the lower variance of
    # the two.
    t_sdtt = torch.rand(B, device=device)
    t_sdtt = (t_sdtt * T).to(torch.int).float() / T + (1.0 / T)
    dt = torch.minimum(torch.full_like(t_sdtt, 1.0 / T), t_sdtt / 2.0)
    t_kappa = (1.0 - t_sdtt).clamp(0.0, 1.0)
    s_kappa = (t_kappa + dt).clamp(0.0, 1.0)

    x_t = mask_forward_noise(x_1, t_kappa, mask_id)
    mk = (x_t == mask_id).float()

    with torch.no_grad():
        ones = torch.ones(B, device=device)
        tlogits, _ = teacher.trunk(x_t, t_kappa, ones)               # [B, 1, L, V]
        tlogits = tlogits[:, 0].clone()
        tlogits[..., mask_id] = _NEG_FLOOR
        log_p0t_teacher = F.log_softmax(tlogits, dim=-1)
        p0t_teacher = log_p0t_teacher.exp()
        x_s = teacher_step_n(teacher, x_t, t_kappa, s_kappa, N)      # [B, N, L]

    if dc["latent_prior"] == "router":
        with torch.no_grad():
            _, router_probe = student.trunk.shared_stack(x_t, t_kappa, torch.ones_like(t_kappa))
            k = torch.multinomial(F.softmax(router_probe, dim=-1), N, replacement=True)
    else:
        k = torch.randint(0, student.cfg.latent_M, (B, N), device=device)

    lp = _student_x0_logprob(student, x_t, t_kappa, k, share_stack=True)
    with torch.no_grad():
        lp_u = _student_x0_logprob(student, x_s, s_kappa, k, share_stack=False)
        x_0 = torch.distributions.Categorical(logits=lp_u).sample()  # [B, N, L]

    terms = di4c_terms(lp, p0t_teacher, log_p0t_teacher, lp_u, x_1, x_0, mk, t_kappa,
                       alpha_t=dc["alpha_t"], alpha_const=dc["alpha_const"],
                       distil_delta=dc["distil_delta"], use_cv=dc["use_cv"])
    del lp_u

    # Router regularisation, identical in form and coefficient to the exact-mixture path so
    # neither arm is advantaged by a different penalty on the mixture weights.
    _, router_logits = student.trunk.shared_stack(x_t, t_kappa, torch.ones_like(t_kappa))
    log_w = F.log_softmax(router_logits, dim=-1)
    w = log_w.exp()
    marginal = w.mean(dim=0)
    H_marginal = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    H_cond = -(w * log_w).sum(dim=-1).mean()
    reg = (- student.cfg.router_entropy_coef * H_marginal
           + student.cfg.router_load_balance_coef * H_cond)
    loss = terms["loss"] + reg

    out = {
        "loss": loss,
        "nll": terms["loss"].detach(),
        "router_entropy": H_marginal.detach(),
        "load_balance": H_cond.detach(),
        "mi_est": torch.zeros((), device=device),
        "post_neg_ent": torch.zeros((), device=device),
        "semigroup": torch.zeros((), device=device),
        "mask_frac": mk.mean().detach(),
        "distil_branch_frac": terms["distil_branch_frac"].detach(),
    }
    for key in ("data_loss", "data_loss_cor", "data_loss_indep",
                "consis_loss", "consis_loss_cor", "consis_loss_indep",
                "distil_loss", "time_coeff"):
        out[key] = terms[key].mean().detach()

    # The exact-mixture NLL on this arm's own teacher target, so the two arms' validation
    # curves can be read on one axis. It costs a full M-branch forward, hence the flag.
    if dc["train_nll_exact"]:
        with torch.no_grad():
            log_mix, _, _ = student.mixture_logprob(x_t, x_s[:, 0], t_kappa, s_kappa)
            out["nll_exact"] = (-log_mix).mean().detach()
    else:
        out["nll_exact"] = out["nll"]
    return out
