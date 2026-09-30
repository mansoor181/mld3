"""The MLD3 training objective: L = L_trans + lambda_0 * L_0 + R."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.mld3 import MLD3, mask_forward_noise
from models.teachers import rollout

_NEG = -1e4


def sample_times(B: int, K: int, T0: int, eta: float, horizon: tuple[float, float], device):
    """Draw the step (t, s) and the matching teacher sub-step count.

    With probability 1 - eta the step comes from the K-step sampling schedule. Otherwise the
    step size h is log-uniform on `horizon` and the sub-step count grows with it. The choice is
    made once per batch rather than per row, because the rollout takes a single scalar count
    and a per-row choice would cost a second rollout every step.
    """
    if torch.rand(()) >= eta:
        j = torch.randint(0, K, (B,), device=device)
        return j / K, (j + 1) / K, T0

    lo, hi = horizon
    h = math.exp(math.log(lo) + float(torch.rand(())) * (math.log(hi) - math.log(lo)))
    t = torch.rand(B, device=device) * (1.0 - h)
    return t, (t + h).clamp(max=1.0), min(32, max(1, round(T0 * h * K)))


def _clean_posterior(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Renormalise onto the real vocabulary. Out of place, so autograd survives it."""
    col = torch.tensor([mask_id], device=logits.device)
    return F.log_softmax(logits.index_fill(-1, col, _NEG), dim=-1)


def _student_marginal_posterior(student: MLD3, x_t, t, mask_id, sel):
    """log sum_k w_k p_k(x_0 | x_t, t) at the selected positions, [N, V].

    The components are accumulated one at a time: softmaxing all M branches at once holds
    several [B, M, L, V] tensors alive, which on the WikiText-103 shape forces the micro-batch
    down by half.
    """
    logits, router = student.trunk(x_t, t, torch.ones_like(t))
    log_w = F.log_softmax(router, dim=-1)
    rows = sel.nonzero(as_tuple=True)[0]
    acc = None
    for k in range(logits.shape[1]):
        term = _clean_posterior(logits[:, k][sel], mask_id) + log_w[:, k][rows][:, None]
        acc = term if acc is None else torch.logaddexp(acc, term)
    return acc


def distill_loss(student: MLD3, teacher, x_1: torch.Tensor, K: int, T0: int,
                 eta: float, horizon: tuple[float, float],
                 x0_coef: float, x0_rows: float) -> dict:
    """One distillation step against a teacher rollout."""
    B = x_1.shape[0]
    mask_id = student.cfg.mask_id
    t, s, T = sample_times(B, K, T0, eta, horizon, x_1.device)

    x_t = mask_forward_noise(x_1, t, mask_id)
    x_s = rollout(teacher, x_t, t, s, T)
    log_mix, log_w, per_component = student.mixture_logprob(x_t, x_s, t, s)
    trans = -log_mix.mean()

    # Entropy regularizer. The first term keeps every component in use across the batch, the
    # second stops the router committing before the components have specialized.
    w = log_w.exp()
    marginal = w.mean(0)
    h_marginal = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    h_conditional = -(w * log_w).sum(-1).mean()
    reg = -student.cfg.router_entropy_coef * (h_marginal + h_conditional)

    # Auxiliary loss: the teacher's full clean-token posterior against the student's
    # component-averaged one, which constrains the average and leaves the components free to
    # specialize. Summed over masked positions and averaged over sequences, so it carries the
    # same units as the transition loss and x0_coef reads as a relative weight.
    x0_kl = torch.zeros((), device=x_1.device)
    if x0_coef > 0.0:
        n = max(1, round(B * x0_rows))
        x_k, t_k = x_t[:n], t[:n]
        sel = x_k == mask_id
        if sel.any():
            with torch.no_grad():
                logits, _ = teacher.trunk(x_k, t_k, torch.ones_like(t_k))
                log_pt = _clean_posterior(logits[:, 0][sel], mask_id)
            log_ps = _student_marginal_posterior(student, x_k, t_k, mask_id, sel)
            x0_kl = (log_pt.exp() * (log_pt - log_ps)).sum() / n

    with torch.no_grad():
        posterior = F.softmax(log_w + per_component, dim=-1)
        h_posterior = -(posterior * posterior.clamp_min(1e-30).log()).sum(-1)
        transition_info = (h_conditional - h_posterior.mean())

    return {
        "loss": trans + x0_coef * x0_kl + reg,
        "trans": trans.detach(),
        "x0_kl": x0_kl.detach(),
        "router_entropy": h_marginal.detach(),
        "transition_info": transition_info.detach(),
    }
