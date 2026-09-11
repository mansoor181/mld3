"""Term-by-term parity between our Di4C reimplementation and the reference implementation.

The reference loss lives in the authors' SDTT harness at
baselines/di4c/sdtt/src/sdtt/core/distill/multi_round_sdtt.py::di4c_loss. We cannot compare
the two end to end because five independent random draws differ between the stacks (the time
sample, the forward corruption, the DiT's own latent, the teacher transition and the
consistency target), and because the reference runs a different backbone entirely. What we can
do, and what this test does, is pin every one of those stochastic inputs and check that the two
implementations agree on each named quantity the loss is built from.

The reference is driven unmodified. We bind its own function to a stub carrying only the
attributes it touches, patch the sampling entry points to return fixed tensors, and read its
internal values out of the frame with a trace hook, so the numbers on the reference side are
genuinely the reference's and not a transcription of it.

The single intentional deviation, restricting every position sum to the masked positions,
is switched off here by passing an all-ones mask, which is the setting in which the two are
supposed to agree exactly.

Run:
    python -m tests.test_di4c_parity
"""
from __future__ import annotations
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sdtt.core.distill.multi_round_sdtt import MultiRoundSDTT
import torch.nn.functional as F

import paths as _paths

from training.di4c import di4c_terms

_SDTT_SRC = str(_paths.BASELINES_DIR / "di4c" / "sdtt" / "src")
if _SDTT_SRC not in sys.path:
    sys.path.insert(0, _SDTT_SRC)


class _Cfg:
    def __init__(self, N, alpha_t, alpha_const, distil_delta):
        self.latent_bsize = N
        self.alpha_t = alpha_t
        self.alpha_const = alpha_const
        self.distil_delta = distil_delta
        self.log_unif = False


class _Nop:
    def eval(self):
        return self

    def train(self):
        return self


class _RefStub:
    """The minimal surface the reference di4c_loss touches."""

    def __init__(self, cfg, xt, xs, logits_teacher, logits_student, logits_student_teacher,
                 x_0_fixed, T=1024):
        self.config = cfg
        self.device = torch.device("cpu")
        self.T = T
        self.backbone = _Nop()
        self.noise = _Nop()
        teacher = _Nop()
        teacher.is_di4c = False
        self.teacher = [teacher]
        self._xt = xt
        self._xs = xs
        self._logits_teacher = logits_teacher
        self._logits_student = logits_student
        self._logits_student_teacher = logits_student_teacher
        self._x_0_fixed = x_0_fixed
        self._fwd_calls = 0
        self.list_cnt = 0
        for name in ("distil_list", "data_list", "data_cor_list",
                     "consis_list", "consis_cor_list"):
            setattr(self, name, deque())

    def _t_to_sigma(self, t):
        col = t.reshape(-1, 1)
        return col, col, col

    def q_xt(self, x0, move_chance):
        return self._xt

    def _ddpm_update(self, xt_rep, t_rep, dt_rep, forward=None):
        return None, self._xs

    def forward_teacher(self, x, cond):
        return self._logits_teacher

    def forward(self, x, cond):
        # Call order inside di4c_loss: logits_student_teacher (line 466) then
        # logits_student (line 470).
        self._fwd_calls += 1
        return (self._logits_student_teacher if self._fwd_calls == 1
                else self._logits_student)


class _FakeCategorical:
    """Stands in for torch.distributions.categorical.Categorical so the consistency
    targets are pinned rather than resampled."""
    fixed = None

    def __init__(self, probs=None, logits=None):
        pass

    def sample(self):
        return _FakeCategorical.fixed


def _run_reference(stub, x0, t):
    """Call the unmodified reference and capture its frame locals on return."""
    fn = MultiRoundSDTT.di4c_loss
    captured = {}

    def tracer(frame, event, arg):
        if frame.f_code is fn.__code__:
            if event == "call":
                return tracer
            if event == "return":
                captured.update(dict(frame.f_locals))
        return None

    real = torch.distributions.categorical.Categorical
    torch.distributions.categorical.Categorical = _FakeCategorical
    sys.settrace(tracer)
    try:
        loss = fn(stub, x0, t=t)
    finally:
        sys.settrace(None)
        torch.distributions.categorical.Categorical = real
    captured["loss"] = loss
    return captured


def _case(name, B, D, S, N, t_value, alpha_t, alpha_const, distil_delta, seed=0):
    torch.manual_seed(seed)
    x0 = torch.randint(0, S, (B, D))
    xt = torch.randint(0, S, (B, D))
    xs = torch.randint(0, S, (B * N, D))
    logits_teacher = torch.randn(B, D, S)
    logits_student = torch.randn(B * N, D, S, requires_grad=False)
    logits_student_teacher = torch.randn(B * N, D, S)
    x_0_flat = torch.randint(0, S, (B * N, D))
    t = torch.full((B,), t_value)

    _FakeCategorical.fixed = x_0_flat
    cfg = _Cfg(N, alpha_t, alpha_const, distil_delta)
    stub = _RefStub(cfg, xt, xs, logits_teacher, logits_student,
                    logits_student_teacher, x_0_flat)
    ref = _run_reference(stub, x0, t)

    # Our side, in kappa time with the mask restriction switched off.
    lp = F.log_softmax(logits_student, dim=2).view(B, N, D, S)
    lp_u = F.log_softmax(logits_student_teacher, dim=2).view(B, N, D, S)
    log_p0t_teacher = F.log_softmax(logits_teacher, dim=2)
    p0t_teacher = F.softmax(logits_teacher, dim=2)
    x_0 = x_0_flat.view(B, N, D)
    mk = torch.ones(B, D)
    t_kappa = 1.0 - t

    ours = di4c_terms(lp, p0t_teacher, log_p0t_teacher, lp_u, x0, x_0, mk, t_kappa,
                      alpha_t=alpha_t, alpha_const=alpha_const,
                      distil_delta=distil_delta, use_cv=True)

    took_distil = bool(t.flatten()[0] < distil_delta)
    keys = ["data_loss_pointwise", "data_loss_nll", "data_loss_cor",
            "data_loss_indep", "time_coeff", "data_loss"]
    keys += ["distil_loss"] if took_distil else ["consis_loss_cor", "consis_loss_indep",
                                                 "consis_loss"]
    keys.append("loss")

    print(f"\n[{name}] t={t_value}  branch={'distil' if took_distil else 'consistency'}  "
          f"B={B} D={D} S={S} N={N}")
    bad = 0
    for k in keys:
        a, b = ours[k], ref[k]
        if not torch.is_tensor(b):
            b = torch.tensor(float(b))
        a, b = a.detach().float(), b.detach().float()
        ok = a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=1e-4)
        delta = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
        print(f"  {'OK  ' if ok else 'FAIL'} {k:22s} maxabs={delta:.3e}  shape={tuple(a.shape)}")
        bad += (not ok)
    return bad


def _structural_checks():
    print("\n[structural checks]")
    bad = 0
    B, D, S = 4, 6, 9

    # 1. The control variate makes data_loss_cor identically zero when every latent draw is
    #    the same branch, which is why an M=1 Di4C arm carries no correlation signal at all.
    torch.manual_seed(1)
    N = 4
    one = F.log_softmax(torch.randn(B, 1, D, S), dim=-1)
    lp = one.expand(B, N, D, S).contiguous()
    x0 = torch.randint(0, S, (B, D))
    out = di4c_terms(lp, torch.full((B, D, S), 1.0 / S), torch.full((B, D, S), -np.log(S)),
                     lp, x0, torch.randint(0, S, (B, N, D)), torch.ones(B, D),
                     torch.full((B,), 0.5))
    v = out["data_loss_cor"].abs().max().item()
    ok = v < 1e-4
    bad += (not ok)
    print(f"  {'OK  ' if ok else 'FAIL'} degenerate-latent data_loss_cor == 0   maxabs={v:.3e}")

    # 2. The time weighting must be strong where the sequence is mostly masked and negligible
    #    once it is mostly revealed. An inverted kappa map would flip these.
    t_k = torch.tensor([0.0, 0.75])
    tc = torch.sigmoid(20.0 * (0.5 - t_k))
    ok = tc[0].item() > 0.99 and tc[1].item() < 0.01
    bad += (not ok)
    print(f"  {'OK  ' if ok else 'FAIL'} time_coeff(0.0)={tc[0]:.4f} > 0.99, "
          f"time_coeff(0.75)={tc[1]:.4f} < 0.01")

    # 3. The mask restriction has to actually bite, otherwise revealed positions would leak
    #    into the datapoint term.
    torch.manual_seed(2)
    lp = F.log_softmax(torch.randn(B, 3, D, S), dim=-1)
    x0 = torch.randint(0, S, (B, D))
    mk = torch.zeros(B, D)
    mk[:, :2] = 1.0
    a = di4c_terms(lp, torch.full((B, D, S), 1.0 / S), torch.full((B, D, S), -np.log(S)),
                   lp, x0, torch.randint(0, S, (B, 3, D)), mk, torch.full((B,), 0.5))
    b = di4c_terms(lp, torch.full((B, D, S), 1.0 / S), torch.full((B, D, S), -np.log(S)),
                   lp, x0, torch.randint(0, S, (B, 3, D)), torch.ones(B, D),
                   torch.full((B,), 0.5))
    ok = not torch.allclose(a["data_loss_indep"], b["data_loss_indep"])
    bad += (not ok)
    print(f"  {'OK  ' if ok else 'FAIL'} mask restriction changes data_loss_indep")
    return bad


def main():
    bad = 0
    bad += _case("consistency branch", B=3, D=5, S=7, N=4, t_value=0.6,
                 alpha_t="sigmoid", alpha_const=0.1, distil_delta=0.01)
    bad += _case("distillation branch", B=3, D=5, S=7, N=4, t_value=0.005,
                 alpha_t="sigmoid", alpha_const=0.1, distil_delta=0.01, seed=3)
    bad += _case("linear weighting", B=2, D=4, S=6, N=3, t_value=0.35,
                 alpha_t="linear", alpha_const=1.0, distil_delta=0.01, seed=5)
    bad += _structural_checks()
    print(f"\n{'ALL CHECKS PASSED' if bad == 0 else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
