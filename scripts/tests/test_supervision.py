"""Tests for the dense supervision changes to the exact-mixture distillation objective.

The dense changes give the objective the three supervision channels Di4C already had, namely
dense time coverage, a dense teacher-posterior target and a lower-variance transition target.
Every one of them is inert at its default, because the grid-supervision results stay in the paper as the
unsupervised-elsewhere rung and we can only read them against dense if an untouched config still
reproduces the reference implementation exactly.

The tests, in the order the plan lists them:

  T1  A config that sets no dense key reproduces the reference loss trajectory. The two code trees are
      separate packages with the same module names, so we drive each one in its own
      subprocess rather than trying to import both into one interpreter.
  T2  The mixed time sampler covers the time axis at the requested rate, keeps the K grid
      oversampled, and draws its horizon log-uniformly over the configured range.
  T3  The auxiliary KL is taken against the k-marginalized student. A per-component KL would
      pull every component onto one factorized distribution and destroy the transition
      information the mixture exists to carry, so this is the guard that matters most.
  T4  The x0 KL gradient agrees with a finite difference, and it reaches the router.
  T5  n_targets > 1 reduces the variance of the transition score without moving its mean.

Run:
    python -m tests.test_supervision
"""
from __future__ import annotations
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.latent_kernel import LKFConfig, LatentKernelFlow
from training.losses import (
    SUPERVISION_DEFAULTS, distill_step_loss, sample_two_times,
    student_x0_logprob_marginal, teacher_x0_logprob,
)

# T1 compares this trainer against an independent reference implementation of the same
# objective, to catch an accidental change in the default supervision path. Point
# MLDF_REFERENCE_TREE at such a tree to enable it; the check skips when none is configured.
_REF_DIR = Path(os.environ["MLDF_REFERENCE_TREE"]) if os.environ.get("MLDF_REFERENCE_TREE") else None
_PY = sys.executable

V, L, M, K, SUB = 24, 16, 4, 4, 2
MASK_ID = V - 1


def _cfg(latent_M: int) -> LKFConfig:
    return LKFConfig(vocab_size=V, seq_len=L, dim=32, depth=2, heads=2, mlp_mult=2,
                     latent_M=latent_M, latent_last_L=1, mask_id=MASK_ID,
                     router_entropy_coef=0.01, router_load_balance_coef=0.01)


def _models(seed: int = 0):
    torch.manual_seed(seed)
    student = LatentKernelFlow(_cfg(M)).eval()
    teacher = LatentKernelFlow(_cfg(1)).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def _v2(**kw) -> dict:
    dense = dict(SUPERVISION_DEFAULTS)
    dense.update(kw)
    return dense


# ---------- T1: defaults reproduce the reference ----------

_DRIVER = r'''
import sys, json, torch
sys.path.insert(0, {code_dir!r})
from models.latent_kernel import LKFConfig, LatentKernelFlow
from training.losses import distill_step_loss
V, L, M, K, SUB, MASK_ID = {V}, {L}, {M}, {K}, {SUB}, {MASK}
def cfg(m):
    return LKFConfig(vocab_size=V, seq_len=L, dim=32, depth=2, heads=2, mlp_mult=2,
                     latent_M=m, latent_last_L=1, mask_id=MASK_ID,
                     router_entropy_coef=0.01, router_load_balance_coef=0.01)
torch.manual_seed(0)
student = LatentKernelFlow(cfg(M)).eval()
teacher = LatentKernelFlow(cfg(1)).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
torch.manual_seed(1234)
out = []
for _ in range({n_steps}):
    x1 = torch.randint(0, MASK_ID, (8, L))
    d = distill_step_loss(student, teacher, x1, K, SUB)
    out.append([float(d["loss"]), float(d["nll"]), float(d["mi_est"])])
print(json.dumps(out))
'''


def _run_driver(code_dir: Path, n_steps: int = 20):
    src = _DRIVER.format(code_dir=str(code_dir), V=V, L=L, M=M, K=K, SUB=SUB,
                         MASK=MASK_ID, n_steps=n_steps)
    r = subprocess.run([_PY, "-c", src], capture_output=True, text=True, cwd=str(code_dir))
    if r.returncode != 0:
        raise RuntimeError(f"driver failed in {code_dir}:\n{r.stderr[-3000:]}")
    return np.array(json.loads(r.stdout.strip().splitlines()[-1]))


def test_t1_defaults_reproduce_reference():
    if _REF_DIR is None or not _REF_DIR.exists():
        print("T1  SKIP (set MLDF_REFERENCE_TREE to a reference implementation to run this)")
        return
    a = _run_driver(_REPO_DIR)
    b = _run_driver(_REF_DIR)
    dev = float(np.abs(a - b).max())
    assert dev < 1e-6, f"defaults drifted from the reference by {dev:.3e}\n{a[:3]}\n{b[:3]}"
    print(f"T1  OK  20 steps identical to the reference tree, max deviation {dev:.2e}")


# ---------- T2: the mixed time sampler ----------

def test_t2_time_sampler():
    dev = torch.device("cpu")
    torch.manual_seed(0)

    # The grid setting must reproduce the original draw, including the RNG it consumes.
    torch.manual_seed(7)
    t_a, s_a, T_a, on_a, h_a = sample_two_times(64, K, SUB, dev, _v2())
    torch.manual_seed(7)
    j = torch.randint(0, K, (64,), device=dev)
    assert torch.equal(t_a, j.float() / K) and torch.equal(s_a, (j + 1).float() / K)
    assert T_a == SUB and on_a == 1.0 and abs(h_a - 1.0 / K) < 1e-12

    frac, ts, hs, subs = 0.5, [], [], []
    n_on = 0
    torch.manual_seed(11)
    dense = _v2(time_sampling="mixed", offgrid_frac=frac, horizon="random",
             substep_mode="scaled")
    for _ in range(4000):
        t, s, T, on, h = sample_two_times(8, K, SUB, dev, dense)
        n_on += int(on == 1.0)
        assert torch.all(s > t) and torch.all(s <= 1.0 + 1e-6)
        if on == 0.0:
            ts.append(t.numpy())
            hs.append(h)
            subs.append(T)
    got = 1.0 - n_on / 4000
    assert abs(got - frac) < 0.03, f"off-grid rate {got:.3f} != {frac}"

    # The horizon is log-uniform on the configured range, which is what puts comparable mass
    # on the one-step horizons the sampler will use at high NFE and on the long horizons it
    # uses at NFE 1. A uniform draw would spend nearly all of its mass on long horizons.
    lo, hi = dense["horizon_range"]
    hs = np.array(hs)
    assert hs.min() >= lo - 1e-9 and hs.max() <= hi + 1e-9
    u = (np.log(hs) - np.log(lo)) / (np.log(hi) - np.log(lo))
    assert abs(u.mean() - 0.5) < 0.03, f"log-horizon mean {u.mean():.3f}"

    # Off-grid times must actually leave the grid, and the scaled sub-step count must track
    # the horizon so a short hop is not charged for a full K-step rollout.
    ts = np.concatenate(ts)
    grid = np.array([i / K for i in range(K)])
    assert np.abs(ts[:, None] - grid[None]).min(1).mean() > 0.02
    subs = np.array(subs)
    assert subs.min() >= 1 and subs.max() <= 32
    assert np.corrcoef(hs, subs)[0, 1] > 0.9, "sub-steps do not track the horizon"
    print(f"T2  OK  off-grid rate {got:.3f}, log-horizon mean {u.mean():.3f}, "
          f"sub-steps {subs.min()}-{subs.max()}")


# ---------- T3: the KL is against the marginal ----------

def test_t3_kl_uses_the_marginal():
    student, teacher = _models()
    x1 = torch.randint(0, MASK_ID, (6, L))
    t = torch.rand(6) * 0.8 + 0.1
    x_t = torch.where(torch.rand(6, L) < t[:, None], x1, torch.full_like(x1, MASK_ID))

    log_ps = student_x0_logprob_marginal(student, x_t, t, MASK_ID)
    assert torch.allclose(log_ps.exp().sum(-1), torch.ones(6, L), atol=1e-5)
    assert float(log_ps[..., MASK_ID].exp().max()) < 1e-3, "mask column not suppressed"

    # The marginal must differ from every single component, otherwise the term would be a
    # per-component KL in disguise and would flatten the mixture.
    ones = torch.ones(6)
    logits, router_logits = student.trunk(x_t, t, ones)
    logits = logits.clone()
    logits[..., MASK_ID] = -1e4
    comp = F.log_softmax(logits, dim=-1)
    log_w = F.log_softmax(router_logits, dim=-1)
    recon = torch.logsumexp(log_w[:, :, None, None] + comp, dim=1)
    assert torch.allclose(recon, log_ps, atol=1e-5)
    gaps = [float((comp[:, m] - log_ps).abs().max()) for m in range(M)]
    assert min(gaps) > 1e-4, f"marginal coincides with a component, gaps {gaps}"

    # The selected-position path is the one training uses, because softmaxing all M branches
    # at every position does not fit in memory at the real model shape. It has to agree with
    # the dense path exactly, otherwise the term we test is not the term we train.
    sel = x_t == MASK_ID
    rows, cols = sel.nonzero(as_tuple=True)
    sel_ps = student_x0_logprob_marginal(student, x_t, t, MASK_ID, sel)
    assert sel_ps.shape == (int(sel.sum()), V)
    assert torch.allclose(sel_ps, log_ps[rows, cols], atol=1e-5)

    log_pt = teacher_x0_logprob(teacher, x_t, t, MASK_ID)
    assert torch.allclose(teacher_x0_logprob(teacher, x_t, t, MASK_ID, sel),
                          log_pt[rows, cols], atol=1e-5)
    assert log_pt.shape == (6, L, V)
    assert torch.allclose(log_pt.exp().sum(-1), torch.ones(6, L), atol=1e-5)

    # A KL is non-negative and vanishes only when the two agree, so pointing the helper at the
    # teacher twice has to give zero.
    self_kl = (log_pt.exp() * (log_pt - log_pt)).sum(-1)
    assert float(self_kl.abs().max()) < 1e-6
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1)
    assert float(kl.min()) > -1e-4, f"negative KL {float(kl.min()):.3e}"
    print(f"T3  OK  marginal normalised, distinct from all {M} components, KL >= 0 "
          f"(mean {float(kl.mean()):.3f} nats)")


# ---------- T4: the KL term has a correct, connected gradient ----------

def test_t4_kl_gradient():
    student, teacher = _models()
    x1 = torch.randint(0, MASK_ID, (4, L))
    torch.manual_seed(3)
    base = distill_step_loss(student, teacher, x1, K, SUB, "exact", _v2())
    torch.manual_seed(3)
    withkl = distill_step_loss(student, teacher, x1, K, SUB, "exact",
                               _v2(x0_kl_coef=1.0))
    assert float(base["x0_kl"]) == 0.0, "the term is not inert at coefficient zero"
    assert abs(float(base["nll"]) - float(withkl["nll"])) < 1e-5, \
        "turning the term on changed the transition score, so the RNG order moved"
    assert float(withkl["x0_kl"]) > 0.0
    # The two terms have to be commensurate or a coefficient of 1 is no term at all. Both are
    # accumulated as a sum over the masked positions of a sequence and a mean over sequences,
    # so they should sit within an order of magnitude of one another. Normalising the KL per
    # token instead leaves it roughly L times too small, which is a silent failure because the
    # run still trains and the diagnostic still looks sensible.
    ratio = float(withkl["x0_kl"]) / float(withkl["nll"])
    assert 0.05 < ratio < 20.0, f"x0_kl is not commensurate with nll, ratio {ratio:.4f}"

    # The gradient must reach the router, because the term is defined on the k-marginal and a
    # term that only reached the component heads would be the per-component KL T3 rules out.
    student.zero_grad()
    withkl["loss"].backward()
    router = [p for n, p in student.named_parameters()
              if "router" in n and p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert router, "no router parameter received gradient from the loss"

    # Finite difference on one scalar of the term itself.
    t = torch.rand(4) * 0.6 + 0.2
    x_t = torch.where(torch.rand(4, L) < t[:, None], x1, torch.full_like(x1, MASK_ID))
    with torch.no_grad():
        log_pt = teacher_x0_logprob(teacher, x_t, t, MASK_ID)
        p_t = log_pt.exp()
    mk = (x_t == MASK_ID).float()

    def kl_of(model):
        log_ps = student_x0_logprob_marginal(model, x_t, t, MASK_ID)
        per_pos = (p_t * (log_pt - log_ps)).sum(-1)
        return (per_pos * mk).sum() / x_t.shape[0]

    student.zero_grad()
    kl_of(student).backward()
    # Differencing a coordinate whose gradient is near zero would pass whatever the term did,
    # so we pick the largest-gradient coordinate of the largest-gradient weight matrix and
    # check that one instead.
    name, param = max(((n, p) for n, p in student.named_parameters()
                       if p.ndim == 2 and p.grad is not None),
                      key=lambda np_: float(np_[1].grad.abs().max()))
    flat = int(param.grad.abs().argmax())
    i, j = flat // param.shape[1], flat % param.shape[1]
    g = param.grad[i, j].item()
    assert abs(g) > 1e-6, f"largest gradient in the model is only {g:.3e}"
    eps = 1e-3
    with torch.no_grad():
        param[i, j] += eps
        up = float(kl_of(student))
        param[i, j] -= 2 * eps
        dn = float(kl_of(student))
        param[i, j] += eps
    fd = (up - dn) / (2 * eps)
    assert abs(fd - g) <= 1e-2 * abs(g), f"grad {g:.6f} vs finite difference {fd:.6f}"
    print(f"T4  OK  x0_kl inert at coef 0, reaches {len(router)} router tensors, "
          f"grad {g:.5f} vs finite difference {fd:.5f} on {name}[{i},{j}]")


# ---------- T5: extra targets cut the variance ----------

def test_t5_n_targets_variance():
    student, teacher = _models()
    x1 = torch.randint(0, MASK_ID, (16, L))

    def spread(n_targets, reps=60):
        vals = []
        for r in range(reps):
            torch.manual_seed(1000 + r)
            d = distill_step_loss(student, teacher, x1, K, SUB, "exact",
                                  _v2(n_targets=n_targets))
            vals.append(float(d["nll"]))
        return np.array(vals)

    a = spread(1)
    b = spread(4)
    # Averaging targets is a variance reduction and not a change of estimand, so the mean has
    # to survive it. The two arms share the (t, s) draw through the seeding above, which makes
    # the comparison of spreads a comparison of the rollout noise alone.
    assert abs(a.mean() - b.mean()) < 0.05 * abs(a.mean()), \
        f"n_targets moved the mean, {a.mean():.4f} -> {b.mean():.4f}"
    va, vb = a.std(), b.std()
    assert vb <= va + 1e-6, f"n_targets=4 did not reduce the spread, {va:.4f} -> {vb:.4f}"
    print(f"T5  OK  mean {a.mean():.4f} -> {b.mean():.4f}, sd {va:.4f} -> {vb:.4f}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    test_t1_defaults_reproduce_reference()
    test_t2_time_sampler()
    test_t3_kl_uses_the_marginal()
    test_t4_kl_gradient()
    test_t5_n_targets_variance()
    print("\nall dense supervision tests passed")
