"""Measured per-step cost of each dense supervision rung against the grid objective and Di4C.

Adding supervision is only a fair comparison if it does not also add compute, and the three dense
changes do not cost the same. Dense time sampling is free, the dense teacher KL buys one extra
teacher forward and one extra student forward, and extra teacher targets buy a further rollout
each. This probe times every rung under the real WikiText-103 model shape so we can set the
sub-step count that holds each rung inside the Di4C arm's measured cost, which is the gate the
launch has to clear rather than a number we assert on paper.

The reference point is the Di4C arm at its launched latent_bsize, since that is the objective
the students have to be level with. A rung that lands above it is not eligible until its
sub-step count is cut.

Run:
    CUDA_VISIBLE_DEVICES=1 python tests/probe_v2_cost.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import yaml

import paths

from models.latent_kernel import LatentKernelFlow, LKFConfig
from models.teacher_adapters import build_mdlm_teacher
from training.di4c import di4c_step_loss
from training.losses import SUPERVISION_DEFAULTS, distill_step_loss

_CFG = str(paths.REPO_DIR / "configs" / "wt103_r2_M8.yaml")
_WARMUP, _ITERS = 5, 20
_MB_CANDIDATES = (8, 16, 32)
_MB_TARGET = 32
_STEPS, _GBS = 200000, 256


def _v2(**kw) -> dict:
    dense = dict(SUPERVISION_DEFAULTS)
    dense.update(kw)
    return dense


def _time(fn, mb, seq, device, model):
    """Time one rung at one micro-batch, leaving no allocation behind for the next rung.

    An OOM inside autograd leaves both the graph and the partially accumulated gradients
    alive, and a later rung would then be timed against a smaller free pool than the one it
    would really have. Clearing the gradients and the cache on every exit is what keeps the
    rungs comparable.
    """
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 50257, (mb, seq), device=device)
    t0 = time.time()
    try:
        for i in range(_WARMUP + _ITERS):
            if i == _WARMUP:
                torch.cuda.synchronize()
                t0 = time.time()
            out = fn(x)
            out["loss"].backward()
            del out
        torch.cuda.synchronize()
        dt = (time.time() - t0) / _ITERS
        gb = torch.cuda.max_memory_allocated() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        dt, gb = None, None
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return dt, gb


def _sweep(fn, seq, device, model):
    """Time a rung at every micro-batch that fits and fit its peak memory against the batch.

    This probe usually has to share a card with a running job, so a micro-batch that OOMs here
    would still fit on a free card and we would wrongly charge the rung for the smaller batch.
    Peak memory is very close to affine in the micro-batch, with the intercept being the
    weights, the gradients and the optimizer state. Fitting that line from the batches that do
    fit lets us report what the rung needs at the target micro-batch rather than guessing.
    """
    pts = []
    for mb in _MB_CANDIDATES:
        dt, gb = _time(fn, mb, seq, device, model)
        if dt is not None:
            pts.append((mb, dt, gb))
    if not pts:
        return None
    if len(pts) >= 2:
        (m0, _, g0), (m1, _, g1) = pts[0], pts[-1]
        slope = (g1 - g0) / (m1 - m0)
        pred = g0 + slope * (_MB_TARGET - m0)
    else:
        pred = float("nan")
    return pts, pred


def main():
    device = torch.device("cuda")
    cfg = yaml.safe_load(open(_CFG))
    mc = cfg["model"]
    seq = 128
    teacher, _ = build_mdlm_teacher(cfg["teacher"]["ckpt"], device=device, seq_len=seq)
    student = LatentKernelFlow(
        LKFConfig(vocab_size=50258, mask_id=50257, seq_len=seq, **mc)).to(device).train()
    K, sub = cfg["distill"]["K"], cfg["distill"]["substeps"]
    print(f"model dim={mc['dim']} depth={mc['depth']} M={mc['latent_M']} "
          f"latent_last_L={mc['latent_last_L']}  K={K} substeps={sub}\n")

    mixed = {"time_sampling": "mixed", "offgrid_frac": 0.5, "horizon": "random",
             "substep_mode": "scaled"}
    rungs = [
        ("R0 grid single term", lambda x: distill_step_loss(
            student, teacher, x, K, sub, "exact", _v2())),
        ("R1 + dense time", lambda x: distill_step_loss(
            student, teacher, x, K, sub, "exact", _v2(**mixed))),
        ("R2 + x0 KL", lambda x: distill_step_loss(
            student, teacher, x, K, sub, "exact", _v2(**mixed, x0_kl_coef=1.0))),
        # The rollout is where the budget goes, and Di4C advances its teacher by a single step
        # of 1/1024, so a student spending 8 teacher sub-steps per training step is already
        # buying more teacher than the baseline it is measured against. Shortening the rollout
        # is therefore the first knob to try, and it moves the arms closer together rather
        # than further apart.
        ("R2 sub=4", lambda x: distill_step_loss(
            student, teacher, x, K, 4, "exact", _v2(**mixed, x0_kl_coef=1.0))),
        ("R2 sub=8 rows=1/4", lambda x: distill_step_loss(
            student, teacher, x, K, sub, "exact",
            _v2(**mixed, x0_kl_coef=1.0, x0_kl_rows=0.25))),
        ("R2 sub=4 rows=1/4", lambda x: distill_step_loss(
            student, teacher, x, K, 4, "exact",
            _v2(**mixed, x0_kl_coef=1.0, x0_kl_rows=0.25))),
        ("R3 sub=4 rows=1/4 tgt2", lambda x: distill_step_loss(
            student, teacher, x, K, 4, "exact",
            _v2(**mixed, x0_kl_coef=1.0, x0_kl_rows=0.25, n_targets=2))),
        # Holding the teacher budget fixed means trading sub-steps against targets, so two
        # targets of two sub-steps each buys the same four teacher forwards that one target of
        # four sub-steps does.
        ("R3 sub=2 rows=1/4 tgt2", lambda x: distill_step_loss(
            student, teacher, x, K, 2, "exact",
            _v2(**mixed, x0_kl_coef=1.0, x0_kl_rows=0.25, n_targets=2))),
        ("R3 sub=2 rows=1/4 tgt4", lambda x: distill_step_loss(
            student, teacher, x, K, 2, "exact",
            _v2(**mixed, x0_kl_coef=1.0, x0_kl_rows=0.25, n_targets=4))),
        (f"di4c latent_bsize={cfg['distill']['di4c']['latent_bsize']}",
            lambda x: di4c_step_loss(student, teacher, x, cfg["distill"]["di4c"])),
    ]

    free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info())
    print(f"card has {free:.1f} GB free of {total:.1f} GB, so a micro-batch may OOM here and "
          f"still fit on a free card\n")

    results = [(name, _sweep(f, seq, device, student)) for name, f in rungs]
    # A rung forced onto a smaller micro-batch pays twice, once for the objective and once for
    # the poorer utilisation, and only the first of those is a property of the objective. We
    # therefore quote the ratio against Di4C at the same micro-batch, and separately report the
    # memory the rung would need to run at the target micro-batch.
    ref_by_mb = {mb: dt for mb, dt, _ in results[-1][1][0]}
    print(f"{'rung':24s} {'mb':>4s} {'s/step':>9s} {'vs di4c @mb':>12s} {'peak GB':>9s} "
          f"{'200k days':>10s}")
    for name, res in results:
        if res is None:
            print(f"{name:24s} {'-':>4s} {'OOM':>9s}")
            continue
        pts, pred = res
        for mb, dt, gb in pts:
            days = dt * (_GBS / mb) * _STEPS / 86400
            print(f"{name:24s} {mb:4d} {dt:9.4f} {dt / ref_by_mb[mb]:11.2f}x {gb:9.2f} "
                  f"{days:10.2f}")
        if pts[-1][0] != _MB_TARGET:
            print(f"{'':24s} {'':4s} {'':9s} {'':12s} needs ~{pred:.0f} GB "
                  f"to reach mb {_MB_TARGET}")


if __name__ == "__main__":
    raise SystemExit(main())
