"""Measured per-step cost of the Di4C arm against the exact-mixture arm on the same trunk.

The compute-matching argument for the Di4C baseline rests on a block-row count that has to be
checked against a stopwatch rather than asserted, because the two objectives spend their budget
in different places. The exact-mixture student pays for an eight-substep teacher rollout and a
single student forward, while Di4C pays for one teacher step and then N student forwards at t
plus N more at s without gradients. We time both here under the real WikiText-103 model shape
and report seconds per step and peak allocated memory, which is what fixes latent_bsize and
micro_batch_size for the launch.

Run:
    CUDA_VISIBLE_DEVICES=1 python tests/probe_di4c_cost.py
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
from training.losses import distill_step_loss

_CFG = str(paths.REPO_DIR / "configs" / "wt103_r2_M8.yaml")
_WARMUP, _ITERS = 5, 20


def _time(fn, mb, seq, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 50257, (mb, seq), device=device)
    try:
        for i in range(_WARMUP + _ITERS):
            if i == _WARMUP:
                torch.cuda.synchronize()
                t0 = time.time()
            out = fn(x)
            out["loss"].backward()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return None, None
    dt = (time.time() - t0) / _ITERS
    return dt, torch.cuda.max_memory_allocated() / 2 ** 30


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
    print(f"{'arm':28s} {'mb':>4s} {'s/step':>9s} {'peak GB':>9s} {'200k days':>10s}")

    rows = []
    for mb in (16, 32):
        dt, gb = _time(lambda x: distill_step_loss(student, teacher, x, K, sub, "exact"),
                       mb, seq, device)
        rows.append(("exact-mixture", mb, dt, gb))
    for nlat in (2, 4, 8):
        for mb in (16, 32):
            dc = dict(cfg["distill"]["di4c"])
            dc["latent_bsize"] = nlat
            dt, gb = _time(lambda x: di4c_step_loss(student, teacher, x, dc), mb, seq, device)
            rows.append((f"di4c latent_bsize={nlat}", mb, dt, gb))

    for name, mb, dt, gb in rows:
        if dt is None:
            print(f"{name:28s} {mb:4d} {'OOM':>9s} {'-':>9s} {'-':>10s}")
            continue
        # The optimizer sees batch_size 256, so a step costs 256/mb micro-steps.
        days = dt * (256 / mb) * 200000 / 86400
        print(f"{name:28s} {mb:4d} {dt:9.4f} {gb:9.2f} {days:10.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
