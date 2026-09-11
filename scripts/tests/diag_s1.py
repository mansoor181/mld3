"""Probe the terminal condition of a two-time kernel, per latent component.

The rollout ends on the pair (t, s=1), and s=1 means every remaining position is revealed, so
a correct kernel puts zero mass on the MASK column there. This feeds partially masked val-like
inputs at a sweep of t with s held at 1 and reports the mass each component places on MASK at
the positions that are still masked. It also reports the same quantity at s = t + 1/K, where a
positive mass is correct, which separates a kernel that has lost the terminal condition from
one that emits no MASK anywhere.

    python tests/diag_s1.py --ckpt <a.pt> [--ckpt <b.pt>]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from models.latent_kernel import LatentKernelFlow, LKFConfig


def load(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = LKFConfig(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = LatentKernelFlow(cfg).to(device)
    model.load_state_dict(ck.get("ema_model") or ck.get("model"))
    model.eval()
    return model, cfg, ck.get("step"), (ck.get("distill") or {}).get("sup") or (ck.get("distill") or {}).get("v2")  # old checkpoints use the "v2" key


@torch.no_grad()
def p_mask_at(model, x_t, t_val, s_val, device):
    """Mean MASK mass per component at the still-masked positions."""
    B, L = x_t.shape
    t = torch.full((B,), t_val, device=device)
    s = torch.full((B,), s_val, device=device)
    logits, _ = model.trunk(x_t, t, s)                        # [B, M, L, V]
    p = F.softmax(logits, dim=-1)[..., model.cfg.mask_id]     # [B, M, L]
    sel = (x_t == model.cfg.mask_id)[:, None, :].expand_as(p)
    return (p * sel).sum((0, 2)) / sel.sum((0, 2)).clamp_min(1)   # [M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    for path in args.ckpt:
        model, cfg, step, sup = load(path, device)
        print(f"\n=== {Path(path).parent.name} step={step} M={cfg.latent_M} "
              f"time_sampling={(sup or {}).get('time_sampling', 'grid')}")
        # Random tokens outside the mask id stand in for revealed context. The terminal
        # condition is a property of the kernel head and does not depend on the context being
        # real text, which keeps this probe free of any data dependency.
        V = cfg.vocab_size if hasattr(cfg, "vocab_size") else 50257
        base = torch.randint(0, min(V, 50256), (args.batch, args.seq_len), device=device)
        for i in range(args.K):
            t_val = i / args.K
            keep = torch.rand(args.batch, args.seq_len, device=device) < t_val
            x_t = torch.where(keep, base, torch.full_like(base, cfg.mask_id))
            pm_1 = p_mask_at(model, x_t, t_val, 1.0, device)
            pm_h = p_mask_at(model, x_t, t_val, min(1.0, t_val + 1.0 / args.K), device)
            f = lambda v: " ".join(f"{x:.3f}" for x in v.tolist())
            print(f"  t={t_val:.2f}  P(MASK) at s=1   per k: {f(pm_1)}")
            if i < args.K - 1:
                print(f"           P(MASK) at s=t+1/K per k: {f(pm_h)}")


if __name__ == "__main__":
    main()
