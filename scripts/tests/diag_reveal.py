"""Trace what the ancestral rollout actually does, step by step, at a given NFE.

The generative-perplexity curve tells us a student is bad at four evaluations without telling
us why. This walks the same rollout `sampling.sampler.sample_lkf` performs and reports, after
every step, the fraction of positions still carrying MASK together with the mass the kernel
head places on the MASK column at the positions that are still masked. Those two numbers
separate the two ways a few-step rollout can fail. A student that empties the mask in one step
is committing every position at once and its later steps are wasted, whereas a student that
never empties it is leaving positions to the greedy fill at the end.

    python tests/diag_reveal.py --ckpt <a.pt> [--ckpt <b.pt>] --nfe 4 --nfe 8

Reads nothing but the checkpoint and writes nothing.
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
    sd = ck.get("ema_model") or ck.get("model")
    model.load_state_dict(sd)
    model.eval()
    sup = (ck.get("distill") or {}).get("sup") or (ck.get("distill") or {}).get("v2")  # old checkpoints use the "v2" key
    return model, cfg, ck.get("step"), sup


@torch.no_grad()
def trace(model, K: int, B: int, L: int, device, commit_k: bool):
    """One ancestral rollout at NFE=K, returning the per-step reveal trace."""
    cfg = model.cfg
    mask_id = cfg.mask_id
    x = torch.full((B, L), mask_id, device=device, dtype=torch.long)
    committed = None
    rows = []
    for i in range(K):
        t = torch.full((B,), i / K, device=device)
        s = torch.full((B,), (i + 1) / K, device=device)
        logits, router_logits = model.trunk(x, t, s)
        if commit_k and committed is not None:
            k_ids = committed
        else:
            k_ids = torch.multinomial(F.softmax(router_logits, dim=-1), 1).squeeze(-1)
            if commit_k:
                committed = k_ids
        gath = logits[torch.arange(B, device=device), k_ids]
        probs = F.softmax(gath, dim=-1)
        still = (x == mask_id)
        # The mass the kernel puts on MASK, averaged over the positions that are still masked,
        # is the per-step hold rate the network chose for itself at this horizon.
        p_mask = probs[..., mask_id]
        hold = float(p_mask[still].mean()) if int(still.sum()) > 0 else float("nan")
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, L)
        x = torch.where(still, sampled, x)
        rows.append((i, float((x == mask_id).float().mean()), hold))
    return rows, float((x == mask_id).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--nfe", action="append", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--commit_k", action="store_true", default=True)
    args = ap.parse_args()
    nfes = args.nfe or [4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.ckpt:
        model, cfg, step, sup = load(path, device)
        name = Path(path).parent.name
        print(f"\n=== {name} step={step} M={cfg.latent_M} "
              f"time_sampling={(sup or {}).get('time_sampling', 'grid')}")
        for K in nfes:
            rows, final = trace(model, K, args.batch, args.seq_len, device, args.commit_k)
            body = "  ".join(f"{i + 1}:{frac:.3f}/{hold:.3f}" for i, frac, hold in rows)
            print(f"  NFE={K:2d} step:maskfrac/P(MASK)  {body}   final_masked={final:.3f}")


if __name__ == "__main__":
    main()
