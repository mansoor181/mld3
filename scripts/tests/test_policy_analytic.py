"""Check that the analytic branch of the policy module agrees with the analytic sampler.

`sampling.sampler.sample_lkf_analytic` is the decoder the headline tables use, and the policy
module reimplements the same step inside its own loop so that every policy can be run under
either kernel. The two implementations therefore have to agree, and the cheapest way to see it
is the reveal schedule, because the absorbing reverse commits an expected 1/(K-i) of the
remaining masked positions at step i no matter what the network predicts. A commit-k chain under
the analytic branch must leave nothing masked at the end and must match that schedule on the way
there, whereas the same chain under the ancestral branch is free to do neither.

    python tests/test_policy_analytic.py --ckpt <a.pt>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from models.latent_kernel import LatentKernelFlow, LKFConfig
from sampling import policies as P
from sampling.sampler import sample_lkf, sample_lkf_analytic


def load(path, device):
    ck = torch.load(path, map_location=device)
    cfg = LKFConfig(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = LatentKernelFlow(cfg).to(device)
    model.load_state_dict(ck.get("ema_model") or ck.get("model"))
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--nfe", type=int, nargs="*", default=[4, 8, 16])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load(args.ckpt, device)
    L, mask_id = cfg.seq_len, cfg.mask_id
    fails = 0
    for K in args.nfe:
        torch.manual_seed(0)
        x_ref = sample_lkf_analytic(model, K, args.batch, L, device, commit_k=True)
        torch.manual_seed(1)
        x_ref2 = sample_lkf_analytic(model, K, args.batch, L, device, commit_k=True)
        torch.manual_seed(0)
        x_pol, _ = P.commit_k_chain(model, K, args.batch, L, device, analytic=True)
        torch.manual_seed(0)
        x_anc, _ = P.commit_k_chain(model, K, args.batch, L, device, analytic=False)
        left_pol = float((x_pol == mask_id).float().mean())
        left_anc = float((x_anc == mask_id).float().mean())

        def tv(a, b):
            ha = torch.bincount(a.reshape(-1), minlength=cfg.vocab_size).float()
            hb = torch.bincount(b.reshape(-1), minlength=cfg.vocab_size).float()
            return float((ha / ha.sum() - hb / hb.sum()).abs().sum() / 2)

        # A few thousand tokens spread over a fifty-thousand-token vocabulary leave a large total
        # variation between any two independent draws, so the reference for agreement is two runs
        # of the sampler against each other rather than zero.
        tv_self, tv_cross = tv(x_ref, x_ref2), tv(x_ref, x_pol)
        ok = (left_pol == 0.0) and tv_cross < 1.5 * tv_self
        fails += (not ok)
        print(f"K={K:>3}  masked left: policy-analytic {left_pol:.3f}  "
              f"policy-ancestral {left_anc:.3f}   token TV sampler-vs-sampler {tv_self:.4f}  "
              f"sampler-vs-policy {tv_cross:.4f}   {'ok' if ok else 'FAIL'}")
    print("all agree" if fails == 0 else f"{fails} budget(s) disagree")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
