"""Measure how much of a sequence an MLDF student actually reveals per sampling step.

The two WikiText-103 arms are decoded by different algorithms, which is a confound nobody
checked. Di4C runs sample_lkf_analytic, where the reveal rate is imposed by the absorbing
schedule through q(MASK) = mc_s, so at one function evaluation mc_s is zero and every position
is revealed no matter what the network says. MLDF runs sample_lkf, where the reveal rate is
whatever the network's own mask column happens to be, and any position still masked after the
last step is filled by a greedy argmax at sampler.py:72. Greedy fill is the worst possible
decoder for generative perplexity, hence a student that under-reveals is punished twice.

This script measures the two quantities that decide whether that is what happens.

The first is calibration. At (t, s) the absorbing schedule says a masked position stays masked
with probability (1 - s) / (1 - t). We read the student's own marginal mask probability at the
same (t, s) and print the two side by side. A student trained only at a horizon of 1/K has no
reason to be calibrated at any other horizon.

The second is the consequence. We run the real ancestral loop at each budget and report the
fraction of positions that are still masked when the loop ends, which is exactly the fraction
handed to the greedy fallback.

Usage:
  python3.10 scripts/probe_reveal_rate.py --ckpt <ckpt.pt> [--nfe 1,2,4,8,16,32] [--batch 64]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_CODE = Path(__file__).resolve().parents[2]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from eval.eval_text import load_model  # noqa: E402


@torch.no_grad()
def mask_prob_at(model, mask_id, seq_len, batch, device, t_val, s_val):
    """The student's k-marginal probability of leaving a masked position masked at (t, s)."""
    x = torch.full((batch, seq_len), mask_id, device=device, dtype=torch.long)
    t = torch.full((batch,), t_val, device=device)
    s = torch.full((batch,), s_val, device=device)
    logits, router_logits = model.trunk(x, t, s)          # [B,M,L,V], [B,M]
    w = F.softmax(router_logits, dim=-1)                  # [B,M]
    # Read the mask column without materialising the [B,M,L,V] softmax, which is several GB.
    p_mask = (logits[..., mask_id] - logits.logsumexp(dim=-1)).exp()   # [B,M,L]
    return float((w[:, :, None] * p_mask).sum(1).mean())


@torch.no_grad()
def run_ancestral(model, mask_id, seq_len, batch, device, K):
    """The real sample_lkf loop, instrumented for the masked fraction after every step."""
    x = torch.full((batch, seq_len), mask_id, device=device, dtype=torch.long)
    committed_k = None
    per_step = []
    for i in range(K):
        t = torch.full((batch,), i / K, device=device)
        s = torch.full((batch,), (i + 1) / K, device=device)
        logits, router_logits = model.trunk(x, t, s)
        if committed_k is None:
            w = F.softmax(router_logits, dim=-1)
            committed_k = torch.multinomial(w, num_samples=1).squeeze(-1)
        gath = logits[torch.arange(batch, device=device), committed_k]
        mask_positions = x == mask_id
        probs = F.softmax(gath, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(batch, seq_len)
        x = torch.where(mask_positions, sampled, x)
        per_step.append(float((x == mask_id).float().mean()))
    return per_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ds, meta, step = load_model(args.ckpt, device)
    mask_id = cfg.mask_id
    seq_len = args.seq_len or int(meta.get("seq_len", 128))
    print(f"[probe] {args.ckpt}")
    print(f"[probe] dataset={ds} step={step} M={cfg.latent_M} seq_len={seq_len} "
          f"schedule={getattr(cfg, 'schedule', 'linear')}")

    print("\ncalibration of the mask column at t=0, all positions masked")
    print(f"{'s':>8s} {'horizon':>8s} {'schedule 1-s':>13s} {'student':>9s}")
    for s_val in (0.125, 0.25, 0.5, 0.75, 1.0):
        got = mask_prob_at(model, mask_id, seq_len, args.batch, device, 0.0, s_val)
        print(f"{s_val:8.3f} {s_val:8.3f} {1.0 - s_val:13.3f} {got:9.3f}")

    print("\nmasked fraction left to the greedy argmax fallback, real ancestral loop")
    print(f"{'NFE':>5s} {'after each step':<44s} {'greedy-filled':>13s}")
    for K in [int(v) for v in args.nfe.split(",")]:
        per_step = run_ancestral(model, mask_id, seq_len, args.batch, device, K)
        shown = " ".join(f"{v:.2f}" for v in per_step[:10])
        if len(per_step) > 10:
            shown += " ..."
        print(f"{K:5d} {shown:<44s} {per_step[-1]:12.1%}")


if __name__ == "__main__":
    main()
