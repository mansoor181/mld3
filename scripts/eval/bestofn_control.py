"""Paired best-of-N control: does selection gain come from the mixture or from reranking?

Best-of-M spends M chains to deliver one sample. A factorized student can spend the same M
chains, drawing them i.i.d. from its single component, and rerank them with the same selector.
If the mixture's advantage is the mixture, it has to survive that pairing. If the advantage is
generic reranking, both arms move by the same amount and the mixture keeps only whatever margin
its committed single chain already had.

The two arms are matched by construction. Both draw N candidates from one shared all-mask step
followed by N committed chains, both rank the candidates by the running evidence the trunk
accumulates for free during generation, both deliver n samples, and both are scored by the same
judge. The only difference is that the MIX arm freezes candidate r to component r while the M1
arm freezes every candidate to its only component. Block passes per delivered sample are equal,
since a committed chain costs D at every M.

Usage:
    python scripts/bestofn_control_v2.py --ckpt_m1 <m1.pt> --ckpt_mix <m4.pt> --out <json>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = paths.PROJECT_DIR
sys.path.insert(0, str(REPO_DIR))

from data import text as textdata                                       # noqa: E402
from eval.eval_text import load_model                                   # noqa: E402
from metrics import metrics as M                                        # noqa: E402
from sampling import policies as P                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_m1", required=True)
    ap.add_argument("--ckpt_mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--gen_samples", type=int, default=256)
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--gen_ppl_batch", type=int, default=16)
    ap.add_argument("--gen_ppl_model", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    m1, cfg1, ds_name, _, step1 = load_model(args.ckpt_m1, device)
    mx, cfgx, dsx, _, stepx = load_model(args.ckpt_mix, device)
    assert cfg1.latent_M == 1, f"--ckpt_m1 must be M=1, got {cfg1.latent_M}"
    N = cfgx.latent_M
    seq_len = cfg1.seq_len

    val_loader, _ = textdata.make_loader(ds_name, "val", args.gen_ppl_batch, shuffle=False,
                                         num_workers=2)
    ref = []
    for xb in val_loader:
        ref.append(xb.numpy())
        if sum(a.shape[0] for a in ref) >= max(args.gen_samples, 1024):
            break
    ref = np.concatenate(ref, 0)[: max(args.gen_samples, 1024), :seq_len]

    print(f"[ctrl] ds={ds_name} N={N} n={args.gen_samples} m1_step={step1} mix_step={stepx}",
          flush=True)

    rows = []
    for K in [int(x) for x in args.nfe.split(",") if x]:
        bank: dict[str, list] = {}
        cost: dict[str, float] = {}
        got = 0
        while got < args.gen_samples:
            b = min(args.sample_batch, args.gen_samples - got)
            torch.manual_seed(args.seed + 1000 * K + got)
            for arm, model, cfg in (("M1", m1, cfg1), ("MIX", mx, cfgx)):
                comps = [0] * N if arm == "M1" else list(range(N))
                cands, ev, _, _, pool_cost = P.rollout_pool(model, K, b, seq_len, device,
                                                            comps=comps, analytic=True)
                S, Lk, D = P.block_costs(cfg)
                bank.setdefault(f"{arm} best-of-{N}", []).append(
                    P.pick(cands, ev.argmax(0)).cpu().numpy())
                cost[f"{arm} best-of-{N}"] = pool_cost
                bank.setdefault(f"{arm} single chain", []).append(cands[0].cpu().numpy())
                cost[f"{arm} single chain"] = S + Lk + (K - 1) * D
            got += b

        for name, chunks in bank.items():
            x = np.concatenate(chunks, 0)
            t0 = time.time()
            texts = [tt for tt in textdata.decode(ds_name, x) if len(tt.strip()) > 0]
            gp = M.gen_ppl_gpt2(texts, model_name=args.gen_ppl_model, device=device,
                                batch_size=8)["gen_ppl"] if texts else float("nan")
            ent = M.unigram_entropy(x, cfg1.vocab_size)
            rows.append({"nfe": K, "policy": name, "blocks_per_sample": float(cost[name]),
                         "gen_ppl": float(gp), "unigram_entropy": float(ent),
                         "unigram_kl": float(M.unigram_kl(x, ref, cfg1.vocab_size)),
                         "unique_frac": float(M.unique_fraction(x))})
            print(f"[K={K:>3}] {name:<20} blocks={int(cost[name]):<6} gen_ppl={gp:9.2f} "
                  f"H={ent:.2f}  ({time.time() - t0:.0f}s)", flush=True)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows,
                                   "meta": {"dataset": ds_name, "N": N,
                                            "ckpt_m1": args.ckpt_m1,
                                            "ckpt_mix": args.ckpt_mix,
                                            "gen_samples": args.gen_samples,
                                            "sampler": "analytic"}}, indent=2))
    print(f"[ctrl] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
