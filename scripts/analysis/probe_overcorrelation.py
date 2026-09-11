"""Do our molecule students over-correlate, and does the decoding policy cause it?

At 4 steps the QM9 and ZINC-250k students emit roughly 2.2x the pairwise positional dependence
of their own training data. Three readings are possible and this separates them.

  1. a property of the mixture      -> the M=1 control should not show it
  2. a property of the policy       -> commit-k should differ from the marginal control,
                                       which averages the components away at every step
  3. a property of the step count   -> it should shrink as K grows and each step reveals less

The statistic is the mean mutual information between pairs of positions, measured identically
on held-out data and on generated samples, so a ratio of 1.0 means the sampler reproduces the
data's dependence and above 1.0 means it invents dependence the corpus does not have.
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from eval.eval_text import load_model, domain_of, loader_module
from sampling import policies as P


def pairwise_mi(x, V, pairs, eps=1e-12):
    N = x.shape[0]
    tot = 0.0
    for i, j in pairs:
        a, b = x[:, i], x[:, j]
        J = torch.zeros(V, V, device=x.device)
        J.index_put_((a, b), torch.ones(N, device=x.device), accumulate=True)
        J /= J.sum()
        pi = J.sum(1, keepdim=True); pj = J.sum(0, keepdim=True)
        nz = J > 0
        tot += float((J[nz] * (J[nz].log() - (pi @ pj)[nz].clamp_min(eps).log())).sum())
    return tot / max(len(pairs), 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--pairs", type=int, default=1200)
    ap.add_argument("--Ks", default="1,2,4,8,32")
    a = ap.parse_args()

    dev = torch.device("cuda")
    model, cfg, ds_name, meta, step = load_model(a.ckpt, dev)
    ds = a.dataset or ds_name
    L, V, M = cfg.seq_len, cfg.vocab_size, cfg.latent_M

    loader, _ = loader_module(domain_of(ds)).make_loader(ds, "val", 256, shuffle=False, num_workers=2)
    real = []
    for xb in loader:
        xb = xb[0] if isinstance(xb, (list, tuple)) else xb
        real.append(xb)
        if sum(r.shape[0] for r in real) >= a.n: break
    real = torch.cat(real)[: a.n].to(dev)

    g = torch.Generator().manual_seed(0)
    pi = torch.randint(0, L, (a.pairs,), generator=g)
    pj = torch.randint(0, L, (a.pairs,), generator=g)
    keep = pi != pj
    pairs = list(zip(pi[keep].tolist(), pj[keep].tolist()))
    mi_data = pairwise_mi(real, V, pairs)

    def gen(fn, K):
        out, got = [], 0
        while got < a.n:
            b = min(256, a.n - got)
            r = fn(model, K, b, L, dev, analytic=True)
            out.append(r[0] if isinstance(r, tuple) else r); got += b
        return torch.cat(out)

    pol = {"commit-k": P.commit_k_chain, "marginal": P.marginal_chain,
           "consensus": P.consensus_chain}
    print("dataset=%s L=%d V=%d M=%d step=%s   data pairwise MI = %.5f"
          % (ds, L, V, M, step, mi_data))
    print("%-10s %s" % ("policy", "  ".join("K=%-2d" % k for k in
                                            [int(v) for v in a.Ks.split(",")])))
    for name, fn in pol.items():
        if M == 1 and name in ("marginal", "consensus"):
            continue
        row = []
        for K in [int(v) for v in a.Ks.split(",")]:
            try:
                x = gen(fn, K)
                row.append("%.2fx" % (pairwise_mi(x, V, pairs) / max(mi_data, 1e-12)))
            except Exception as e:
                row.append("err")
        print("%-10s %s" % (name, "  ".join("%-5s" % v for v in row)))


if __name__ == "__main__":
    main()
