"""Is the correlation the mixture creates the correlation the data actually has?

TC says how much dependence a kernel induces. It does not say whether that dependence matches
the target. This compares pairwise mutual information between positions for real data, for the
teacher's multi-step rollout (the distillation target) and for the student's own samples. A
student whose pairwise MI sits far above the data's is inventing dependence rather than
reproducing it, which would cost sample quality even while TC looks healthy.
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from eval.eval_text import load_model, domain_of, loader_module
from sampling.policies import commit_k_chain


def pairwise_mi(x, V, pairs, eps=1e-12):
    """Mean I(X_i; X_j) in nats over the given position pairs, from samples x [N,L]."""
    N = x.shape[0]
    tot = 0.0
    for i, j in pairs:
        a, b = x[:, i], x[:, j]
        joint = torch.zeros(V, V, device=x.device)
        joint.index_put_((a, b), torch.ones(N, device=x.device), accumulate=True)
        joint /= joint.sum()
        pi = joint.sum(1, keepdim=True)
        pj = joint.sum(0, keepdim=True)
        nz = joint > 0
        tot += float((joint[nz] * (joint[nz].log() - (pi @ pj)[nz].clamp_min(eps).log())).sum())
    return tot / len(pairs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--pairs", type=int, default=1500)
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
        if sum(r.shape[0] for r in real) >= a.n:
            break
    real = torch.cat(real)[: a.n].to(dev)

    g = torch.Generator(device="cpu").manual_seed(0)
    pi = torch.randint(0, L, (a.pairs,), generator=g)
    pj = torch.randint(0, L, (a.pairs,), generator=g)
    keep = pi != pj
    pairs = list(zip(pi[keep].tolist(), pj[keep].tolist()))

    gen = []
    got = 0
    while got < a.n:
        b = min(256, a.n - got)
        x, _ = commit_k_chain(model, a.K, b, L, dev, analytic=True)
        gen.append(x); got += b
    gen = torch.cat(gen)

    mi_real = pairwise_mi(real, V, pairs)
    mi_gen = pairwise_mi(gen, V, pairs)
    print("%-12s L=%-4d M=%-2d K=%-2d   pairwise MI  data=%.5f  student=%.5f  ratio=%.2fx"
          % (ds, L, M, a.K, mi_real, mi_gen, mi_gen / max(mi_real, 1e-9)))


if __name__ == "__main__":
    main()
