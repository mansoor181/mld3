"""Is the few-step gain of consensus commit a search over components, or just sharpening?

Consensus takes the argmax over component proposals at every step. Measured on QM9 at one
evaluation it cuts per-position entropy to 0.72 of the data's and normalised pairwise
dependence to 0.50, which is the signature of a mode-seeking decoder rather than of a policy
exploiting correlation. If that is all it does, then lowering the sampling temperature of
commit-k, which no mixture is needed for, should buy the same validity. If consensus still
leads a temperature-matched commit-k, the gain is doing something a factorized model cannot.

We sweep the temperature of commit-k, report validity and the entropy each setting produces,
and read consensus against the commit-k setting that matches its entropy rather than against
the default one.
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
from sampling.policies import step_times, analytic_logits, _advance, consensus_chain
from eval.eval_text import load_model, loader_module, domain_of
from metrics import mol_metrics as MM
from data import mols as moldata


def pos_entropy(x, V):
    N, L = x.shape
    oh = torch.zeros(L, V, device=x.device)
    oh.scatter_add_(1, x.T, torch.ones(L, N, device=x.device))
    oh /= oh.sum(1, keepdim=True)
    return float(-(oh * oh.clamp_min(1e-12).log()).sum(1).mean())


@torch.no_grad()
def commit_k_temp(model, K, B, L, dev, temp=1.0, analytic=True):
    """commit-k with the per-step logits divided by `temp` before sampling."""
    cfg = model.cfg
    x = torch.full((B, L), cfg.mask_id, device=dev, dtype=torch.long)
    k, lg = None, None
    for i in range(K):
        t, s = step_times(K, i, B, dev, analytic)
        h, router = model.trunk.shared_stack(x, t, s, return_router=(k is None))
        if k is None:
            k = torch.multinomial(F.softmax(router, dim=-1), 1).squeeze(-1)
        lg = model.trunk.latent_stack(h, t, s, k_select=k)[:, 0]
        if analytic:
            lg = analytic_logits(lg, cfg, i, K)
        x, _, _ = _advance(x, lg / temp, cfg.mask_id)
    still = (x == cfg.mask_id)
    if still.any():
        x = torch.where(still, lg.argmax(dim=-1), x)
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--Ks", default="1,2,4")
    ap.add_argument("--temps", default="1.0,0.8,0.6,0.4,0.2")
    a = ap.parse_args()

    dev = torch.device("cuda")
    model, cfg, ds_name, meta, step = load_model(a.ckpt, dev)
    ds = a.dataset or ds_name
    L, V = cfg.seq_len, cfg.vocab_size

    loader, _ = loader_module(domain_of(ds)).make_loader(ds, "val", 256, shuffle=False, num_workers=2)
    real = []
    for xb in loader:
        xb = xb[0] if isinstance(xb, (list, tuple)) else xb
        real.append(xb)
        if sum(r.shape[0] for r in real) >= a.n: break
    H_data = pos_entropy(torch.cat(real)[: a.n].to(dev), V)

    def validity(x):
        seqs = moldata.decode(ds, x.cpu().numpy())
        return MM.evaluate([seqs], ds)["valid_pct"]

    def gen(fn):
        out, got = [], 0
        while got < a.n:
            b = min(256, a.n - got)
            r = fn(b)
            out.append(r[0] if isinstance(r, tuple) else r); got += b
        return torch.cat(out)

    print("dataset=%s  M=%d  data entropy=%.4f" % (ds, cfg.latent_M, H_data))
    for K in [int(v) for v in a.Ks.split(",")]:
        print("--- K=%d ---" % K)
        for T in [float(v) for v in a.temps.split(",")]:
            x = gen(lambda b: commit_k_temp(model, K, b, L, dev, temp=T))
            print("  commit-k  T=%-4s  valid=%.4f  H/data=%.2f" %
                  (T, validity(x), pos_entropy(x, V) / H_data))
        x = gen(lambda b: consensus_chain(model, K, b, L, dev, analytic=True))
        print("  consensus  (argmax) valid=%.4f  H/data=%.2f" %
              (validity(x), pos_entropy(x, V) / H_data))


if __name__ == "__main__":
    main()
