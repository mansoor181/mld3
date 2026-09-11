"""Measure the coupling a student expresses where the target actually puts mass.

For a target q and a model p over x = (x_1..x_L), with p_i the marginals of p, the following
is exact:

    KL(q || p) = sum_i KL(q_i || p_i)  +  TC(q)  -  C_p(q),
    C_p(q) := E_{x ~ q} [ log p(x) - sum_i log p_i(x_i) ].

C_p(q) is the coupling p expresses at points the target generates. Three consequences:
  * p factorized gives C_p(q) = 0 exactly, so it pays TC(q) in full. That is the ceiling.
  * with matched marginals, C_p(q) <= TC(q), with equality only at p = q. The correlation a
    model can usefully express is capped by the correlation the target has.
  * C_p(p) = TC(p). A model whose TC(p) is large while C_p(q) is small is correlating, just
    not where the target lives.

For a finite mixture both terms are closed form, so we measure C_p(q) by drawing the transition
from the teacher and TC(p) by drawing it from the student, using identical code.
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
from models.latent_kernel import mask_forward_noise, mask_two_time_transition


def coupling(log_w, gathered, mask):
    """Return (joint log p(x), marginal sum_i log p_i(x_i), and their difference C)."""
    g = gathered * mask[:, None, :]
    joint = torch.logsumexp(log_w + g.sum(-1), dim=-1)                     # [B]
    marg = (torch.logsumexp(log_w[:, :, None] + gathered, dim=1) * mask).sum(-1)
    return joint, marg, joint - marg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_batches", type=int, default=8); ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    dev = torch.device("cuda")
    model, cfg, ds_name, meta, step = load_model(a.ckpt, dev)
    ds = a.dataset or ds_name
    loader, _ = loader_module(domain_of(ds)).make_loader(ds, "val", a.batch_size, shuffle=False, num_workers=2)

    Cq, Tp, n, NLL, MARG, NTOK = 0.0, 0.0, 0, 0.0, 0.0, 0.0
    for bi, xb in enumerate(loader):
        if bi >= a.n_batches: break
        x1 = xb[0] if isinstance(xb, (list, tuple)) else xb
        x1 = x1.to(dev); B = x1.shape[0]
        t = torch.rand(B, device=dev) * (1 - cfg.dt_min)
        s = (t + cfg.dt_min + torch.rand(B, device=dev) * (1 - t - cfg.dt_min).clamp_min(0)).clamp(max=1.0)
        x_t = mask_forward_noise(x1, t, cfg.mask_id)
        mask = (x_t == cfg.mask_id).float()

        logits, router_logits = model.trunk(x_t, t, s)
        log_w = F.log_softmax(router_logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)

        def gather(xs):
            idx = xs[:, None, :, None].expand(-1, logp.shape[1], -1, 1)
            return torch.gather(logp, -1, idx).squeeze(-1)

        # target transition: what the teacher/data actually produces
        xs_q = mask_two_time_transition(x_t, x1, t, s, cfg.mask_id)
        rev_q = (mask.bool() & (xs_q != cfg.mask_id)).float()
        jq, mq, cq = coupling(log_w, gather(xs_q), rev_q)
        Cq += float(cq.sum()); NLL += float((-jq).sum()); MARG += float((-mq).sum())
        NTOK += float(rev_q.sum())

        # the student's own transition
        k = torch.multinomial(log_w.exp(), 1).squeeze(-1)
        pk = logp[torch.arange(B, device=dev), k]
        xs_p = torch.multinomial(pk.exp().reshape(-1, pk.shape[-1]), 1).reshape(B, -1)
        Tp += float(coupling(log_w, gather(xs_p), rev_q)[2].sum())
        n += B

    print("%-26s C_p(q)=%8.4f  TC(p)=%8.4f  marginalNLL/tok=%.4f  jointNLL/tok=%.4f  gain/tok=%.4f"
          % (a.ckpt.split("/")[-2], Cq/n, Tp/n, MARG/NTOK, NLL/NTOK, (MARG-NLL)/NTOK))


if __name__ == "__main__":
    main()
