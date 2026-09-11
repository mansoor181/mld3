"""Measure the total correlation a mixture actually induces, not just its identifiability.

The paper reports Ihat = I(k; X_s | x_t). Its own Proposition gives
    TC = sum_i I(k; X_s^i | x_t) - Ihat,
and TC is the quantity tied to correlated sampling. A mixture in which one position tags the
component reaches the Ihat ceiling while inducing TC = 0, so Ihat alone does not certify that
the kernel correlates positions. This measures TC directly.

Two differences from eval_text.transition_mi, both deliberate:
  * x_s is drawn from the STUDENT's own transition rather than from the teacher path, since
    TC is a property of the kernel the student samples from.
  * per-position posteriors over k are accumulated, which is what the identity needs.

Run with --self_test to check it against an exactly enumerable mixture.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.eval_text import domain_of, load_model, loader_module
from models.latent_kernel import mask_forward_noise


def _entropy(p, dim=-1):
    return -(p * p.clamp_min(1e-30).log()).sum(dim)


def tc_from_parts(log_w, gathered, mask_positions=None):
    """log_w: [B,M] log router. gathered: [B,M,L] per-position component log-probs.

    Returns (Ihat, sum_i I(k;X_i), TC) averaged over the batch, in nats.
    """
    if mask_positions is not None:
        gathered = gathered * mask_positions[:, None, :]
    per = gathered.sum(-1)                                   # [B,M]
    w = log_w.exp()
    H_prior = _entropy(w)                                    # [B]

    post_joint = F.softmax(log_w + per, dim=-1)              # [B,M]
    I_joint = H_prior - _entropy(post_joint)                 # [B]

    post_pos = F.softmax(log_w[:, :, None] + gathered, dim=1)   # [B,M,L]
    H_pos = _entropy(post_pos, dim=1)                           # [B,L]
    if mask_positions is not None:
        keep = mask_positions.bool()
        I_pos = ((H_prior[:, None] - H_pos) * keep.float()).sum(-1)
    else:
        I_pos = (H_prior[:, None] - H_pos).sum(-1)              # [B]

    return I_joint.mean().item(), I_pos.mean().item(), (I_pos - I_joint).mean().item()


def self_test():
    """Exact check against an enumerable mixture, matching scripts/verify_mi.py."""
    torch.manual_seed(0)
    L, V, M, B = 3, 3, 2, 200000
    log_w = torch.log(torch.full((B, M), 0.5))
    for name, tag_positions in [("identical", []), ("one_position", [0]),
                                ("all_positions", [0, 1, 2])]:
        # component k puts all mass on symbol k at tagged positions, uniform elsewhere
        P = torch.full((M, L, V), 1.0 / V)
        for k in range(M):
            for i in tag_positions:
                P[k, i] = 0.0
                P[k, i, k] = 1.0
        k = torch.randint(0, M, (B,))
        xs = torch.stack([torch.multinomial(P[k, i], 1).squeeze(-1) for i in range(L)], dim=1)
        gathered = torch.stack(
            [torch.log(P[:, i, :].clamp_min(1e-30))[:, xs[:, i]].T for i in range(L)], dim=2)
        Ih, Ipos, tc = tc_from_parts(log_w, gathered)
        exact_tc = {"identical": 0.0, "one_position": 0.0,
                    "all_positions": 2 * math.log(2)}[name]
        print("%-14s Ihat=%.4f  sum_i I=%.4f  TC=%.4f   (exact TC %.4f)  %s"
              % (name, Ih, Ipos, tc, exact_tc,
                 "OK" if abs(tc - exact_tc) < 0.02 else "MISMATCH"))


@torch.no_grad()
def measure(ckpt, dataset, n_batches, batch_size):
    dev = torch.device("cuda")
    model, cfg, ds_name, meta, step = load_model(ckpt, dev)
    ds_name = dataset or ds_name
    loader, _ = loader_module(domain_of(ds_name)).make_loader(
        ds_name, "val", batch_size, shuffle=False, num_workers=2)
    tot = [0.0, 0.0, 0.0]; n = 0
    for bi, xb in enumerate(loader):
        if bi >= n_batches: break
        x1 = xb[0] if isinstance(xb, (list, tuple)) else xb
        x1 = x1.to(dev); B = x1.shape[0]
        t = torch.rand(B, device=dev) * (1 - cfg.dt_min)
        s = (t + cfg.dt_min + torch.rand(B, device=dev) * (1 - t - cfg.dt_min).clamp_min(0)).clamp(max=1.0)
        x_t = mask_forward_noise(x1, t, cfg.mask_id)
        logits, router_logits = model.trunk(x_t, t, s)          # [B,M,L,V], [B,M]
        log_w = F.log_softmax(router_logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        # x_s drawn from the STUDENT's own transition: pick k ~ w, then sample positions
        k = torch.multinomial(log_w.exp(), 1).squeeze(-1)        # [B]
        pk = logp[torch.arange(B, device=dev), k]                # [B,L,V]
        xs = torch.multinomial(pk.exp().reshape(-1, pk.shape[-1]), 1).reshape(B, -1)
        idx = xs[:, None, :, None].expand(-1, logp.shape[1], -1, 1)
        gathered = torch.gather(logp, -1, idx).squeeze(-1)       # [B,M,L]
        mp = (x_t == cfg.mask_id).float()
        r = tc_from_parts(log_w, gathered, mp)
        for j in range(3): tot[j] += r[j] * B
        n += B
    print("ckpt=%s step=%s  Ihat=%.4f  sum_i I(k;X_i)=%.4f  TC=%.4f  (M=%d, log M=%.3f)"
          % (ckpt.split("/")[-2], step, tot[0]/n, tot[1]/n, tot[2]/n,
             cfg.latent_M, math.log(max(cfg.latent_M, 1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--ckpt"); ap.add_argument("--dataset")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    if a.self_test: self_test()
    else: measure(a.ckpt, a.dataset, a.n_batches, a.batch_size)
