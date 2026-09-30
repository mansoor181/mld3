"""Samplers for an MLD3 student.

`spread_consensus` is the sampler of Section 4.3 and is the default everywhere in the paper.
`commit_k` and `best_of_m` are the alternatives tabulated in Appendix E.2.

Costs are quoted in transformer-block passes: the shared stack is D - L_lat blocks and each
evaluated component adds L_lat, so a call over every component costs C(M) = D - L_lat + M*L_lat
while a call naming one component costs D.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.teachers import mask_rate


def full_call_cost(cfg) -> int:
    return cfg.depth - cfg.latent_last_L + cfg.latent_M * cfg.latent_last_L


def bit_reversal_order(length: int) -> list[int]:
    """Positions ordered so that consecutive entries are far apart in the sequence."""
    bits = max(1, (length - 1).bit_length())
    order = (int(format(i, f"0{bits}b")[::-1], 2) for i in range(1 << bits))
    return [i for i in order if i < length]


def reveal_counts(cfg, length: int, K: int) -> list[int]:
    """How many positions each step reveals, so that the schedule empties the sequence in K."""
    m0 = float(mask_rate(cfg, torch.zeros(())))
    remaining = [round(length * float(mask_rate(cfg, torch.tensor((i + 1) / K))) / m0)
                 for i in range(K)]
    remaining[-1] = 0
    counts, prev = [], length
    for r in remaining:
        counts.append(prev - r)
        prev = r
    return counts


def spread_order(length: int, batch_size: int, device) -> torch.Tensor:
    """Bit-reversal order, rotated by an independent random offset per sequence."""
    base = torch.tensor(bit_reversal_order(length), device=device)
    shift = torch.randint(length, (batch_size, 1), device=device)
    return base[(torch.arange(length, device=device)[None] + shift) % length]


def _clean_posterior(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    """log p_0 over the real vocabulary, with the absorbing symbol removed."""
    return F.log_softmax(logits.index_fill(-1, torch.tensor([mask_id], device=logits.device),
                                           -1e4), dim=-1)


@torch.no_grad()
def spread_consensus(model, K: int, batch_size: int, seq_len: int, device,
                     n_components: int = 2):
    """Sample in K steps with the spread order, top-m routing and consensus scoring.

    Each step reveals exactly the number of positions the schedule expects, taken in the spread
    order, so positions revealed together are far apart. Only the `n_components` highest-weight
    components are evaluated; each proposes tokens for those positions, and the chain advances
    with the proposal the renormalised mixture scores highest.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    m = min(n_components, cfg.latent_M)

    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    order = spread_order(seq_len, batch_size, device)
    counts = reveal_counts(cfg, seq_len, K)
    rows = torch.arange(batch_size, device=device)
    filled = 0

    for i, n in enumerate(counts):
        t = torch.full((batch_size,), i / K, device=device)
        s = torch.full((batch_size,), (i + 1) / K, device=device)
        ones = torch.ones(batch_size, device=device)

        h, router = model.trunk.shared_stack(x, t, ones)
        chosen = router.topk(m, dim=-1).indices                          # [B, m]
        log_w = F.log_softmax(router.gather(-1, chosen), dim=-1)         # renormalised over E
        logits = model.trunk.latent_stack(h, t, ones, components=chosen)  # [B, m, L, V]
        del h
        log_p0 = _clean_posterior(logits, mask_id)
        del logits

        # Eq. 2 at the currently masked positions, dropping the normaliser m_t, which is
        # shared by every component and every proposal.
        log_move = torch.log((mask_rate(cfg, t) - mask_rate(cfg, s)).clamp_min(1e-30))
        log_stay = torch.log(mask_rate(cfg, s).clamp_min(1e-30))
        still_masked = x == mask_id

        reveal = order[:, filled:filled + n]                             # [B, n]
        filled += n

        proposals, scores = [], []
        for j in range(m):
            # Only the positions being revealed need a draw.
            at_reveal = log_p0[:, j].gather(
                1, reveal[..., None].expand(-1, -1, log_p0.shape[-1]))
            token = torch.multinomial(at_reveal.exp().reshape(-1, at_reveal.shape[-1]),
                                      1).view(batch_size, n)
            cand = x.scatter(1, reveal, token)
            # log q_k of this proposal at every masked position, then log sum_k w_k q_k.
            gathered = log_p0.gather(-1, cand[:, None, :, None].expand(-1, m, -1, 1)).squeeze(-1)
            log_q = torch.where((cand == mask_id)[:, None], log_stay[:, None, None],
                                gathered + log_move[:, None, None])
            mixed = torch.logsumexp(log_w[:, :, None] + log_q, dim=1)    # [B, L]
            proposals.append(cand)
            scores.append((mixed * still_masked).sum(-1))

        best = torch.stack(scores).argmax(0)
        x = torch.stack(proposals)[best, rows]

    return x, K * (cfg.depth + (m - 1) * cfg.latent_last_L)


@torch.no_grad()
def commit_k(model, K: int, batch_size: int, seq_len: int, device):
    """Draw one component from the router at the all-mask state and hold it for the trajectory.

    Only that component's blocks are ever evaluated, so a step costs D, the same as a
    factorized denoiser of equal depth.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    ones = torch.ones(batch_size, device=device)
    k, p0 = None, None

    for i in range(K):
        t = torch.full((batch_size,), i / K, device=device)
        s = torch.full((batch_size,), (i + 1) / K, device=device)
        h, router = model.trunk.shared_stack(x, t, ones, return_router=(k is None))
        if k is None:
            k = torch.multinomial(F.softmax(router, dim=-1), 1)
        logits = model.trunk.latent_stack(h, t, ones, components=k)[:, 0]
        del h
        p0 = _clean_posterior(logits, mask_id).exp()

        q = p0 * (mask_rate(cfg, t) - mask_rate(cfg, s))[:, None, None]
        q[..., mask_id] = mask_rate(cfg, s)[:, None]
        drawn = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).view(batch_size, seq_len)
        x = torch.where(x == mask_id, drawn, x)

    still = x == mask_id
    if still.any():
        x = torch.where(still, p0.argmax(-1), x)
    return x, K * cfg.depth


@torch.no_grad()
def best_of_m(model, K: int, batch_size: int, seq_len: int, device):
    """Run one commit-k trajectory per component and keep the one with the highest evidence.

    The evidence is the log-probability the frozen component gave the tokens its own trajectory
    revealed, which the trunk already produces during generation.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    M = cfg.latent_M
    ones = torch.ones(batch_size, device=device)
    candidates, evidence = [], []

    for k in range(M):
        x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
        ev = torch.zeros(batch_size, device=device)
        idx = torch.full((batch_size, 1), k, device=device, dtype=torch.long)
        p0 = None
        for i in range(K):
            t = torch.full((batch_size,), i / K, device=device)
            s = torch.full((batch_size,), (i + 1) / K, device=device)
            logits, _ = model.trunk(x, t, ones, return_router=False, components=idx)
            log_p0 = _clean_posterior(logits[:, 0], mask_id)
            p0 = log_p0.exp()

            q = p0 * (mask_rate(cfg, t) - mask_rate(cfg, s))[:, None, None]
            q[..., mask_id] = mask_rate(cfg, s)[:, None]
            drawn = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).view(batch_size, seq_len)
            revealed = (x == mask_id) & (drawn != mask_id)
            ev += (log_p0.gather(-1, drawn[..., None]).squeeze(-1) * revealed).sum(-1)
            x = torch.where(x == mask_id, drawn, x)
        still = x == mask_id
        if still.any():
            x = torch.where(still, p0.argmax(-1), x)
        candidates.append(x)
        evidence.append(ev)

    best = torch.stack(evidence).argmax(0)
    x = torch.stack(candidates)[best, torch.arange(batch_size, device=device)]
    return x, full_call_cost(cfg) + (K - 1) * M * cfg.depth
