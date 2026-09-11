"""Decoding policies for the mixture flow, priced in transformer-block passes.

The trunk splits into a component-independent stack of $D - L_{lat}$ blocks and a per-component
stack of $L_{lat}$ blocks, so a call that asks for every component costs $C(M) = D - L_{lat} +
M L_{lat}$ block passes while a call that names one component costs only $D$, the same as a
factorized denoiser of equal depth. Every policy here is written against that split and reports
what it actually spends.

Three consequences shape the policies below.

A rollout that has committed to one component never needs another branch, so commit-$k$ costs
$K D$ block passes rather than $K C(M)$, and the mixture student is then priced exactly like the
factorized control at the same step count.

Best-of-$M$ inherits the same saving on each of its $M$ chains, and because all of them start
from the same all-mask state, a single shared stack plus $M$ branches seeds the whole pool. The
policy costs $D - L_{lat} + M L_{lat} + (K-1) M D$, which at $K{=}1$ is one full call for the
entire decode instead of $M$ of them.

In the other direction, a call that does pay for all $M$ branches enumerates all $M$
continuations of the current state at once. A single chain can therefore search over which
component to follow at each step for $C(M)$ per step, which is cheaper than the $M D$ per step
that a best-of-$M$ pool spends whenever $D - L_{lat} + M L_{lat} < M D$, that is whenever the
shared stack is deeper than the per-component stack.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# cost model, in transformer-block passes
# --------------------------------------------------------------------------------------

def block_costs(cfg) -> tuple[int, int, int]:
    """Return (shared-stack cost, per-branch cost, cost of a call that names one component)."""
    Lk = cfg.latent_last_L
    return cfg.depth - Lk, Lk, cfg.depth


def full_call_cost(cfg) -> int:
    """C(M): the cost of one call that evaluates every component."""
    S, Lk, _ = block_costs(cfg)
    return S + cfg.latent_M * Lk


# --------------------------------------------------------------------------------------
# low-level step helpers (masking path, ancestral kernel)
# --------------------------------------------------------------------------------------

def _sample_and_logp(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-max categorical sample plus the log-probability of the drawn token.

    Recovering log p(y) from the logits and their log-sum-exp avoids keeping a full log-softmax
    tensor alongside the logits, which matters because the logits are already [B, M, L, V].
    """
    u = torch.rand_like(logits).clamp_(1e-20, 1.0)
    y = (logits - (-u.log()).log()).argmax(dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1) - lse
    return y, logp


def _advance(x: torch.Tensor, logits: torch.Tensor, mask_id: int,
             rho: float | None = None, last: bool = True):
    """One ancestral step of the masking kernel at the currently masked positions.

    With `rho` set, only the most confident fraction of the proposed reveals is kept.
    """
    y, logp = _sample_and_logp(logits)
    at_mask = (x == mask_id)
    revealed = (at_mask & (y != mask_id)) if rho is None else _gate(x, y, logits, mask_id,
                                                                   rho, last)
    x_new = torch.where(revealed, y, x)
    return x_new, logp, revealed


def _gate(x, y, score_src, mask_id: int, rho: float, last: bool):
    """Accept only the top `rho` fraction of the proposed reveals, ranked by `score_src`.

    Positions that are not accepted go back to the mask, which lets a later step revisit them
    with more context. The final step accepts everything so that the sequence always completes.
    """
    at_mask = (x == mask_id)
    proposed = at_mask & (y != mask_id)
    if last:
        return proposed
    agree = score_src.gather(-1, y.unsqueeze(-1)).squeeze(-1).masked_fill(~proposed, -float("inf"))
    n_keep = torch.ceil(proposed.sum(-1).float() * rho).long()
    order = agree.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device).expand_as(order))
    return proposed & (rank < n_keep[:, None])


def _log_marginal(logits: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """log of the router-weighted marginal denoiser, [B, L, V] from [B, M, L, V]."""
    w = F.softmax(router_logits, dim=-1)
    return torch.log((w[:, :, None, None] * F.softmax(logits, dim=-1)).sum(1).clamp_min(1e-30))


# --------------------------------------------------------------------------------------
# analytic (schedule-enforced) reverse step
# --------------------------------------------------------------------------------------

def step_times(K: int, i: int, batch_size: int, device, analytic: bool):
    """The (t, s) pair a policy hands the trunk at step i of K.

    The ancestral policies ask the kernel for the state at the next grid point and let it choose
    how much to reveal. The analytic policies instead read the clean-data prediction, which lives
    at s = 1, and then impose the reveal rate from the corruption schedule themselves.
    """
    t = torch.full((batch_size,), i / K, device=device)
    s = torch.ones(batch_size, device=device) if analytic \
        else torch.full((batch_size,), (i + 1) / K, device=device)
    return t, s


def analytic_logits(logits: torch.Tensor, cfg, i: int, K: int) -> torch.Tensor:
    """Turn a clean-data prediction into the log of the schedule-enforced reverse kernel.

    We drop the MASK column of the prediction, renormalise what is left to a distribution over
    real tokens, and then place the exact absorbing-schedule reverse on top of it, so a position
    is revealed with probability (mc_t - mc_s) / mc_t and otherwise stays masked. Returning the
    log of that distribution rather than the distribution itself lets every downstream helper,
    which samples by Gumbel-max over logits and ranks by the same quantity, run unchanged.
    """
    mask_id = cfg.mask_id
    p = F.softmax(logits, dim=-1).clone()
    p[..., mask_id] = 0.0
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if getattr(cfg, "schedule", "linear") == "loglinear":
        ome = 1.0 - getattr(cfg, "time_eps", 1e-3)
        mc_t, mc_s = ome * (1.0 - i / K), ome * (1.0 - (i + 1) / K)
    else:
        mc_t, mc_s = 1.0 - i / K, 1.0 - (i + 1) / K
    q = p * (mc_t - mc_s)
    q[..., mask_id] = mc_s
    return torch.log(q.clamp_min(1e-30))


# --------------------------------------------------------------------------------------
# candidate pool: M frozen-component chains with a shared all-mask first step
# --------------------------------------------------------------------------------------

@torch.no_grad()
def rollout_pool(model, K: int, batch_size: int, seq_len: int, device,
                 comps: list[int] | None = None, rho: float | None = None,
                 analytic: bool = False):
    """Roll out one frozen-component chain per entry of `comps`.

    Every chain starts from the same all-mask state, so one shared stack plus one branch per
    component seeds all of them. Every later step evaluates its own shared stack on its own
    state and then only the branch it has committed to. Alongside the candidates we accumulate
    the running evidence of each chain, meaning the log-probability the frozen component assigns
    to the tokens it reveals, which the trunk produces during generation and which therefore
    costs nothing extra to collect. Passing `rho` gates each chain by its own confidence in the
    same way the single-chain gated policy does, which lets us ask whether selection across
    candidates still buys anything once every candidate is already decoded in confidence order.

    Returns (cands [R,B,L], ev [R,B], ev_hist [K,R,B], w0 [B,M], block passes).
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    comps = list(range(cfg.latent_M)) if comps is None else list(comps)
    R = len(comps)
    S, Lk, D = block_costs(cfg)

    x = torch.full((R, batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    ev = torch.zeros(R, batch_size, device=device)
    ev_hist = torch.zeros(K, R, batch_size, device=device)
    last_logits = [None] * R

    # step 0: one shared stack on the common all-mask state, then one branch per component
    t, s = step_times(K, 0, batch_size, device, analytic)
    h, router0 = model.trunk.shared_stack(x[0], t, s)
    w0 = F.softmax(router0, dim=-1)
    for r, m in enumerate(comps):
        ks = torch.full((batch_size,), m, device=device, dtype=torch.long)
        lg = model.trunk.latent_stack(h, t, s, k_select=ks)[:, 0]
        if analytic:
            lg = analytic_logits(lg, cfg, 0, K)
        x[r], logp, revealed = _advance(x[r], lg, mask_id, rho=rho, last=(K == 1))
        ev[r] = (logp * revealed).sum(-1)
        last_logits[r] = lg
    del h
    ev_hist[0] = ev
    cost = S + R * Lk

    # steps 1..K-1: each chain pays its own shared stack and one branch
    for i in range(1, K):
        t, s = step_times(K, i, batch_size, device, analytic)
        for r, m in enumerate(comps):
            ks = torch.full((batch_size,), m, device=device, dtype=torch.long)
            h, _ = model.trunk.shared_stack(x[r], t, s, return_router=False)
            lg = model.trunk.latent_stack(h, t, s, k_select=ks)[:, 0]
            del h
            if analytic:
                lg = analytic_logits(lg, cfg, i, K)
            x[r], logp, revealed = _advance(x[r], lg, mask_id, rho=rho, last=(i == K - 1))
            ev[r] = ev[r] + (logp * revealed).sum(-1)
            last_logits[r] = lg
        ev_hist[i] = ev
        cost += R * D

    for r in range(R):
        still = (x[r] == mask_id)
        if still.any():
            x[r] = torch.where(still, last_logits[r].argmax(dim=-1), x[r])
    return x, ev, ev_hist, w0, cost


@torch.no_grad()
def score_mixture_nelbo(model, cands: torch.Tensor, draws: int = 4):
    """Sequence-level mixture NELBO of each candidate under shared (t, mask) draws.

    This is the selector the headline best-of-$M$ results use. Each draw needs the marginal
    denoiser and therefore every component, so the selector adds $R \\times draws$ full calls on
    top of the pool. Lower is better.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    R, B, L = cands.shape
    device = cands.device
    ones = torch.ones(B, device=device)
    scores = torch.zeros(R, B, device=device)
    for _ in range(draws):
        t = torch.rand(B, device=device).clamp(1e-3, 1 - 1e-3)
        keep = torch.rand(B, L, device=device) < t[:, None]
        weight = (1.0 / (1.0 - t)).clamp_max(1e4)
        for r in range(R):
            x1 = cands[r]
            x_t = torch.where(keep, x1, torch.full_like(x1, mask_id))
            logits, router_logits = model.trunk(x_t, t, ones)
            lp = _log_marginal(logits, router_logits).gather(-1, x1[:, :, None]).squeeze(-1)
            del logits
            masked = (x_t == mask_id).float()
            scores[r] += (weight[:, None] * (-lp) * masked).sum(1) / L
    return scores / max(draws, 1), R * draws * full_call_cost(cfg)


def pick(cands: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = cands.shape[1]
    return cands[idx, torch.arange(B, device=cands.device)]


def successive_halving(cfg, ev_hist: torch.Tensor, K: int, R: int):
    """Prune the pool from R to max(2, R//4) at K/4 and to one at K/2, on running evidence.

    Each chain carries its own noise stream, so removing a chain never perturbs a survivor and
    the whole schedule can be read off one complete pool without resampling. Returns the winner
    index per slot and the block passes the pruned run would have spent.
    """
    S, Lk, D = block_costs(cfg)
    B = ev_hist.shape[-1]
    ar = torch.arange(B, device=ev_hist.device)
    K1, K2 = max(1, K // 4), max(1, K // 2)
    if K > K1 and R > 2:
        keep = max(2, R // 4)
        alive = ev_hist[K1 - 1].argsort(dim=0, descending=True)[:keep]      # [keep, B]
    else:
        alive = torch.arange(R, device=ev_hist.device)[:, None].expand(R, B)
    n2 = alive.shape[0]
    win = alive[ev_hist[K - 1][alive, ar[None, :]].argmax(dim=0), ar]
    steps = max(K1 - 1, 0) * R + max(K2 - K1, 0) * n2 + max(K - K2, 0)
    return win, S + R * Lk + steps * D


# --------------------------------------------------------------------------------------
# single-chain policies
# --------------------------------------------------------------------------------------

@torch.no_grad()
def commit_k_chain(model, K, batch_size, seq_len, device, analytic: bool = False):
    """Draw one component from the router at the all-mask state and hold it.

    The router pools the shared representation, so it is read before any branch runs and the
    committed component is the only branch this policy ever evaluates. Every step therefore
    costs $D$ block passes, which is exactly the cost of a factorized denoiser of the same depth,
    and the whole policy costs $K D$.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    _, _, D = block_costs(cfg)
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    k, lg = None, None
    for i in range(K):
        t, s = step_times(K, i, batch_size, device, analytic)
        h, router = model.trunk.shared_stack(x, t, s, return_router=(k is None))
        if k is None:
            k = torch.multinomial(F.softmax(router, dim=-1), 1).squeeze(-1)
        lg = model.trunk.latent_stack(h, t, s, k_select=k)[:, 0]
        del h
        if analytic:
            lg = analytic_logits(lg, cfg, i, K)
        x, _, _ = _advance(x, lg, mask_id)
    still = (x == mask_id)
    if still.any():
        x = torch.where(still, lg.argmax(dim=-1), x)
    return x, K * D


@torch.no_grad()
def marginal_chain(model, K, batch_size, seq_len, device, analytic: bool = False):
    """Sample each position from the router-weighted marginal of the M components.

    The mixture is removed altogether in this control, because averaging the components at the
    position level returns a product of marginals and destroys exactly the correlation the latent
    is there to carry. It costs one full call per step.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    lg = None
    for i in range(K):
        t, s = step_times(K, i, batch_size, device, analytic)
        logits, router = model.trunk(x, t, s)
        lg = _log_marginal(logits, router)
        del logits
        if analytic:
            lg = analytic_logits(lg, cfg, i, K)
        x, _, _ = _advance(x, lg, mask_id)
    still = (x == mask_id)
    if still.any():
        x = torch.where(still, lg.argmax(dim=-1), x)
    return x, K * full_call_cost(cfg)


@torch.no_grad()
def consensus_chain(model, K, batch_size, seq_len, device, soft: bool = False,
                    analytic: bool = False):
    """Search over the component sequence inside a single chain.

    One full call at the current state enumerates all $M$ continuations, so at every step we draw
    one proposal per component and keep the proposal that the router-weighted mixture itself finds
    most likely, scoring a proposal by the marginal log-probability of the whole transition it
    induces at the masked positions. Because the score covers positions that stay masked as well
    as positions that are revealed, components that reveal different numbers of tokens remain
    comparable. With `soft` the component is drawn from a softmax of the per-masked-token score
    instead of taken at the argmax, which trades part of the gain back for diversity. The policy
    keeps a single chain and costs $K C(M)$.
    """
    cfg = model.cfg
    mask_id, M = cfg.mask_id, cfg.latent_M
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    ar = torch.arange(batch_size, device=device)
    lg_best = None
    for i in range(K):
        t, s = step_times(K, i, batch_size, device, analytic)
        logits, router = model.trunk(x, t, s)                          # [B,M,L,V]
        if analytic:
            logits = analytic_logits(logits, cfg, i, K)
        log_marg = _log_marginal(logits, router)                       # [B,L,V]
        at_mask = (x == mask_id)
        n_mask = at_mask.sum(-1).clamp_min(1)
        props, scores = [], []
        for m in range(M):
            y, _ = _sample_and_logp(logits[:, m])
            props.append(y)
            scores.append((log_marg.gather(-1, y.unsqueeze(-1)).squeeze(-1) * at_mask).sum(-1))
        props = torch.stack(props, 0)                                  # [M,B,L]
        scores = torch.stack(scores, 0)                                # [M,B]
        if soft:
            pm = F.softmax((scores / n_mask[None, :]).transpose(0, 1), dim=-1)
            best = torch.multinomial(pm, 1).squeeze(-1)
        else:
            best = scores.argmax(dim=0)
        lg_best = logits[ar, best]
        del logits, log_marg
        x = torch.where(at_mask, props[best, ar], x)
    still = (x == mask_id)
    if still.any():
        x = torch.where(still, lg_best.argmax(dim=-1), x)
    return x, K * full_call_cost(cfg)


@torch.no_grad()
def gated_commit_chain(model, K, batch_size, seq_len, device, rho: float = 0.5,
                       arbiter: str = "mixture"):
    """Commit to one component but accept only the reveals an arbiter ranks highest.

    A committed component proposes a correlated set of tokens at every step, and a single token
    that the rest of the mixture disagrees with is enough to push the sample off support, which is
    the failure mode the per-step factorization leaves open. We therefore rank the proposed reveals
    by the log-probability an arbiter assigns them, accept the top `rho` fraction, and return the
    rest to the mask so that a later step can revisit them. The last step accepts everything so the
    sequence always completes.

    With `arbiter="mixture"` the ranking uses the router-weighted marginal, meaning that the other
    components decide which of the committed component's proposals to trust, and the policy costs
    $K C(M)$. With `arbiter="self"` the ranking uses the committed component's own confidence, no
    other branch is ever evaluated, and the policy costs $K D$. The second variant is the control
    that separates the value of confidence-ordered unmasking, which any factorized denoiser can
    do, from the value of using the rest of the mixture as the arbiter. It runs unchanged at
    $M{=}1$, where it is exactly confidence-ordered decoding for the factorized student.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    _, _, D = block_costs(cfg)
    use_mix = (arbiter == "mixture") and cfg.latent_M > 1
    x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    ar = torch.arange(batch_size, device=device)
    k, lg = None, None
    for i in range(K):
        t = torch.full((batch_size,), i / K, device=device)
        s = torch.full((batch_size,), (i + 1) / K, device=device)
        if use_mix:
            logits, router = model.trunk(x, t, s)
            if k is None:
                k = torch.multinomial(F.softmax(router, dim=-1), 1).squeeze(-1)
            score_src = _log_marginal(logits, router)
            lg = logits[ar, k]
            del logits
        else:
            h, router = model.trunk.shared_stack(x, t, s, return_router=(k is None))
            if k is None:
                k = torch.multinomial(F.softmax(router, dim=-1), 1).squeeze(-1)
            lg = model.trunk.latent_stack(h, t, s, k_select=k)[:, 0]
            del h
            score_src = lg
        y, _ = _sample_and_logp(lg)
        accept = _gate(x, y, score_src, mask_id, rho, last=(i == K - 1))
        del score_src
        x = torch.where(accept, y, x)
    still = (x == mask_id)
    if still.any():
        x = torch.where(still, lg.argmax(dim=-1), x)
    return x, K * (full_call_cost(cfg) if use_mix else D)
