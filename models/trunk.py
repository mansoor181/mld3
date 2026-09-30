"""DiT trunk with a router head and component-specific top blocks.

The first `depth - latent_last_L` blocks are shared and run once per state. The remaining
blocks run once per evaluated component under the same weights, with the component entering
only through a learned embedding added to the adaLN conditioning vector. A call that asks for
every component therefore costs (depth - latent_last_L) + M * latent_last_L block passes,
and a call that names one component costs `depth`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(0, half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class LayerNorm(nn.Module):
    """Weight-only layer norm evaluated in fp32, as in MDLM."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            y = F.layer_norm(x.float(), self.dim)
        return (y * self.weight).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.scale


class GELUMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim * mult * 2 / 3 // 8) * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.wo = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.silu(self.w1(x)) * self.w2(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        f = torch.einsum("i,j->ij", torch.arange(seq_len, device=device).float(), self.inv_freq)
        return f.cos(), f.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.rot = Rotary(self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        cos, sin = self.rot(T, x.device)
        q, k, v = rope(qkv[0], cos, sin), rope(qkv[1], cos, sin), qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class AdaLN(nn.Module):
    """Per-block scale, shift and gate read off the conditioning vector."""

    def __init__(self, dim: int, cond_dim: int, n_params: int = 6) -> None:
        super().__init__()
        self.map = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_params * dim))
        nn.init.zeros_(self.map[1].weight)
        nn.init.zeros_(self.map[1].bias)
        self.n_params = n_params

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.map(cond).chunk(self.n_params, dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int, cond_dim: int,
                 dropout: float, mdlm: bool) -> None:
        super().__init__()
        norm = LayerNorm if mdlm else RMSNorm
        self.norm1, self.norm2 = norm(dim), norm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = GELUMLP(dim, mlp_mult) if mdlm else SwiGLU(dim, mlp_mult)
        self.ada = AdaLN(dim, cond_dim)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, gate1, g2, b2, gate2 = self.ada(cond)
        h = self.norm1(x) * (1 + g1[:, None]) + b1[:, None]
        x = x + gate1[:, None] * self.drop1(self.attn(h))
        h = self.norm2(x) * (1 + g2[:, None]) + b2[:, None]
        return x + gate2[:, None] * self.drop2(self.mlp(h))


@dataclass
class TrunkConfig:
    vocab_size: int
    seq_len: int
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_mult: int = 4
    latent_M: int = 4
    latent_last_L: int = 4
    time_embed_dim: int = 256
    cond_dim: int = 128
    dropout: float = 0.1
    time_eps: float = 1e-3
    schedule: str = "loglinear"
    # "mdlm" mirrors the MDLM teacher block for block, which is what makes a warm start
    # possible; it conditions on the noise level sigma(t) alone. "lkf" is the ImageNet trunk,
    # which cannot warm-start from MaskGIT and instead conditions on the pair (t, s).
    arch: str = "mdlm"
    tie_embeddings: bool = False
    # The molecule and DNA teachers were trained without time conditioning, and a student
    # warm-started from one has to drop it too.
    time_conditioning: bool = True
    # MDLM applies SiLU once to the conditioning vector; the ImageNet trunk applies it twice.
    cond_extra_silu: bool = True


class Trunk(nn.Module):
    """Returns per-component token logits [B, M, L, V] and router logits [B, M]."""

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mdlm = cfg.arch == "mdlm"
        cd = cfg.cond_dim

        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        if self.mdlm:
            nn.init.kaiming_uniform_(self.tok.weight, a=math.sqrt(5))
        self.pos = None if self.mdlm else nn.Embedding(cfg.seq_len, cfg.dim)

        def time_mlp():
            return nn.Sequential(nn.Linear(cfg.time_embed_dim, cd), nn.SiLU(), nn.Linear(cd, cd))

        self.t_mlp = time_mlp()
        self.s_mlp = None if self.mdlm else time_mlp()
        # One extra row for the shared blocks, which carry no component.
        self.k_emb = nn.Embedding(cfg.latent_M + 1, cd)
        nn.init.normal_(self.k_emb.weight, std=0.02)

        def block():
            return Block(cfg.dim, cfg.heads, cfg.mlp_mult, cd, cfg.dropout, self.mdlm)

        self.shared_blocks = nn.ModuleList(
            [block() for _ in range(cfg.depth - cfg.latent_last_L)])
        self.latent_blocks = nn.ModuleList([block() for _ in range(cfg.latent_last_L)])
        # The component reaches the output only through the adaLN of these blocks, so a
        # zero-initialised map would leave every component identical and with zero gradient
        # to separate them.
        for blk in self.latent_blocks:
            nn.init.normal_(blk.ada.map[1].weight, std=0.02)

        self.final_norm = LayerNorm(cfg.dim) if self.mdlm else RMSNorm(cfg.dim)
        self.final_ada = AdaLN(cfg.dim, cd, n_params=2) if self.mdlm else None
        untied_head = self.mdlm and not cfg.tie_embeddings
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=untied_head)
        if untied_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.router = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.latent_M))

    def _cond(self, t: torch.Tensor, s: torch.Tensor, k: torch.Tensor | None) -> torch.Tensor:
        if self.mdlm:
            if self.cfg.time_conditioning:
                tau = (1.0 - t).clamp(self.cfg.time_eps, 1.0)
                sigma = -torch.log1p(-(1.0 - self.cfg.time_eps) * tau) \
                    if self.cfg.schedule == "loglinear" else tau
            else:
                sigma = torch.zeros_like(t)
            cond = self.t_mlp(sinusoidal_time_emb(sigma, self.cfg.time_embed_dim))
            if self.cfg.cond_extra_silu:
                cond = F.silu(cond)
        else:
            cond = self.t_mlp(sinusoidal_time_emb(t, self.cfg.time_embed_dim)) \
                + self.s_mlp(sinusoidal_time_emb(s, self.cfg.time_embed_dim))
        return cond if k is None else cond + self.k_emb(k)

    def _amp(self, x: torch.Tensor):
        return torch.cuda.amp.autocast(enabled=self.mdlm and x.is_cuda, dtype=torch.bfloat16)

    def shared_stack(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                     return_router: bool = True):
        """Run the component-independent blocks and read the router off their mean-pooled output."""
        with self._amp(x_t):
            x = self.tok(x_t)
            if self.pos is not None:
                x = x + self.pos(torch.arange(x_t.shape[1], device=x_t.device))[None]
            cond = self._cond(t, s, None)
            for blk in self.shared_blocks:
                x = blk(x, cond)
            router = self.router(x.mean(1)) if return_router else None
        return x, (None if router is None else router.float())

    def latent_stack(self, h: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                     components: torch.Tensor | None = None) -> torch.Tensor:
        """Run the component-specific blocks on a shared representation.

        `components` is [B, R] of component indices; None means all M of them. Returns
        [B, R, L, V].
        """
        B, L, D = h.shape
        if components is None:
            components = torch.arange(self.cfg.latent_M, device=h.device).expand(B, -1)
        R = components.shape[1]
        with self._amp(h):
            flat = h[:, None].expand(B, R, L, D).reshape(B * R, L, D)
            rep = lambda v: v[:, None].expand(B, R).reshape(B * R)
            cond = self._cond(rep(t), rep(s), components.reshape(B * R) + 1)
            out = flat
            for blk in self.latent_blocks:
                out = blk(out, cond)
            out = self.final_norm(out)
            if self.final_ada is not None:
                g, b = self.final_ada(cond)
                out = out * (1 + g[:, None]) + b[:, None]
            logits = self.head(out)
        return logits.float().reshape(B, R, L, self.cfg.vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                return_router: bool = True, components: torch.Tensor | None = None):
        h, router = self.shared_stack(x_t, t, s, return_router)
        return self.latent_stack(h, t, s, components), router
