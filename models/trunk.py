"""Bidirectional transformer trunk with time-conditioning + late latent injection.

Design principles:
  * MDLM/DiT-style: pre-norm transformer, RMSNorm, SwiGLU MLP, rotary attention.
  * Two-time conditioning (t, s) via adaLN (feature-wise affine) applied on every block.
  * The latent k ∈ {1..M} is injected only into the LAST L_k blocks via FiLM/adaLN so that
    marginalizing over M is only ~M × (L_k / L) of a full pass, not M × full passes.
  * All import statements at top-of-file per project convention.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------ small helpers ------------

def sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for t ∈ [0,1]-valued scalar times."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


class MDLMLayerNorm(nn.Module):
    """MDLM's custom LayerNorm: fp32 F.layer_norm, weight-only affine (no bias)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            y = F.layer_norm(x.float(), self.dim)
        return (y * self.weight).to(x.dtype)


class GELUMLP(nn.Module):
    """MDLM DDiTBlock MLP: Linear(dim, mult*dim, bias) -> GELU(tanh) -> Linear(mult*dim, dim, bias)."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim, bias=True)
        self.fc2 = nn.Linear(mult * dim, dim, bias=True)

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
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        f = torch.einsum("i,j->ij", t, self.inv_freq)
        return f.cos(), f.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.

    x:  [B, H, T, D] with D even.
    cos, sin: [T, D/2].
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    # broadcast cos, sin over B and H
    c = cos[None, None]                    # [1,1,T,D/2]
    s = sin[None, None]                    # [1,1,T,D/2]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class MHA(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.rot = Rotary(self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rot(T, x.device)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        # scaled_dot_product_attention: bidirectional (no causal mask)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)


# ------------ adaLN blocks ------------

class AdaLNCond(nn.Module):
    """Produce per-block gamma/beta from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int, n_params: int = 6) -> None:
        super().__init__()
        # scale, shift for attn + mlp; plus 2 gate params
        self.map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, n_params * dim, bias=True),
        )
        # zero-init like DiT: the first forward pass acts as identity
        nn.init.zeros_(self.map[1].weight)
        nn.init.zeros_(self.map[1].bias)
        self.n_params = n_params
        self.dim = dim

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p = self.map(cond)                             # [B, n_params*dim]
        return p.chunk(self.n_params, dim=-1)          # each [B, dim]


class DiTBlock(nn.Module):
    """DiT-style block: adaLN modulation from a conditioning vector.

    dropout > 0 mirrors MDLM's DDiTBlock placement exactly: one Dropout on the attention
    output and one on the MLP output, applied INSIDE the gated residual add
    (MDLM: residual + scale * dropout(out)). 0.0 = legacy behavior (identity).
    """

    def __init__(self, dim: int, heads: int, mlp_mult: int, cond_dim: int,
                 dropout: float = 0.0, mdlm_dit: bool = False) -> None:
        super().__init__()
        # mdlm_dit: MDLM-faithful block internals, fp32 weight-only LayerNorm plus GELU(tanh)
        # 4x-MLP with biases (MDLM DDiTBlock). Legacy lkf: RMSNorm + SwiGLU (no biases).
        norm_cls = MDLMLayerNorm if mdlm_dit else RMSNorm
        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim)
        self.attn = MHA(dim, heads)
        self.mlp = GELUMLP(dim, mult=mlp_mult) if mdlm_dit else SwiGLU(dim, mult=mlp_mult)
        self.ada = AdaLNCond(dim, cond_dim, n_params=6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gate1, gamma2, beta2, gate2 = self.ada(cond)
        h = self.norm1(x) * (1 + gamma1[:, None]) + beta1[:, None]
        x = x + gate1[:, None] * self.drop1(self.attn(h))
        h = self.norm2(x) * (1 + gamma2[:, None]) + beta2[:, None]
        x = x + gate2[:, None] * self.drop2(self.mlp(h))
        return x


# ------------ full trunk with latent injection ------------

@dataclass
class TrunkConfig:
    vocab_size: int
    seq_len: int
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_mult: int = 4
    latent_M: int = 4
    latent_last_L: int = 4          # inject k into the last L_k blocks
    time_embed_dim: int = 128
    tie_embeddings: bool = True
    # Trunk architecture variant:
    #   "lkf":      two-time (t,s) conditioning, learned plus rotary position, no final adaLN.
    #   "mdlm_dit": MDLM-DiT-faithful single-time SIGMA conditioning, which drops the s pathway,
    #                rotary-only (no learned abs pos), + a final adaLN layer before the head.
    #                Used to test whether the LKF trunk (not the objective) is the gen-PPL gap.
    trunk_arch: str = "lkf"
    time_eps: float = 1e-3          # loglinear eps (for the mdlm_dit sigma conditioning)
    schedule: str = "loglinear"
    # MDLM-native tricks (defaults = legacy behavior, all off):
    dropout: float = 0.0            # MDLM DDiTBlock dropout (native small: 0.1)
    # adaLN conditioning width. 0 = legacy (cond_dim = dim). MDLM small uses a NARROW
    # cond_dim=128, which is most of the param difference vs our legacy blocks.
    cond_dim: int = 0
    # The legacy trunk passes the conditioning vector through SiLU twice, once when the vector
    # is built and again inside AdaLNCond, where MDLM applies it once. A from-scratch run does
    # not care, but the second application blocks an exact weight transfer out of an MDLM
    # checkpoint, and a warm-started run therefore sets this to False.
    cond_extra_silu: bool = True
    # PairFlow trains its molecule and DNA MDLM teachers with time conditioning switched off,
    # which zeroes sigma before it reaches the backbone. A student warm-started from one of
    # those teachers has to zero it as well.
    time_conditioning: bool = True


class LatentKernelTrunk(nn.Module):
    """Trunk producing per-position, per-latent logits.

    forward(x_t, t, s) → logits of shape [B, M, L, V]:
        the [B, k, :, :] slice is the factorized denoiser P^θ_i(· | x_t, k, t, s).

    The last `latent_last_L` blocks are run once per latent k (marginalization cost);
    all earlier blocks are shared and run once per batch.
    """

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mdlm_dit = (getattr(cfg, "trunk_arch", "lkf") == "mdlm_dit")
        # Conditioning width: 0 = legacy (= dim); MDLM small uses a narrow cond_dim=128.
        cd = getattr(cfg, "cond_dim", 0) or cfg.dim
        self.cd = cd
        dropout = getattr(cfg, "dropout", 0.0)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        if self.mdlm_dit:
            # MDLM EmbeddingLayer init: kaiming_uniform_(a=sqrt(5)), not N(0,1).
            nn.init.kaiming_uniform_(self.tok.weight, a=math.sqrt(5))
        # Learned absolute position: only in the LKF variant. mdlm_dit is rotary-only (like MDLM).
        self.pos = None if self.mdlm_dit else nn.Embedding(cfg.seq_len, cfg.dim)
        # Conditioning: [t_emb, s_emb, k_emb] → cond vec. In mdlm_dit we drop the s pathway and feed
        # the noise level sigma (not raw t) through t_mlp, which is MDLM's single-time conditioning.
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.s_mlp = None if self.mdlm_dit else nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.k_emb = nn.Embedding(cfg.latent_M + 1, cd)  # +1 for "no-latent" (shared blocks)
        # Symmetry break (fix #1a): distinct small-random per-latent embeddings so the M
        # components are NOT identical at step 0. (Zero-init here + zero-init adaLN below made
        # ∂logits/∂k a product of two zeros → components could never differentiate → collapse.)
        nn.init.normal_(self.k_emb.weight, mean=0.0, std=0.02)
        # Shared blocks (no latent):
        shared_depth = cfg.depth - cfg.latent_last_L
        assert shared_depth >= 0 and cfg.latent_last_L >= 0
        self.shared_blocks = nn.ModuleList([
            DiTBlock(cfg.dim, cfg.heads, cfg.mlp_mult, cd, dropout=dropout,
                     mdlm_dit=self.mdlm_dit)
            for _ in range(shared_depth)
        ])
        self.latent_blocks = nn.ModuleList([
            DiTBlock(cfg.dim, cfg.heads, cfg.mlp_mult, cd, dropout=dropout,
                     mdlm_dit=self.mdlm_dit)
            for _ in range(cfg.latent_last_L)
        ])
        # Symmetry break (fix #1b): the latent conditioning only acts through the latent
        # blocks' adaLN. DiT zero-inits that map for stability, but with a zero map the
        # k-embedding has zero effect at init. Give the LATENT blocks' adaLN a small non-zero
        # init so the latent modulates the output at first-order from step 0. (Shared blocks
        # keep DiT zero-init, since they carry only the t,s conditioning where stability matters.)
        for blk in self.latent_blocks:
            nn.init.normal_(blk.ada.map[1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(blk.ada.map[1].bias)
        self.final_norm = MDLMLayerNorm(cfg.dim) if self.mdlm_dit else RMSNorm(cfg.dim)
        # MDLM's DDitFinalLayer modulates the final norm by the conditioning before projecting.
        # Zero-init (n_params=2: shift/scale) -> identity at start, like DiT. Only in mdlm_dit.
        self.final_ada = AdaLNCond(cfg.dim, cd, n_params=2) if self.mdlm_dit else None
        # Output heads. MDLM's DDitFinalLayer linear has bias=True and ZERO-inits both weight
        # and bias (uniform initial logits); only meaningful when untied.
        use_mdlm_head = self.mdlm_dit and not cfg.tie_embeddings
        self.token_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=use_mdlm_head)
        if use_mdlm_head:
            nn.init.zeros_(self.token_head.weight)
            nn.init.zeros_(self.token_head.bias)
        if cfg.tie_embeddings:
            self.token_head.weight = self.tok.weight
        # Router head: pool trunk hidden → weights over M latents (unnormalized logits)
        self.router = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.latent_M))

    # ---- conditioning ----

    def _cond(self, t: torch.Tensor, s: torch.Tensor, k: torch.Tensor | None) -> torch.Tensor:
        """Build the adaLN conditioning vector for a block.

        lkf:      cond = t_mlp(emb(t)) + s_mlp(emb(s)) [+ k_emb(k)]   (two-time, raw t & s).
        mdlm_dit: cond = t_mlp(emb(sigma)) [+ k_emb(k)]              (single-time noise level).
                  The trunk-time t is the fraction KEPT (kappa_t=t), so the MDLM noise time is
                  tau = 1 - t and sigma = -log1p(-(1-eps)*tau) for the loglinear schedule. Feeding
                  sigma (range ~0..-log(eps)) instead of t in [0,1] gives MDLM's high-resolution
                  noise conditioning. The s argument is ignored (x0-mode always passes s=1).
        """
        if self.mdlm_dit:
            if getattr(self.cfg, "time_conditioning", True):
                tau = (1.0 - t).clamp(min=self.cfg.time_eps, max=1.0)
                if self.cfg.schedule == "loglinear":
                    sigma = -torch.log1p(-(1.0 - self.cfg.time_eps) * tau)
                else:
                    sigma = tau
            else:
                sigma = torch.zeros_like(t)
            # MDLM: DIT.forward applies an EXTRA F.silu on top of TimestepEmbedder's output
            # (c = F.silu(self.sigma_map(sigma))).
            cond = self.t_mlp(sinusoidal_time_emb(sigma, self.cfg.time_embed_dim))
            if getattr(self.cfg, "cond_extra_silu", True):
                cond = F.silu(cond)
            if k is not None:
                cond = cond + self.k_emb(k)
            return cond
        t_e = self.t_mlp(sinusoidal_time_emb(t, self.cfg.time_embed_dim))
        s_e = self.s_mlp(sinusoidal_time_emb(s, self.cfg.time_embed_dim))
        cond = t_e + s_e
        if k is not None:
            cond = cond + self.k_emb(k)
        return cond

    # ---- forward ----

    def shared_stack(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                     return_router: bool = True):
        """Run the `depth - latent_last_L` component-independent blocks and read the router.

        The router pools the shared representation, so it is available before any latent
        branch has been evaluated. Splitting the trunk here lets a decoding policy pay for the
        shared stack once and then evaluate only the components it actually needs, which is
        what makes a committed rollout cost exactly `depth` block passes per step instead of
        `depth - latent_last_L + latent_last_L * M`.

        Returns (h_shared [B, L, D], router_logits [B, M] or None).
        """
        B, L = x_t.shape
        use_amp = self.mdlm_dit and x_t.is_cuda
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            x = self.tok(x_t)
            if self.pos is not None:
                x = x + self.pos(torch.arange(L, device=x_t.device))[None]
            cond_shared = self._cond(t, s, k=None)
            for blk in self.shared_blocks:
                x = blk(x, cond_shared)
            router_logits = self.router(x.mean(dim=1)) if return_router else None
        return x, (None if router_logits is None else router_logits.float())

    def latent_stack(self, h: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                     k_select: torch.Tensor | None = None) -> torch.Tensor:
        """Run the last `latent_last_L` blocks and the token head on a shared representation.

        With `k_select` set to a [B] index vector only that component is evaluated and the
        result is [B, 1, L, V]. With `k_select` left as None every component is evaluated and
        the result is [B, M, L, V].
        """
        B, L, _ = h.shape
        M = self.cfg.latent_M
        R = M if k_select is None else 1
        use_amp = self.mdlm_dit and h.is_cuda
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            if k_select is None:
                h_tiled = h.unsqueeze(1).expand(B, M, L, self.cfg.dim).reshape(B * M, L, self.cfg.dim)
                t_tiled = t.unsqueeze(1).expand(B, M).reshape(B * M)
                s_tiled = s.unsqueeze(1).expand(B, M).reshape(B * M)
                k_ids = torch.arange(1, M + 1, device=h.device).unsqueeze(0).expand(B, M).reshape(B * M)
            else:
                h_tiled, t_tiled, s_tiled = h, t, s
                k_ids = k_select.to(torch.long) + 1
            cond_k = self._cond(t_tiled, s_tiled, k_ids)
            out = h_tiled
            for blk in self.latent_blocks:
                out = blk(out, cond_k)
            out = self.final_norm(out)
            if self.final_ada is not None:             # MDLM DDitFinalLayer modulation
                gamma, beta = self.final_ada(cond_k)
                out = out * (1 + gamma[:, None]) + beta[:, None]
            logits = self.token_head(out)              # [B*R, L, V]
        return logits.float().reshape(B, R, L, self.cfg.vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor,
                return_router: bool = True,
                k_select: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x_t: [B, L] int64;  t, s: [B] floats in [0,1].

        Args:
            k_select: optional [B] int64 in {0..M-1}. When given, only that latent branch is
                run through the last `latent_last_L` blocks and the token head, so the call
                costs (depth - latent_last_L) + latent_last_L block passes on B rows instead
                of (depth - latent_last_L) + latent_last_L * M. The router still comes from
                the shared stack, which is where it is read. A rollout that has already
                committed to one component needs no other branch, so this is the cheap path
                for every decoding policy that holds k fixed along a trajectory.

        Returns:
            logits: [B, M, L, V], the per-component factorized denoiser logits, or [B, 1, L, V]
                when k_select is given
            router_logits: [B, M] or None
        """
        # MDLM's DIT.forward hardcodes torch.cuda.amp.autocast(dtype=bfloat16) around its
        # blocks + output layer, which shared_stack and latent_stack each mirror.
        h, router_logits = self.shared_stack(x_t, t, s, return_router=return_router)
        logits = self.latent_stack(h, t, s, k_select=k_select)
        return logits, router_logits
