"""Warm-start a Latent-Kernel student from a trained MDLM DiT backbone.

The molecule and DNA students get a tenth of the teacher's step budget, which only buys a
useful model if the student starts from the teacher rather than from scratch. The two
architectures agree block for block once three bookkeeping differences are undone. MDLM
packs its adaLN outputs as (shift, scale, gate) per sub-layer where our AdaLNCond emits
(scale, shift, gate), MDLM builds its sinusoidal time embedding as cos followed by sin
where ours is sin followed by cos, and MDLM passes the conditioning vector through SiLU
once where the legacy trunk passes it through twice.

Given a student with `depth = D` and `latent_last_L = L`, the first `D - L` teacher blocks
land in `shared_blocks` and the last `L` are broadcast into every one of the `M` latent
branches. With the latent embedding zeroed the resulting student reproduces the teacher
exactly under any single-component route, which `tests/test_warm_start.py` checks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Our AdaLNCond emits (scale, shift, gate) per sub-layer; MDLM emits (shift, scale, gate).
_ADALN6_PERM = [1, 0, 2, 4, 3, 5]
# The final layer emits (scale, shift) here and (shift, scale) there.
_ADALN2_PERM = [1, 0]


def _permute_blocks(w: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Reorder the row blocks of an adaLN projection so it matches our chunk order."""
    n = len(perm)
    if w.shape[0] % n:
        raise ValueError(f"adaLN output {w.shape[0]} not divisible by {n}")
    chunks = list(w.chunk(n, dim=0))
    return torch.cat([chunks[i] for i in perm], dim=0)


@torch.no_grad()
def _copy_block(dst: nn.Module, src: nn.Module) -> None:
    """Copy one MDLM DDiTBlock into one of our DiTBlocks."""
    dst.norm1.weight.copy_(src.norm1.weight)
    dst.norm2.weight.copy_(src.norm2.weight)
    dst.attn.qkv.weight.copy_(src.attn_qkv.weight)
    dst.attn.o.weight.copy_(src.attn_out.weight)
    dst.mlp.fc1.weight.copy_(src.mlp[0].weight)
    dst.mlp.fc1.bias.copy_(src.mlp[0].bias)
    dst.mlp.fc2.weight.copy_(src.mlp[2].weight)
    dst.mlp.fc2.bias.copy_(src.mlp[2].bias)
    dst.ada.map[1].weight.copy_(_permute_blocks(src.adaLN_modulation.weight, _ADALN6_PERM))
    dst.ada.map[1].bias.copy_(_permute_blocks(src.adaLN_modulation.bias, _ADALN6_PERM))


@torch.no_grad()
def warm_start_from_mdlm(student: nn.Module, teacher_net: nn.Module,
                         k_noise: float = 0.02, verbose: bool = True) -> None:
    """Copy a trained MDLM DiT backbone into a Latent-Kernel student in place.

    `student` is a LatentKernelFlow whose trunk uses `trunk_arch="mdlm_dit"`,
    `cond_extra_silu=False` and a matching width, depth, head count, conditioning width and
    vocabulary. `teacher_net` is the DiT the teacher adapter holds in `.net`, already
    carrying its EMA weights. `k_noise` sets the standard deviation of the latent embedding
    after the copy, and zero leaves the M branches exactly equal to the teacher.
    """
    trunk = student.trunk
    cfg = trunk.cfg
    if not getattr(trunk, "mdlm_dit", False):
        raise ValueError("warm start requires the student trunk_arch to be mdlm_dit")
    if getattr(cfg, "cond_extra_silu", True):
        raise ValueError(
            "warm start requires cond_extra_silu=False; the legacy trunk applies SiLU twice "
            "and would not reproduce the teacher")
    if getattr(cfg, "time_conditioning", True):
        print("[warm-start] WARNING: the student is time-conditioned while the two time "
              "embeddings use different frequency grids, so the copy is approximate on the "
              "time pathway only")

    src_blocks = list(teacher_net.blocks)
    n_shared = cfg.depth - cfg.latent_last_L
    if len(src_blocks) != cfg.depth:
        raise ValueError(f"teacher has {len(src_blocks)} blocks, student depth is {cfg.depth}")

    # Token embedding. MDLM keeps a bare Parameter where we keep an nn.Embedding.
    trunk.tok.weight.copy_(teacher_net.vocab_embed.embedding)
    if trunk.pos is not None:
        raise ValueError("warm start expects a rotary-only student (pos must be None)")

    # Time pathway. Our sinusoidal embedding is sin followed by cos and MDLM's is the other
    # way round, so the first linear's input columns swap halves.
    half = cfg.time_embed_dim // 2
    w = teacher_net.sigma_map.mlp[0].weight
    if w.shape[1] != cfg.time_embed_dim:
        raise ValueError(f"teacher time embedding is {w.shape[1]} wide, student is "
                         f"{cfg.time_embed_dim}")
    trunk.t_mlp[0].weight.copy_(torch.cat([w[:, half:], w[:, :half]], dim=1))
    trunk.t_mlp[0].bias.copy_(teacher_net.sigma_map.mlp[0].bias)
    trunk.t_mlp[2].weight.copy_(teacher_net.sigma_map.mlp[2].weight)
    trunk.t_mlp[2].bias.copy_(teacher_net.sigma_map.mlp[2].bias)

    for i in range(n_shared):
        _copy_block(trunk.shared_blocks[i], src_blocks[i])
    for j in range(cfg.latent_last_L):
        _copy_block(trunk.latent_blocks[j], src_blocks[n_shared + j])

    trunk.final_norm.weight.copy_(teacher_net.output_layer.norm_final.weight)
    trunk.token_head.weight.copy_(teacher_net.output_layer.linear.weight)
    if trunk.token_head.bias is not None:
        trunk.token_head.bias.copy_(teacher_net.output_layer.linear.bias)
    if trunk.final_ada is None:
        raise ValueError("warm start expects the student to carry a final adaLN layer")
    trunk.final_ada.map[1].weight.copy_(
        _permute_blocks(teacher_net.output_layer.adaLN_modulation.weight, _ADALN2_PERM))
    trunk.final_ada.map[1].bias.copy_(
        _permute_blocks(teacher_net.output_layer.adaLN_modulation.bias, _ADALN2_PERM))

    # The teacher has no latent axis, so every branch starts as a copy of it. A small random
    # latent embedding is what breaks the tie once training starts.
    trunk.k_emb.weight.zero_()
    if k_noise > 0:
        trunk.k_emb.weight.normal_(mean=0.0, std=k_noise)
    if verbose:
        print(f"[warm-start] copied {n_shared} shared and {cfg.latent_last_L} latent blocks "
              f"into M={cfg.latent_M} branches (k_noise={k_noise})")
