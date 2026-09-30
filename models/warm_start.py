"""Copy a trained MDLM backbone into an MLD3 student.

The molecule and DNA students get a tenth of the teacher's step budget, which only buys a
useful model if they start from the teacher. The two architectures agree block for block once
three bookkeeping differences are undone: MDLM packs its adaLN outputs as (shift, scale, gate)
where ours are (scale, shift, gate), its sinusoidal embedding is cos followed by sin where ours
is sin followed by cos, and it applies SiLU to the conditioning vector once.

The first `depth - latent_last_L` teacher blocks land in the shared stack and the rest are
broadcast into every component, so the student reproduces the teacher under any single
component until `k_noise` breaks the tie.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ADALN6 = [1, 0, 2, 4, 3, 5]
_ADALN2 = [1, 0]


def _permute(w: torch.Tensor, perm: list[int]) -> torch.Tensor:
    chunks = list(w.chunk(len(perm), dim=0))
    return torch.cat([chunks[i] for i in perm], dim=0)


@torch.no_grad()
def _copy_block(dst: nn.Module, src: nn.Module) -> None:
    dst.norm1.weight.copy_(src.norm1.weight)
    dst.norm2.weight.copy_(src.norm2.weight)
    dst.attn.qkv.weight.copy_(src.attn_qkv.weight)
    dst.attn.o.weight.copy_(src.attn_out.weight)
    dst.mlp.fc1.weight.copy_(src.mlp[0].weight)
    dst.mlp.fc1.bias.copy_(src.mlp[0].bias)
    dst.mlp.fc2.weight.copy_(src.mlp[2].weight)
    dst.mlp.fc2.bias.copy_(src.mlp[2].bias)
    dst.ada.map[1].weight.copy_(_permute(src.adaLN_modulation.weight, _ADALN6))
    dst.ada.map[1].bias.copy_(_permute(src.adaLN_modulation.bias, _ADALN6))


@torch.no_grad()
def warm_start(student, teacher_net: nn.Module, k_noise: float = 0.02) -> None:
    """Copy `teacher_net` into `student` in place. Requires arch="mdlm", cond_extra_silu=False."""
    trunk = student.trunk
    cfg = trunk.cfg
    if not trunk.mdlm or cfg.cond_extra_silu:
        raise ValueError('warm start requires arch="mdlm" and cond_extra_silu=False')

    trunk.tok.weight.copy_(teacher_net.vocab_embed.embedding)

    half = cfg.time_embed_dim // 2
    src = teacher_net.sigma_map.mlp
    trunk.t_mlp[0].weight.copy_(torch.cat([src[0].weight[:, half:],
                                           src[0].weight[:, :half]], dim=1))
    trunk.t_mlp[0].bias.copy_(src[0].bias)
    trunk.t_mlp[2].weight.copy_(src[2].weight)
    trunk.t_mlp[2].bias.copy_(src[2].bias)

    blocks = list(teacher_net.blocks)
    n_shared = cfg.depth - cfg.latent_last_L
    for i, dst in enumerate(trunk.shared_blocks):
        _copy_block(dst, blocks[i])
    for j, dst in enumerate(trunk.latent_blocks):
        _copy_block(dst, blocks[n_shared + j])

    out = teacher_net.output_layer
    trunk.final_norm.weight.copy_(out.norm_final.weight)
    trunk.head.weight.copy_(out.linear.weight)
    if trunk.head.bias is not None:
        trunk.head.bias.copy_(out.linear.bias)
    trunk.final_ada.map[1].weight.copy_(_permute(out.adaLN_modulation.weight, _ADALN2))
    trunk.final_ada.map[1].bias.copy_(_permute(out.adaLN_modulation.bias, _ADALN2))

    trunk.k_emb.weight.normal_(std=k_noise) if k_noise > 0 else trunk.k_emb.weight.zero_()
