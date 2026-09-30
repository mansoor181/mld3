"""Frozen teachers and the analytic rollout the student is distilled against.

A teacher exposes one call, ``trunk(x_t, t, s) -> (logits[B, 1, L, V], router[B, 1])``, whose
softmax over the vocabulary is the clean-token posterior. MDLM (text, molecules, DNA) and
MaskGIT (images) are stored in their own formats, so each is wrapped in a thin adapter. Our
time is kappa, the fraction of positions kept: 0 is fully masked and 1 is clean.
"""
from __future__ import annotations

import importlib.util
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from paths import MASKGIT_DIR, MDLM_DIR, PAIRFLOW_DIR


@dataclass
class TeacherConfig:
    seq_len: int
    vocab_size: int
    mask_id: int
    schedule: str = "loglinear"
    time_eps: float = 1e-3
    latent_M: int = 1


def _load_module(name: str, path: str):
    """Import a baseline module by file path.

    Both baseline trees ship a top-level ``models`` package that collides with ours, and the
    ReDi tree ships a second ``Network`` package, so whichever imported first would otherwise
    win for the whole process.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kappa_to_sigma(t: torch.Tensor, eps: float) -> torch.Tensor:
    """The baselines' log-linear noise level at which the mask rate equals (1 - eps)(1 - t)."""
    return -torch.log((1.0 - (1.0 - eps) * (1.0 - t)).clamp_min(1e-8))


class MDLMTeacher(nn.Module):
    def __init__(self, net: nn.Module, cfg: TeacherConfig, time_conditioning: bool = True):
        super().__init__()
        self.net, self.cfg = net, cfg
        self.time_conditioning = time_conditioning

    @torch.no_grad()
    def trunk(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        sigma = _kappa_to_sigma(t, self.cfg.time_eps).reshape(x_t.shape[0])
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        logits = self.net(x_t, sigma).float().unsqueeze(1)
        return logits, torch.zeros(x_t.shape[0], 1, device=x_t.device)


class MaskGITTeacher(nn.Module):
    """MaskGIT over a 16x16 token grid. It has no time input, so t and s are ignored."""

    def __init__(self, net: nn.Module, cfg: TeacherConfig, bf16: bool = False,
                 n_classes: int = 1000):
        super().__init__()
        self.net, self.cfg, self.bf16 = net, cfg, bf16
        self.n_classes = n_classes
        self.grid = math.isqrt(cfg.seq_len)

    @torch.no_grad()
    def trunk(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        B = x_t.shape[0]
        y = torch.zeros(B, dtype=torch.long, device=x_t.device)
        drop = torch.ones(B, dtype=torch.bool, device=x_t.device)
        grid = x_t.view(B, self.grid, self.grid)
        if self.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.net(grid, y=y, drop_label=drop)
        else:
            logits = self.net(grid, y=y, drop_label=drop)
        return logits.float().unsqueeze(1), torch.zeros(B, 1, device=x_t.device)


def _load_lightning_ckpt(path: str):
    """Unpickle a Lightning checkpoint, tolerating classes we cannot import.

    PairFlow pickles a trust_remote_code tokenizer into its hyper-parameters, and that class
    only exists inside the process that downloaded it. We read only tensors.
    """
    class Placeholder:
        def __init__(self, *a, **k): pass
        def __setstate__(self, state): pass
        def __call__(self, *a, **k): return self

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return Placeholder

    class Module:
        Unpickler = Unpickler
        UnpicklingError = pickle.UnpicklingError

        @staticmethod
        def load(f, **kw):
            return Unpickler(f, **kw).load()

    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=Module)


def build_mdlm_teacher(ckpt: str, device, seq_len: int, vocab_size: int, mask_id: int,
                       dit_source: str = "mdlm", time_conditioning: bool = True,
                       hidden_size: int = 768, n_blocks: int = 12, n_heads: int = 12,
                       cond_dim: int = 128, dropout: float = 0.1, **_):
    """Load an MDLM Lightning checkpoint into its own DiT class.

    The two baseline trees share a constructor and a forward signature but differ inside the
    block, so a checkpoint must go back into the class that produced it.
    """
    root = {"mdlm": MDLM_DIR, "pairflow": PAIRFLOW_DIR}[dit_source]
    DIT = _load_module(f"{dit_source}_dit", str(Path(root) / "models" / "dit.py")).DIT
    net = DIT(OmegaConf.create({
        "model": {"hidden_size": hidden_size, "cond_dim": cond_dim, "length": seq_len,
                  "n_blocks": n_blocks, "n_heads": n_heads, "scale_by_sigma": True,
                  "dropout": dropout},
        "algo": {"causal_attention": False},
    }), vocab_size=vocab_size).to(device)

    ck = _load_lightning_ckpt(ckpt)
    prefix = "backbone."
    net.load_state_dict({k[len(prefix):]: v for k, v in ck["state_dict"].items()
                         if k.startswith(prefix)}, strict=False)
    ema = ck.get("ema")
    if isinstance(ema, dict) and "shadow_params" in ema:
        with torch.no_grad():
            for p, shadow in zip(net.parameters(), ema["shadow_params"]):
                p.copy_(shadow.to(device))
    net.eval().requires_grad_(False)
    cfg = TeacherConfig(seq_len=seq_len, vocab_size=vocab_size, mask_id=mask_id)
    return MDLMTeacher(net, cfg, time_conditioning).to(device)


def build_maskgit_teacher(ckpt: str, device, seq_len: int = 256, vocab_size: int = 1025,
                          mask_id: int = 1024, codebook_size: int = 1024, n_classes: int = 1000,
                          hidden_dim: int = 768, depth: int = 24, heads: int = 16,
                          mlp_dim: int = 3072, img_size: int = 256, bf16: bool = False, **_):
    """Load the pretrained MaskGIT ImageNet-256 checkpoint.

    The head count is not recoverable from the weights, since nn.MultiheadAttention stores
    in_proj_weight as [3d, d] whatever it is, so a wrong value loads cleanly and then computes
    nonsense. The released weights use 16.
    """
    mg = _load_module("maskgit_transformer", str(Path(MASKGIT_DIR) / "Network" / "transformer.py"))
    net = mg.MaskTransformer(img_size=img_size, hidden_dim=hidden_dim,
                             codebook_size=codebook_size, depth=depth, heads=heads,
                             mlp_dim=mlp_dim, dropout=0.0, nclass=n_classes)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    # The vendored Attention asks MultiheadAttention for the averaged attention map, which
    # forfeits the fused kernel to build a [B, 257, 257] tensor per layer that nothing reads.
    for m in net.modules():
        if isinstance(m, mg.Attention):
            m.forward = (lambda self, x, need_weights=False:
                         self.mha(x, x, x, need_weights=need_weights)).__get__(m, mg.Attention)
    net.to(device).eval().requires_grad_(False)
    # MaskGIT has no schedule, so the linear one with a zero epsilon is a definition rather
    # than a description. The student must share it, since both sides turn a kappa into a mask
    # rate through these two fields.
    cfg = TeacherConfig(seq_len=seq_len, vocab_size=vocab_size, mask_id=mask_id,
                        schedule="linear", time_eps=0.0)
    return MaskGITTeacher(net, cfg, bf16=bf16, n_classes=n_classes).to(device)


def build_teacher(spec: dict, device, seq_len: int):
    builder = {"mdlm": build_mdlm_teacher, "maskgit": build_maskgit_teacher}[spec["kind"]]
    return builder(device=device, seq_len=seq_len,
                   **{k: v for k, v in spec.items() if k != "kind"})


def mask_rate(cfg, t) -> torch.Tensor:
    """Probability that a position is still masked at time t."""
    return (1.0 - cfg.time_eps) * (1.0 - t) if cfg.schedule == "loglinear" else 1.0 - t


@torch.no_grad()
def analytic_step(teacher, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """One analytic reverse sub-step: q(v) ~ p_0(v)(m_t - m_s) and q(MASK) ~ m_s."""
    cfg = teacher.cfg
    logits, _ = teacher.trunk(x_t, t, torch.ones_like(t))
    p0 = F.softmax(logits[:, 0], dim=-1)
    p0[..., cfg.mask_id] = 0.0
    p0 = p0 / p0.sum(-1, keepdim=True).clamp_min(1e-8)

    m_t, m_s = mask_rate(cfg, t), mask_rate(cfg, s)
    q = p0 * (m_t - m_s).clamp_min(0.0)[:, None, None]
    q[..., cfg.mask_id] = m_s[:, None]
    drawn = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).view(x_t.shape)
    return torch.where(x_t == cfg.mask_id, drawn, x_t)


@torch.no_grad()
def rollout(teacher, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, T: int) -> torch.Tensor:
    """Advance the teacher from t to s in T analytic sub-steps."""
    x = x_t
    for i in range(T):
        x = analytic_step(teacher, x, t + (s - t) * (i / T), t + (s - t) * ((i + 1) / T))
    return x
