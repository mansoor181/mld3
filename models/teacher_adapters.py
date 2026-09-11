"""Native teacher adapters for cross-family distillation.

The distiller in training/trainer_distill.py queries its teacher through a single
call, ``model.trunk(x_t, t_kappa, ones) -> (logits[B, M, L, V], router_logits[B, M])``,
and reads a handful of fields off ``model.cfg`` (mask_id, schedule, time_eps). It also
consumes a small state dict with ``cfg`` (mask_id, use_mask_prior, ...), ``meta``
(vocab_size, seq_len) and ``step``. A Latent-Kernel checkpoint provides all of this, but the
MDLM and MaskGIT teachers are stored in their own formats, so we wrap each one in a thin
adapter that presents the same trunk interface backed by that teacher's own absorbing x0
denoiser.

MDLM runs on the same GPT-2 BPE vocabulary of 50257 tokens with a single appended
mask/absorbing token at index 50257, giving a model vocabulary of 50258. That matches the
Latent-Kernel student's extended vocabulary when use_mask_prior is set, so no index
remapping is needed. It uses the same log-linear schedule, and we convert the Latent-Kernel
time kappa (the fraction of tokens kept, 1 = clean, 0 = fully masked) to the baseline noise
level with sigma = -log(1 - (1 - eps) * (1 - kappa)), which is exactly the level at which its
mask probability equals the masked fraction the distiller assumes at kappa.

MDLM's DiT returns raw logits over the 50258-way vocabulary, and a softmax over the last
dimension is the x0 posterior the distiller wants (it zeroes the mask column itself).
"""
from __future__ import annotations
import importlib.util
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from paths import MDLM_DIR, PAIRFLOW_DIR, MASKGIT_DIR
from utils.vendor_import import load_file_module

# Baseline source trees. The MDLM and PairFlow DiT classes are loaded by file path inside
# their factories, since the two ship a module of the same name.
_MDLM_DIR = MDLM_DIR
_PAIRFLOW_DIR = PAIRFLOW_DIR
# The MaskGIT transformer comes from the Di4C authors' own vendored fork rather than from the
# ReDi tree, because the Di4C image experiment we compare against was run against this copy.
_MASKGIT_DIR = MASKGIT_DIR

# GPT-2 BPE base vocabulary and the single appended mask/absorbing token.
_BASE_VOCAB = 50257
_MASK_ID = 50257
_MODEL_VOCAB = 50258


@dataclass
class AdapterCfg:
    """The subset of LKFConfig fields the distiller reads off a teacher.

    The distiller's teacher_analytic_step reads mask_id, schedule and time_eps, and the
    setup code reads mask_id, use_mask_prior, latent_M, objective, trunk_arch, dim, depth
    and heads for its sanity guards and student defaults. We fill all of them.
    """
    mask_id: int = _MASK_ID
    use_mask_prior: bool = True
    latent_M: int = 1
    objective: str = "x0"
    trunk_arch: str = "mdlm_dit"
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_mult: int = 4
    latent_last_L: int = 4
    time_embed_dim: int = 128
    tie_embeddings: bool = False
    cond_dim: int = 128
    schedule: str = "loglinear"
    time_eps: float = 1e-3
    # Fields the eval harness reads directly off cfg for a teacher run.
    seq_len: int = 128
    vocab_size: int = _MODEL_VOCAB
    pad_id: int = -1

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def _kappa_to_sigma(t_kappa: torch.Tensor, eps: float) -> torch.Tensor:
    """Convert a Latent-Kernel kept-fraction time to the baseline log-linear sigma.

    The distiller assumes a masked fraction of (1 - eps) * (1 - kappa) at time kappa under
    the log-linear schedule. Both baselines define sigma so that their mask probability is
    1 - exp(-sigma). Setting the two equal gives sigma = -log(1 - (1 - eps) * (1 - kappa)).
    """
    kept = 1.0 - (1.0 - eps) * (1.0 - t_kappa)
    return -torch.log(kept.clamp_min(1e-8))


class MDLMTeacherAdapter(nn.Module):
    """Wraps a trained MDLM DiT backbone as an absorbing x0 denoiser trunk."""

    def __init__(self, net: nn.Module, cfg: AdapterCfg, time_conditioning: bool = True):
        super().__init__()
        self.net = net
        self.cfg = cfg
        # MDLM zeroes sigma inside its own `_process_sigma` when the backbone was trained
        # without time conditioning. We call the backbone directly and so must reproduce
        # that here, otherwise a teacher trained at sigma = 0 is queried off-distribution.
        self.time_conditioning = bool(time_conditioning)

    @torch.no_grad()
    def trunk(self, x_t: torch.Tensor, t_kappa: torch.Tensor, s: torch.Tensor):
        """Return (logits[B, 1, L, V], router_logits[B, 1]).

        MDLM's DiT already returns logits whose softmax over the vocabulary is the x0
        posterior, so we forward once and add the singleton mixture axis.
        """
        B = x_t.shape[0]
        sigma = _kappa_to_sigma(t_kappa, self.cfg.time_eps).reshape(B)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        logits = self.net(x_t, sigma)                     # [B, L, V]
        logits = logits.float().unsqueeze(1)              # [B, 1, L, V]
        router = torch.zeros(B, 1, device=x_t.device, dtype=logits.dtype)
        return logits, router




class MaskGITTeacherAdapter(nn.Module):
    """Wraps the pretrained MaskGIT ImageNet-256 transformer as an absorbing x0 denoiser trunk.

    MaskGIT is already exactly the object the distiller wants, a bidirectional encoder that maps
    a partially masked token grid to a categorical posterior over the clean code at every
    position. It carries no notion of time, so `t_kappa` and `s` are accepted and ignored in the
    same way the PairFlow molecule and DNA teachers ignore them, and the mask rate reaches the
    network only through how much of `x_t` is actually masked.

    The two shape adjustments are that the distiller works in a flat length-256 sequence while
    MaskGIT wants the 16x16 grid, and that the distiller expects a mixture axis which a single
    deterministic teacher fills with one component.
    """

    def __init__(self, net: nn.Module, cfg: AdapterCfg, bf16: bool = False,
                 label_mode: str = "null", n_classes: int = 1000):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.bf16 = bf16
        self.n_classes = n_classes
        if label_mode not in ("null", "random", "held"):
            raise ValueError(f"label_mode must be null, random or held, got {label_mode}")
        self.label_mode = label_mode
        self._held = None
        self.grid = math.isqrt(cfg.seq_len)
        if self.grid * self.grid != cfg.seq_len:
            raise ValueError(f"seq_len {cfg.seq_len} is not a square, so it is not a token grid")

    def resample_labels(self, batch_size: int, device) -> None:
        """Draw one class per row and keep it for the whole rollout.

        Under `held` the teacher is the class marginal E_y p(x | y), which is a legitimate
        unconditional model. That only holds if a trajectory conditions on one class throughout.
        Redrawing at every reverse step instead conditions step one on a goldfish and step two on
        an airliner, and we measured that this costs 21 FID at 8 steps against the null label.
        """
        self._held = torch.randint(self.n_classes, (batch_size,), device=device)

    @torch.no_grad()
    def trunk(self, x_t: torch.Tensor, t_kappa: torch.Tensor, s: torch.Tensor):
        """Return (logits[B, 1, L, V], router_logits[B, 1]).

        MaskGIT concatenates a class token at position 256 internally and slices it off before
        returning, hence the output is already [B, 256, 1025] over the visual vocabulary with
        the mask column at 1024, which is precisely our model vocabulary and needs no surgery.

        The `y` argument is not optional in the vendored forward, so we pass zeros and set
        `drop_label`, which overwrites every entry with the null class at 2025. We measured the
        cost of discarding the label at between 0.036 and 0.090 nats across masking rates from
        ten to ninety percent, so the unconditional teacher is only marginally weaker than the
        conditional one on likelihood.
        """
        B = x_t.shape[0]
        if self.label_mode == "held":
            if self._held is None or self._held.shape[0] != B:
                self.resample_labels(B, x_t.device)
            y = self._held
            drop = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        elif self.label_mode == "random":
            # Drawing the label uniformly at every call makes the teacher the class marginal
            # E_y p(x | y), which is itself a legitimate unconditional model of ImageNet and is
            # the target the unconditional student is asked to match. The alternative of a
            # single null label asks MaskGIT to work in a regime it barely saw during training.
            y = torch.randint(self.n_classes, (B,), device=x_t.device)
            drop = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        else:
            y = torch.zeros(B, dtype=torch.long, device=x_t.device)
            drop = torch.ones(B, dtype=torch.bool, device=x_t.device)
        grid = x_t.view(B, self.grid, self.grid)
        if self.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.net(grid, y=y, drop_label=drop)
        else:
            logits = self.net(grid, y=y, drop_label=drop)
        logits = logits.float().unsqueeze(1)               # [B, 1, L, V]
        router = torch.zeros(B, 1, device=x_t.device, dtype=logits.dtype)
        return logits, router


def _attn_forward_no_weights(self, x, need_weights=False):
    """Attention.forward without the averaged attention map.

    The vendored version at `Network/transformer.py:73` calls MultiheadAttention with the
    default `need_weights=True`, which forfeits the fused kernel and materialises a
    [B, 257, 257] map on each of the 24 layers. Nothing in our path reads those maps, since
    `return_attn` stays False, and dropping them is a pure speed change with identical numerics.

    We have since threaded the same flag through the vendored file itself, so that the released
    Di4C trainer also takes the fused kernel, and `TransformerEncoder.forward` now passes
    `need_weights` down. This shim accepts and forwards it so that the two patches compose, and
    it keeps defaulting to False for any caller that predates the change.
    """
    return self.mha(x, x, x, need_weights=need_weights)


def build_maskgit_teacher(ckpt_path: str, device: torch.device, seq_len: int = 256,
                          vocab_size: int = 1025, mask_id: int = 1024,
                          codebook_size: int = 1024, n_classes: int = 1000,
                          hidden_dim: int = 768, depth: int = 24, heads: int = 16,
                          mlp_dim: int = 3072, img_size: int = 256, bf16: bool = False,
                          label_mode: str = "null"):
    """Load the pretrained MaskGIT ImageNet-256 checkpoint as a teacher trunk.

    The head count defaults to 16 because that is what the released weights were trained with,
    and getting it wrong is a silent failure rather than a loud one. `nn.MultiheadAttention`
    stores `in_proj_weight` as [3d, d] whatever the head count, so 8 and 12 both load with zero
    missing and zero unexpected keys and then compute nonsense. We measured 5.003 nats per
    masked token at 16 heads against 5.718 at 12 and 6.071 at 8, with chance at 6.931, using
    `scripts/image_gates.py --gate 3`. Loading is strict here so that a checkpoint which does
    not fit these hyperparameters stops the run instead of quietly degrading it.

    Returns (adapter, tstate) matching build_teacher's contract.
    """
    # Loaded by file path rather than through sys.path, because the ReDi image tree ships a
    # second top-level `Network` package whose MaskTransformer takes a required `randmask`
    # argument. Whichever tree imports first would otherwise win for the whole process, and a
    # run that also builds the ReDi VQGAN for FID would silently get the wrong class here.
    _mg = load_file_module("_di4c_maskgit_transformer",
                           os.path.join(_MASKGIT_DIR, "Network", "transformer.py"))
    Attention, MaskTransformer = _mg.Attention, _mg.MaskTransformer

    net = MaskTransformer(img_size=img_size, hidden_dim=hidden_dim, codebook_size=codebook_size,
                          depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=0.0,
                          nclass=n_classes)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    n_patched = 0
    for m in net.modules():
        if isinstance(m, Attention):
            m.forward = _attn_forward_no_weights.__get__(m, Attention)
            n_patched += 1
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if mask_id != codebook_size:
        raise ValueError(f"MaskGIT reserves its visual mask at {codebook_size}, not {mask_id}")
    if vocab_size != codebook_size + 1:
        raise ValueError(f"model vocabulary must be {codebook_size + 1}, got {vocab_size}")
    # MaskGIT has no schedule and no time input, hence the linear schedule with a zero epsilon
    # is the honest reading of its time axis rather than a description of one it possesses. Both
    # teacher_analytic_step and sample_lkf_analytic read these two fields to turn a kappa into a
    # mask rate, and under `linear` the mask rate is exactly 1 - kappa.
    #
    # The trunk_arch field is not a description of MaskGIT, which is a vanilla bidirectional
    # encoder resembling neither of our two trunks. It is the fallback the student picks up at
    # trainer_distill.py:840 when a config omits its own, and the image student must be `lkf`.
    # MaskGIT carries learned absolute positions and a per-position output bias, so no warm
    # start from it is faithful and the only reason to prefer `mdlm_dit` does not arise here.
    cfg = AdapterCfg(trunk_arch="lkf", seq_len=seq_len, mask_id=mask_id, vocab_size=vocab_size,
                     dim=hidden_dim, depth=depth, heads=heads, schedule="linear", time_eps=0.0,
                     latent_M=1, objective="x0", use_mask_prior=True)
    adapter = MaskGITTeacherAdapter(net, cfg, bf16=bf16, label_mode=label_mode,
                                    n_classes=n_classes).to(device)
    tstate = {"cfg": cfg, "meta": {"vocab_size": vocab_size, "seq_len": seq_len}, "step": -1}
    print(f"[teacher-adapter] MaskGIT teacher ready (vocab={vocab_size}, mask_id={mask_id}, "
          f"heads={heads}, depth={depth}, {n_patched} attention blocks unfused-weights-off)")
    return adapter, tstate


def _apply_ema_list(net: nn.Module, shadow_params, device):
    """Copy an EMA shadow-parameter list (in parameter order) into the network."""
    params = [p for p in net.parameters()]
    if len(shadow_params) != len(params):
        print(f"[teacher-adapter] EMA length {len(shadow_params)} != params {len(params)}; "
              f"skipping EMA, using raw weights")
        return
    with torch.no_grad():
        for p, s in zip(params, shadow_params):
            p.copy_(s.to(device))
    print("[teacher-adapter] applied EMA shadow weights")


def _load_dit_class(source: str):
    """Import a baseline's DiT class from its own tree by file path.

    Both baselines keep their DiT in a `models` package that collides with this project's
    `models` package, so we load the module file directly. `dit.py` uses only absolute
    imports in either tree, which makes a standalone file load safe. The two
    implementations share a constructor and a forward signature but differ inside the
    block, so a checkpoint must be loaded into the class that produced it.
    """
    dirs = {"mdlm": _MDLM_DIR, "pairflow": _PAIRFLOW_DIR}
    if source not in dirs:
        raise ValueError(f"unknown DiT source {source!r}; known={list(dirs)}")
    path = str(Path(dirs[source]) / "models" / "dit.py")
    spec = importlib.util.spec_from_file_location(f"{source}_dit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DIT


def _load_lightning_ckpt(ckpt_path: str):
    """Unpickle a Lightning checkpoint, tolerating objects whose classes we cannot import.

    The PairFlow runs pickle their trust_remote_code tokenizer into `hyper_parameters`, and
    that tokenizer lives in a `transformers_modules` package that only exists inside the
    process that downloaded it. We never touch those hyper-parameters, so we substitute an
    inert placeholder for any class that fails to resolve and keep the tensors.
    """

    class _Placeholder:
        def __init__(self, *a, **k): pass
        def __setstate__(self, state): pass
        def __call__(self, *a, **k): return self

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Placeholder

    class _PickleModule:
        Unpickler = _Unpickler
        UnpicklingError = pickle.UnpicklingError

        @staticmethod
        def load(f, **kw): return _Unpickler(f, **kw).load()

    return torch.load(ckpt_path, map_location="cpu", weights_only=False,
                      pickle_module=_PickleModule)


def build_mdlm_teacher(ckpt_path: str, device: torch.device, seq_len: int = 128,
                       vocab_size: int | None = None, mask_id: int | None = None,
                       dit_source: str = "mdlm", time_conditioning: bool = True,
                       hidden_size: int = 768, n_blocks: int = 12, n_heads: int = 12,
                       cond_dim: int = 128, dropout: float = 0.1):
    """Load an MDLM Lightning checkpoint into a DiT backbone wrapped as a teacher trunk.

    The text teachers run on the GPT-2 vocabulary with an appended mask, which the defaults
    reproduce. The molecule and DNA teachers run on their own small vocabularies with the
    mask already inside, and they were trained by the PairFlow fork without time
    conditioning, so those runs pass `vocab_size`, `mask_id`, `dit_source="pairflow"` and
    `time_conditioning=False`.

    Returns (adapter, tstate) matching build_teacher's contract.
    """
    DIT = _load_dit_class(dit_source)

    # A caller that names both the vocabulary and the mask is describing a corpus whose mask
    # already sits inside the vocabulary, which is how the molecule and the DNA runs are
    # configured, and the loader reports that same count. A caller that names neither is on
    # the text default, where the mask was appended one past the end of the GPT-2 vocabulary
    # and the loader reports one fewer symbol than the model carries. We cannot read this off
    # the mask index, because an in-vocabulary mask is free to occupy the last slot, which is
    # exactly what DeepSTARR does with five nucleotide symbols and a mask at index five.
    mask_in_vocab = vocab_size is not None and mask_id is not None
    model_vocab = int(vocab_size) if vocab_size is not None else _MODEL_VOCAB
    if mask_id is None:
        mask_id = _MASK_ID if vocab_size is None else model_vocab - 1
    mask_id = int(mask_id)
    if not 0 <= mask_id < model_vocab:
        raise ValueError(f"mask_id {mask_id} outside model vocabulary {model_vocab}")
    data_vocab = model_vocab if mask_in_vocab else model_vocab - 1

    ck = _load_lightning_ckpt(ckpt_path)
    model_cfg = OmegaConf.create({
        "model": {"hidden_size": hidden_size, "cond_dim": cond_dim, "length": seq_len,
                  "n_blocks": n_blocks, "n_heads": n_heads, "scale_by_sigma": True,
                  "dropout": dropout},
        # PairFlow's DiT reads algo.causal_attention where MDLM's ignores the key entirely.
        "algo": {"causal_attention": False},
    })
    net = DIT(model_cfg, vocab_size=model_vocab).to(device)
    sd = ck["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = net.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[teacher-adapter] MDLM missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[teacher-adapter] MDLM unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    ema = ck.get("ema")
    if isinstance(ema, dict) and "shadow_params" in ema:
        _apply_ema_list(net, ema["shadow_params"], device)
    else:
        print("[teacher-adapter] MDLM no EMA shadow in ckpt; using raw weights")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cfg = AdapterCfg(trunk_arch="mdlm_dit", seq_len=seq_len, mask_id=mask_id,
                     vocab_size=model_vocab, dim=hidden_size, depth=n_blocks,
                     heads=n_heads, cond_dim=cond_dim)
    adapter = MDLMTeacherAdapter(net, cfg, time_conditioning=time_conditioning).to(device)
    step = int(ck.get("global_step", -1))
    tstate = {"cfg": cfg, "meta": {"vocab_size": data_vocab, "seq_len": seq_len}, "step": step}
    print(f"[teacher-adapter] MDLM teacher ready (step={step}, vocab={model_vocab}, "
          f"mask_id={mask_id}, dit={dit_source}, time_conditioning={time_conditioning})")
    return adapter, tstate


