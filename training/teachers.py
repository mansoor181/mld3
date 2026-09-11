"""Frozen-teacher construction and rollout.

A teacher is either a Latent-Kernel checkpoint or a cross-family model (MDLM for text,
molecules and DNA, MaskGIT for images) routed through models/teacher_adapters.py so that every
family exposes the same absorbing x0 denoiser interface. The rollout applies the analytic absorbing reverse transition for a fixed
number of sub-steps, which is the trajectory the student is distilled against.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.latent_kernel import LatentKernelFlow, LKFConfig
from models.teacher_adapters import build_maskgit_teacher, build_mdlm_teacher

# ---------- teacher construction ----------

def build_teacher(teacher_cfg, device: torch.device, seq_len: int = 128):
    """Load a distillation teacher.

    ``teacher_cfg`` may be a plain checkpoint path or a mapping with a ``kind`` field.
    ``kind: lkf`` (the default) loads a Latent-Kernel checkpoint, which is how the LM1B cells
    carry their MDLM-equivalent x0 teacher. ``kind: mdlm`` loads a trained MDLM checkpoint and
    ``kind: maskgit`` the frozen ImageNet MaskGIT, both through the adapters in
    models/teacher_adapters.py, so every teacher exposes the same absorbing x0 denoiser trunk.
    Prefers EMA weights if present, matching MDLM practice and eval_text.
    """
    if isinstance(teacher_cfg, dict):
        kind = str(teacher_cfg.get("kind", "lkf")).lower()
    else:
        kind = "lkf"
        teacher_cfg = {"ckpt": teacher_cfg}
    if kind == "mdlm":
        # The non-text teachers were trained by the PairFlow fork on their own small
        # vocabularies without time conditioning, and the config names those deviations.
        extra = {k: teacher_cfg[k] for k in
                 ("vocab_size", "mask_id", "dit_source", "time_conditioning",
                  "hidden_size", "n_blocks", "n_heads", "cond_dim", "dropout")
                 if k in teacher_cfg}
        return build_mdlm_teacher(teacher_cfg["ckpt"], device, seq_len=seq_len, **extra)
    if kind == "maskgit":
        extra = {k: teacher_cfg[k] for k in
                 ("vocab_size", "mask_id", "codebook_size", "n_classes",
                  "hidden_dim", "depth", "heads", "mlp_dim", "img_size", "bf16",
                  "label_mode")
                 if k in teacher_cfg}
        return build_maskgit_teacher(teacher_cfg["ckpt"], device, seq_len=seq_len, **extra)
    ckpt_path = teacher_cfg["ckpt"]
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state["cfg"]
    # LKFConfig accepts the same keys the trainer wrote, but is order-sensitive; use kwargs.
    lkf_cfg = LKFConfig(**tcfg)
    teacher = LatentKernelFlow(lkf_cfg).to(device)
    teacher.load_state_dict(state["model"])
    if state.get("ema") is not None:
        shadow = state["ema"].get("shadow", {})
        # Copy EMA shadow into the model's parameters (in place). Any missing key falls
        # back to the raw weight.
        with torch.no_grad():
            for n, p in teacher.named_parameters():
                if n in shadow:
                    p.copy_(shadow[n].to(device))
        print(f"[teacher] loaded EMA weights (decay={state['ema'].get('decay')})")
    else:
        print("[teacher] no EMA in ckpt; using raw weights")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, state


# ---------- analytic teacher sub-step (MDLM-equivalent absorbing reverse) ----------

@torch.no_grad()
def teacher_analytic_step(model: LatentKernelFlow, x_t: torch.Tensor,
                          t_kappa: torch.Tensor, s_kappa: torch.Tensor,
                          arange_B: torch.Tensor) -> torch.Tensor:
    """One analytic-reverse (MDLM-equivalent) sub-step from x_t at time t_kappa to time s_kappa.

    t_kappa, s_kappa are LKF times (kappa = fraction kept; 0 = fully masked, 1 = clean).
    Uses the same math as sampling/sampler.py::sample_lkf_analytic. Only MASK positions
    move; kept positions are the identity.
    """
    cfg = model.cfg
    mask_id = cfg.mask_id
    B, L = x_t.shape
    device = x_t.device
    ones = torch.ones(B, device=device)

    # Query clean posterior head at s = 1.
    logits, router_logits = model.trunk(x_t, t_kappa, ones)         # [B,M,L,V], [B,M]
    w = F.softmax(router_logits, dim=-1)                            # [B, M]
    k_ids = torch.multinomial(w, num_samples=1).squeeze(-1)         # [B]
    gath = logits[arange_B, k_ids]                                  # [B, L, V]
    p_x0 = F.softmax(gath, dim=-1)                                  # [B, L, V]
    p_x0 = p_x0.clone()
    p_x0[..., mask_id] = 0.0
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Mask/move chance at LKF times t_kappa, s_kappa. Same schedule as sample_lkf_analytic.
    schedule = getattr(cfg, "schedule", "linear")
    if schedule == "loglinear":
        ome = 1.0 - getattr(cfg, "time_eps", 1e-3)
        mc_t = ome * (1.0 - t_kappa)
        mc_s = ome * (1.0 - s_kappa)
    else:
        mc_t = 1.0 - t_kappa
        mc_s = 1.0 - s_kappa
    # mc_t, mc_s are [B]; broadcast to [B, 1, 1] against p_x0 [B, L, V]
    delta = (mc_t - mc_s).clamp_min(0.0)[:, None, None]
    q = p_x0 * delta                                                # [B, L, V]
    q[..., mask_id] = mc_s[:, None]
    sampled = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).view(B, L)
    mask_positions = (x_t == mask_id)
    return torch.where(mask_positions, sampled, x_t)


@torch.no_grad()
def teacher_rollout(teacher: LatentKernelFlow, x_t: torch.Tensor,
                    t_kappa: torch.Tensor, s_kappa: torch.Tensor,
                    T: int) -> torch.Tensor:
    """Roll out T analytic sub-steps from (x_t, t_kappa) to (?, s_kappa)."""
    B = x_t.shape[0]
    arange_B = torch.arange(B, device=x_t.device)
    x = x_t
    for i in range(T):
        a = t_kappa + (s_kappa - t_kappa) * (i / T)
        b = t_kappa + (s_kappa - t_kappa) * ((i + 1) / T)
        x = teacher_analytic_step(teacher, x, a, b, arange_B)
    return x
