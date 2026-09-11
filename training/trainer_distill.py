"""Step-KL trajectory-matching distillation for the Latent-Kernel Flow.

Usage:
    python -m training.trainer_distill --config configs/distill/lm1b_M4_K4.yaml

Design
------
Teacher. A previously trained LKF checkpoint (x0-mode, MDLM-equivalent). At each
training step the teacher rolls out `T = distill.substeps` analytic sub-steps from
x_t (at time t) to x_s (at time s), using `sample_lkf_analytic` semantics
(query trunk at s=1 to get the clean posterior; then apply the analytic absorbing
reverse transition per sub-step). Any teacher M works.

Student. A fresh LKF in two_time-mode. Its `mixture_logprob(x_t, x_s, t, s)`
scores the teacher-rolled x_s as a JOINT distribution over the L positions
(logsumexp over M mixture components), so a mixture student (M>1) can represent
the correlation that the teacher's T sub-steps induce, whereas a factorized
student (M=1) cannot. Loss = -log p_S(x_s_teacher | x_t), averaged over batch.

Schedule. K = distill.K student boundaries, uniform in kappa = t. For each
batch example, an index j in {0, ..., K-1} is drawn; t = j/K, s = (j+1)/K.

Everything else (data loading, optimizer, EMA, checkpoint atomic write, resume,
W&B) mirrors training/trainer.py so downstream aggregation is unchanged.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from data import infinite
from data import text as textdata
from data import mols as moldata
from data import dna as dnadata
from data import images as imgdata
from models.latent_kernel import LatentKernelFlow, LKFConfig
from utils.wandb_utils import init as wb_init, log as wb_log
from utils.wandb_utils import summary as wb_summary
from utils.wandb_utils import finish as wb_finish
from training.common import EMA, seed_all, get_device, load_config, maybe_extend_vocab_for_mask
from training.teachers import build_teacher
from training.losses import SUPERVISION_DEFAULTS, read_supervision_cfg, distill_step_loss
from training.di4c import di4c_step_loss
from models.warm_start import warm_start_from_mdlm


def make_loader(name, split, batch_size, shuffle=True, num_workers=2):
    if textdata.is_text_dataset(name):
        return textdata.make_loader(name, split, batch_size, shuffle=shuffle,
                                    num_workers=num_workers)
    if moldata.is_mol_dataset(name):
        return moldata.make_loader(name, split, batch_size, shuffle=shuffle,
                                   num_workers=num_workers)
    if dnadata.is_dna_dataset(name):
        return dnadata.make_loader(name, split, batch_size, shuffle=shuffle,
                                   num_workers=num_workers)
    if imgdata.is_image_dataset(name):
        return imgdata.make_loader(name, split, batch_size, shuffle=shuffle,
                                   num_workers=num_workers)
    raise ValueError(f"unknown dataset {name!r}: not a text, molecule, DNA, or image corpus")


# ---------- main ----------

def train(cfg: dict) -> None:
    seed_all(cfg.get("seed", 0))
    device = get_device()
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["dataset"]["name"]
    eff_bs = cfg["batch_size"]
    micro_bs = cfg.get("micro_batch_size", eff_bs)
    accum = max(1, eff_bs // micro_bs)
    if micro_bs * accum != eff_bs:
        print(f"[warn] batch_size {eff_bs} not divisible by micro_batch_size {micro_bs}; "
              f"using effective {micro_bs * accum} (micro {micro_bs} x accum {accum})")
    print(f"batch: effective={micro_bs * accum} (micro={micro_bs} x accum={accum})")
    loader, meta = make_loader(dataset_name, "train", micro_bs,
                               shuffle=True, num_workers=cfg.get("num_workers", 2))
    val_loader, _ = make_loader(dataset_name, "val", micro_bs,
                                shuffle=False, num_workers=cfg.get("num_workers", 2))

    vocab_size_data = meta["vocab_size"]
    seq_len = meta["seq_len"]
    use_mask_prior = cfg["model"].get("use_mask_prior", True)
    # Molecule and DNA corpora reserve a MASK inside their own vocabulary and surface it
    # through the loader meta. The config may override it, and text corpora supply neither,
    # which leaves the historical append-past-the-end behaviour untouched.
    explicit_mask_id = cfg["model"].get("mask_id", meta.get("mask_id"))
    vocab_ext, mask_id = maybe_extend_vocab_for_mask(
        vocab_size_data, use_mask_prior, explicit_mask_id)
    if explicit_mask_id is not None:
        print(f"[data] domain={meta.get('domain', 'text')} vocab={vocab_ext} "
              f"mask_id={mask_id} (in-vocabulary)")

    # Teacher first, so we can pin student trunk hyperparams to it where the config omits them.
    teacher_cfg = cfg["teacher"]
    print(f"[teacher] loading {teacher_cfg}")
    teacher, tstate = build_teacher(teacher_cfg, device, seq_len=seq_len)
    tcfg = tstate["cfg"]
    tmeta = tstate.get("meta", {})
    # Sanity guards.
    if tmeta.get("vocab_size") != vocab_size_data:
        raise SystemExit(f"vocab mismatch: teacher {tmeta.get('vocab_size')} vs data {vocab_size_data}")
    if tmeta.get("seq_len") != seq_len:
        raise SystemExit(f"seq_len mismatch: teacher {tmeta.get('seq_len')} vs data {seq_len}")
    if tcfg.get("mask_id") != (mask_id if mask_id >= 0 else 0):
        raise SystemExit(f"mask_id mismatch: teacher {tcfg.get('mask_id')} vs data {mask_id}")
    if tcfg.get("use_mask_prior", True) != use_mask_prior:
        raise SystemExit(f"use_mask_prior mismatch: teacher {tcfg.get('use_mask_prior')} vs cfg {use_mask_prior}")
    teacher_step = int(tstate.get("step", -1))
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher] step={teacher_step}  M={tcfg.get('latent_M')}  "
          f"objective={tcfg.get('objective')}  trunk={tcfg.get('trunk_arch')}  "
          f"params={n_teacher/1e6:.2f}M")

    # Student: two_time-mode; inherit trunk arch/schedule/time_eps from the teacher unless overridden.
    smcfg = cfg["model"]
    student_cfg = LKFConfig(
        vocab_size=vocab_ext,
        seq_len=seq_len,
        dim=smcfg.get("dim", tcfg["dim"]),
        depth=smcfg.get("depth", tcfg["depth"]),
        heads=smcfg.get("heads", tcfg["heads"]),
        mlp_mult=smcfg.get("mlp_mult", tcfg.get("mlp_mult", 4)),
        latent_M=smcfg["latent_M"],
        latent_last_L=smcfg.get("latent_last_L", tcfg.get("latent_last_L", 4)),
        time_embed_dim=smcfg.get("time_embed_dim", tcfg.get("time_embed_dim", 128)),
        tie_embeddings=smcfg.get("tie_embeddings", tcfg.get("tie_embeddings", False)),
        mask_id=mask_id if mask_id >= 0 else 0,
        use_mask_prior=use_mask_prior,
        objective="two_time",   # student always models p(x_s|x_t) directly
        pad_id=-1,              # distillation loss does not need pad exclusion (uses two_time NLL)
        antithetic_t=False,
        dropout=smcfg.get("dropout", tcfg.get("dropout", 0.0)),
        cond_dim=smcfg.get("cond_dim", tcfg.get("cond_dim", 0)),
        trunk_arch=smcfg.get("trunk_arch", tcfg.get("trunk_arch", "lkf")),
        time_eps=smcfg.get("time_eps", tcfg.get("time_eps", 1e-3)),
        schedule=smcfg.get("schedule", tcfg.get("schedule", "loglinear")),
        dt_min=smcfg.get("dt_min", 0.02),
        dt_max=smcfg.get("dt_max", 1.0),
        router_entropy_coef=smcfg.get("router_entropy_coef", 0.05),
        router_load_balance_coef=smcfg.get("router_load_balance_coef", -0.05),
        semigroup_coef=0.0,     # distillation supersedes the model's own semigroup consistency
        semigroup_dt_min=0.05,
        cond_extra_silu=smcfg.get("cond_extra_silu", tcfg.get("cond_extra_silu", True)),
        time_conditioning=smcfg.get("time_conditioning", tcfg.get("time_conditioning", True)),
    )
    student = LatentKernelFlow(student_cfg).to(device)
    # Warm start. The molecule and DNA students run for a tenth of the teacher's budget, which
    # is only worth anything if they begin as copies of the teacher rather than at random.
    if smcfg.get("warm_start", False):
        if not hasattr(teacher, "net"):
            raise SystemExit("warm_start requires an MDLM-adapter teacher (kind: mdlm)")
        warm_start_from_mdlm(student, teacher.net,
                             k_noise=float(smcfg.get("warm_start_k_noise", 0.02)))
    n_student = sum(p.numel() for p in student.parameters())
    print(f"[student] M={student_cfg.latent_M} params={n_student/1e6:.2f}M "
          f"dim={student_cfg.dim} depth={student_cfg.depth} heads={student_cfg.heads} "
          f"trunk={student_cfg.trunk_arch}")

    # Distill knobs.
    dcfg = cfg.get("distill", {})
    K = int(dcfg.get("K", 4))
    T_per_step = int(dcfg.get("substeps", 8))
    objective_variant = str(dcfg.get("objective_variant", "exact"))
    if objective_variant not in ("exact", "mc_sample", "di4c"):
        raise SystemExit(f"unknown distill.objective_variant: {objective_variant}")
    print(f"[distill] K={K} substeps={T_per_step} objective_variant={objective_variant} "
          f"(teacher takes {K * T_per_step} sub-steps end-to-end)")
    sup = read_supervision_cfg(dcfg)
    if sup != SUPERVISION_DEFAULTS:
        print(f"[distill] sup {sup}")

    # Di4C baseline knobs. The reference defaults live in the authors' harness at
    # baselines/di4c/sdtt/src/sdtt/configs/config.yaml (latent_bsize 16, alpha_t sigmoid,
    # alpha_const 0.1, distil_delta 0.02, T 1024), and their lm1b runs overrode
    # latent_bsize to 8 and distil_delta to 0.01.
    _dd = dcfg.get("di4c", {}) or {}
    di4c_cfg = {
        "latent_bsize": int(_dd.get("latent_bsize", 4)),
        "alpha_t": str(_dd.get("alpha_t", "sigmoid")),
        "alpha_const": float(_dd.get("alpha_const", 0.1)),
        "distil_delta": float(_dd.get("distil_delta", 0.01)),
        "T": int(_dd.get("T", 1024)),
        "use_cv": bool(_dd.get("use_cv", True)),
        "latent_prior": str(_dd.get("latent_prior", "uniform")),
        "train_nll_exact": bool(_dd.get("train_nll_exact", True)),
    }
    if objective_variant == "di4c":
        if di4c_cfg["latent_prior"] not in ("uniform", "router"):
            raise SystemExit(f"unknown distill.di4c.latent_prior: {di4c_cfg['latent_prior']}")
        print(f"[distill] di4c {di4c_cfg}")

    def step_loss(model_batch):
        if objective_variant == "di4c":
            return di4c_step_loss(student, teacher, model_batch, di4c_cfg)
        return distill_step_loss(student, teacher, model_batch, K, T_per_step,
                                 objective_variant, sup)

    run_name = f"{dataset_name}_{Path(cfg['out_dir']).name}"
    baseline_tag = Path(cfg['out_dir']).name.split('_seed')[0]
    domain_tag = "text" if meta.get("is_text") else "synthetic"
    wb_run = wb_init(cfg, run_name=run_name, group=f"distill_{dataset_name}",
                     extra_tags=[domain_tag, baseline_tag, dataset_name, "distill"])
    wb_summary(wb_run, {
        "n_student_params": n_student,
        "n_teacher_params": n_teacher,
        "teacher_step": teacher_step,
        "teacher_M": tcfg.get("latent_M"),
        "student_M": student_cfg.latent_M,
        "distill_K": K,
        "distill_substeps": T_per_step,
        "vocab_ext": vocab_ext,
        "seq_len": seq_len,
    })

    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("wd", 0.01), betas=betas)
    lr_schedule = cfg.get("lr_schedule", "cosine")
    if lr_schedule == "constant_warmup":
        warmup = max(1, int(cfg.get("warmup_steps", 2500)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, float(s) / warmup))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    print(f"optim: AdamW lr={cfg['lr']} wd={cfg.get('wd', 0.01)} betas={betas} sched={lr_schedule}")

    ema_decay = float(cfg.get("ema", 0.0))
    ema = EMA(student, ema_decay) if ema_decay > 0 else None
    print(f"ema: {'ON decay=%.5f' % ema_decay if ema else 'off'}")

    # Resume the student (not the teacher; teacher is fixed).
    start_step = 0
    if cfg.get("resume", True):
        for _latest in sorted(out_dir.glob("ckpt_*.pt"), reverse=True):
            try:
                _state = torch.load(_latest, map_location=device, weights_only=False)
            except Exception as _e:
                print(f"skipping corrupt checkpoint {_latest.name}: {_e}")
                try:
                    _latest.unlink()
                except OSError:
                    pass
                continue
            student.load_state_dict(_state["model"])
            if _state.get("opt") is not None:
                opt.load_state_dict(_state["opt"])
            if _state.get("sched") is not None:
                sched.load_state_dict(_state["sched"])
            if ema is not None and _state.get("ema") is not None:
                ema.load_state_dict(_state["ema"])
                ema.shadow = {n: v.to(device) for n, v in ema.shadow.items()}
            elif ema is not None:
                ema.shadow = {n: p.detach().clone()
                              for n, p in student.named_parameters() if p.requires_grad}
            start_step = int(_state.get("step", 0))
            print(f"resumed student from {_latest.name} at step {start_step}")
            break

    it = infinite(loader)

    log_every = cfg.get("log_every", 100)
    ckpt_every = cfg.get("ckpt_every", 5000)
    val_every = cfg.get("val_every", 1000)

    logs: list[dict] = []
    t_start = time.time()
    for step in range(start_step + 1, cfg["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        agg: dict | None = None
        for _ in range(accum):
            batch = next(it).to(device, non_blocking=True)
            # A class-marginal teacher draws one label per row and holds it for the whole rollout,
            # so the label has to be refreshed here rather than inside the trunk, which sees one
            # sub-step at a time and cannot tell where a rollout begins. Only the MaskGIT adapter
            # defines this, and only when its label_mode is `held`.
            if hasattr(teacher, "resample_labels") and teacher.label_mode == "held":
                teacher.resample_labels(batch.shape[0], device)
            out = step_loss(batch)
            (out["loss"] / accum).backward()
            det = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(float(v)))
                   for k, v in out.items()}
            agg = det if agg is None else {k: agg[k] + det.get(k, agg[k]) for k in agg}
        out = {k: v / accum for k, v in agg.items()}
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(student)

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            log = {
                "step": step,
                "loss": float(out["loss"].item()),
                "nll": float(out["nll"].item()),
                "mi_est": float(out["mi_est"].item()),
                "router_entropy": float(out["router_entropy"].item()),
                "load_balance": float(out["load_balance"].item()),
                "mask_frac": float(out["mask_frac"].item()),
                "lr": opt.param_groups[0]["lr"],
                "elapsed": elapsed,
            }
            # The Di4C terms are what tell us whether the correlation-aware part of the
            # objective is doing any work, so they belong in the log rather than only in
            # the loss scalar.
            for k in ("data_loss", "data_loss_cor", "data_loss_indep", "consis_loss",
                      "consis_loss_cor", "consis_loss_indep", "distil_loss",
                      "time_coeff", "distil_branch_frac",
                      "x0_kl", "frac_ongrid", "mean_horizon", "mean_t"):
                if k in out:
                    log[k] = float(out[k].item())
            logs.append(log)
            wb_log(wb_run, {f"train/{k}": v for k, v in log.items() if k != "step"}, step=step)
            if objective_variant == "di4c":
                print(f"[{step:6d}] loss={log['loss']:.4f} data={log['data_loss']:.4f} "
                      f"(cor={log['data_loss_cor']:.3e} indep={log['data_loss_indep']:.4f}) "
                      f"consis={log['consis_loss']:.4f} distil={log['distil_loss']:.4f} "
                      f"dfrac={log['distil_branch_frac']:.3f} "
                      f"Hrouter={log['router_entropy']:.3f} mask={log['mask_frac']:.3f} "
                      f"lr={log['lr']:.2e}")
            else:
                _x0 = f" x0kl={log['x0_kl']:.4f}" if sup["x0_kl_coef"] > 0 else ""
                print(f"[{step:6d}] loss={log['loss']:.4f} nll={log['nll']:.4f}{_x0} "
                      f"MI={log['mi_est']:.4f} Hrouter={log['router_entropy']:.3f} "
                      f"lb={log['load_balance']:.3f} mask={log['mask_frac']:.3f} "
                      f"lr={log['lr']:.2e}")

        if step % val_every == 0:
            student.eval()
            val_nll_sum = 0.0
            val_nll_exact_sum = 0.0
            n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vb = vb.to(device, non_blocking=True)
                    vo = step_loss(vb)
                    val_nll_sum += float(vo["nll"].item()) * vb.shape[0]
                    val_nll_exact_sum += float(vo["nll_exact"].item()) * vb.shape[0]
                    n += vb.shape[0]
                    if n >= 4096:
                        break
            val_nll = val_nll_sum / max(n, 1)
            val_nll_exact = val_nll_exact_sum / max(n, 1)
            print(f"  [val@{step}] distill_nll={val_nll:.4f} distill_nll_exact={val_nll_exact:.4f}")
            if logs:
                logs[-1]["val_nll"] = val_nll
                logs[-1]["val_nll_exact"] = val_nll_exact
            wb_log(wb_run, {"val/distill_nll": val_nll,
                            "val/distill_nll_exact": val_nll_exact}, step=step)
            student.train()

        if step % ckpt_every == 0 or step == cfg["steps"]:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            _tmp = out_dir / f".ckpt_{step:07d}.pt.tmp"
            torch.save({"step": step, "model": student.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "cfg": asdict(student_cfg), "dataset": dataset_name,
                        "meta": {"tokenizer": meta.get("tokenizer"),
                                 "vocab_size": vocab_size_data,
                                 "seq_len": seq_len,
                                 "is_text": meta.get("is_text", False)},
                        "distill": {"teacher_ckpt": teacher_cfg.get("ckpt", teacher_cfg.get("run_dir")),
                                    "teacher_step": teacher_step,
                                    "K": K, "substeps": T_per_step,
                                    "objective_variant": objective_variant,
                                    "sup": sup,
                                    "di4c": di4c_cfg if objective_variant == "di4c" else None}},
                       _tmp)
            os.replace(_tmp, ckpt_path)
            _keep = int(cfg.get("keep_last_ckpts", 3))
            for _old in sorted(out_dir.glob("ckpt_*.pt"))[:-_keep]:
                try:
                    _old.unlink()
                except OSError:
                    pass
            print(f"  saved {ckpt_path}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"cfg": cfg, "logs": logs}, f, indent=2)

    if logs:
        last = logs[-1]
        wb_summary(wb_run, {
            "final/nll": last.get("nll"),
            "final/loss": last.get("loss"),
            "final/val_nll": last.get("val_nll"),
            "final/step": last.get("step"),
        })
    wb_finish(wb_run)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[],
                   help="key=val pairs to override top-level YAML keys")
    args = p.parse_args()
    cfg = load_config(args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        try:
            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v
        cfg[k] = v_parsed
    train(cfg)


if __name__ == "__main__":
    main()
