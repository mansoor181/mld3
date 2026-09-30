"""Distill a frozen teacher into an MLD3 student.

    python -m training.train --config configs/lm1b_M4.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

import data
from models.mld3 import MLD3, MLD3Config
from models.teachers import build_teacher
from models.warm_start import warm_start
from paths import config_vars
from training.losses import distill_loss


def load_config(path: str) -> dict:
    def expand(node):
        if isinstance(node, dict):
            return {k: expand(v) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(v) for v in node]
        if isinstance(node, str):
            for name, value in config_vars().items():
                node = node.replace("${" + name + "}", value)
        return node

    return expand(yaml.safe_load(Path(path).read_text()))


class EMA:
    """Parameter average with MDLM's decay warmup, keyed by name so it survives a round trip."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay, self.updates = decay, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in model.named_parameters():
            self.shadow[n].sub_((1.0 - d) * (self.shadow[n] - p))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.__dict__.update(decay=state["decay"], updates=state["updates"],
                             shadow=state["shadow"])


def build_student(cfg: dict, meta: dict, device) -> MLD3:
    model_cfg = MLD3Config(vocab_size=meta["model_vocab"], seq_len=meta["seq_len"],
                           mask_id=meta["mask_id"],
                           **{k: v for k, v in cfg["model"].items()
                              if k not in ("warm_start", "warm_start_noise")})
    return MLD3(model_cfg).to(device)


def train(cfg: dict) -> None:
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    micro = cfg["micro_batch_size"]
    accum = max(1, cfg["batch_size"] // micro)
    loader, meta = data.make_loader(cfg["dataset"], "train", micro,
                                    num_workers=cfg.get("num_workers", 2))

    teacher = build_teacher(cfg["teacher"], device, meta["seq_len"])
    student = build_student(cfg, meta, device)
    if cfg["model"].get("warm_start"):
        warm_start(student, teacher.net, cfg["model"].get("warm_start_noise", 0.02))
    print(f"[student] M={student.cfg.latent_M} "
          f"params={sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")

    d = cfg["distill"]
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("wd", 0.0),
                            betas=tuple(cfg.get("betas", (0.9, 0.999))))
    warmup = cfg.get("warmup_steps", 1000)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    ema = EMA(student, cfg["ema"]) if cfg.get("ema", 0) else None

    start = 0
    latest = sorted(out_dir.glob("ckpt_*.pt"))
    if latest:
        state = torch.load(latest[-1], map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if ema:
            ema.load_state_dict(state["ema"])
        start = state["step"]
        print(f"[resume] {latest[-1].name} at step {start}")

    def save(step: int) -> None:
        path = out_dir / f"ckpt_{step:07d}.pt"
        torch.save({"step": step, "cfg": asdict(student.cfg), "dataset": cfg["dataset"],
                    "meta": meta, "model": student.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "ema": ema.state_dict() if ema else None}, path)
        for old in sorted(out_dir.glob("ckpt_*.pt"))[:-cfg.get("keep_last_ckpts", 1)]:
            old.unlink()

    batches = data.infinite(loader)
    log_file = (out_dir / "log.jsonl").open("a")
    t0 = time.time()

    for step in range(start + 1, cfg["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        totals: dict[str, torch.Tensor] = {}
        for _ in range(accum):
            out = distill_loss(student, teacher, next(batches).to(device), d["K"],
                               d["substeps"], d["eta"], tuple(d["horizon"]),
                               d["x0_coef"], d["x0_rows"])
            (out["loss"] / accum).backward()
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + v.detach() / accum
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema:
            ema.update(student)

        if step % cfg.get("log_every", 100) == 0:
            row = {"step": step, "elapsed": round(time.time() - t0, 1),
                   **{k: round(float(v), 4) for k, v in totals.items()}}
            print(" ".join(f"{k}={v}" for k, v in row.items()))
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
        if step % cfg.get("ckpt_every", 5000) == 0 or step == cfg["steps"]:
            save(step)

    log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train(load_config(ap.parse_args().config))
