"""Thin wandb wrapper shared by both trainers.

Defaults to entity=alibilab-gsu, project=flow (override via env
WANDB_ENTITY / WANDB_PROJECT or a `wandb:` block in the yaml config).

If wandb is not installed OR WANDB_MODE=disabled OR init() raises for any reason,
every call becomes a silent no-op, and training keeps going without W&B.

Usage:
    from utils.wandb_utils import init as wb_init, log as wb_log, finish as wb_finish
    run = wb_init(cfg, run_name="AR_parity_d16_seed0",
                  extra_tags=["synthetic", "AR"], group="parity_d16")
    wb_log(run, {"train/nll": 1.23}, step=100)
    wb_finish(run)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from paths import WANDB_ENTITY as DEFAULT_ENTITY, WANDB_PROJECT as DEFAULT_PROJECT

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False


class _NoopRun:
    """Stand-in run object used when wandb is unavailable or disabled."""
    def log(self, *a: Any, **k: Any) -> None: pass
    def finish(self, *a: Any, **k: Any) -> None: pass
    @property
    def summary(self) -> dict: return {}


def init(cfg: dict, run_name: str | None = None,
         extra_tags: list[str] | None = None,
         group: str | None = None) -> Any:
    """Initialize a wandb run. Returns the run handle (or a no-op stub)."""
    if not _HAS_WANDB:
        return _NoopRun()

    wcfg = cfg.get("wandb", {}) or {}
    mode = os.environ.get("WANDB_MODE", wcfg.get("mode", "online"))
    if mode == "disabled":
        return _NoopRun()

    entity = os.environ.get("WANDB_ENTITY", wcfg.get("entity", DEFAULT_ENTITY))
    project = os.environ.get("WANDB_PROJECT", wcfg.get("project", DEFAULT_PROJECT))
    tags = list(wcfg.get("tags", []) or []) + list(extra_tags or [])
    group = group or wcfg.get("group")
    if run_name is None:
        run_name = Path(cfg.get("out_dir", "run")).name

    try:
        run = _wandb.init(
            entity=entity, project=project, name=run_name, tags=tags,
            group=group, config=cfg, mode=mode, reinit=True,
            settings=_wandb.Settings(start_method="thread") if hasattr(_wandb, "Settings") else None,
        )
        return run
    except Exception as e:
        print(f"[wandb] init failed ({type(e).__name__}: {e}); continuing without wandb")
        return _NoopRun()


def log(run: Any, data: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception:
        pass


def summary(run: Any, data: dict) -> None:
    """Write final scalars to run.summary (persisted separately from log history)."""
    if run is None:
        return
    try:
        for k, v in data.items():
            run.summary[k] = v
    except Exception:
        pass


def finish(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
