"""Checkpoint files shared by every training method.

Inside a run folder:
    checkpoint_resume.pth  written every `ckpt_every` iterations while training (model + optimiser + RNG);
                           used to resume after a disconnect; DELETED automatically once training finishes.
    model_final.pth        written once training finishes: the trained weights + the exact config.
                           Kept permanently; this is the file used for evaluation and analysis.
(Older runs used `last.pth`; it is still recognised for resuming and evaluation.)
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..utils import load_checkpoint, save_checkpoint

RESUME = "checkpoint_resume.pth"
FINAL = "model_final.pth"
LEGACY = "last.pth"


def is_finished(run_dir: Path) -> bool:
    return (Path(run_dir) / FINAL).exists()


def load_resume(run_dir: Path) -> dict | None:
    for name in (RESUME, LEGACY):
        state = load_checkpoint(Path(run_dir) / name)
        if state is not None:
            return state
    return None


def trim_log(run_dir: Path, iteration: int) -> None:
    """After resuming from `iteration`, drop log rows written after that checkpoint (they are re-run),
    so log.csv never contains an iteration twice."""
    path = Path(run_dir) / "log.csv"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = lines[:1] + [ln for ln in lines[1:] if ln.strip() and int(float(ln.split(",", 1)[0])) <= iteration]
    if len(keep) != len(lines):
        path.write_text("".join(keep), encoding="utf-8")


def save_resume(run_dir: Path, payload: dict) -> None:
    save_checkpoint(Path(run_dir) / RESUME, payload)


def finalize(run_dir: Path, model: torch.nn.Module, cfg: dict, iteration: int, extra: dict | None = None) -> Path:
    """Write model_final.pth, then remove the (large) resume checkpoint."""
    run_dir = Path(run_dir)
    payload = {"model": model.state_dict(), "config": cfg, "iteration": iteration, **(extra or {})}
    save_checkpoint(run_dir / FINAL, payload)
    torch.load(run_dir / FINAL, map_location="cpu", weights_only=False)      # verify it is readable
    for name in (RESUME, LEGACY):
        (run_dir / name).unlink(missing_ok=True)
    return run_dir / FINAL


def load_final(run_dir: Path) -> dict:
    """The trained model of a run (model_final.pth, or a finished legacy last.pth)."""
    run_dir = Path(run_dir)
    for name in (FINAL, LEGACY):
        state = load_checkpoint(run_dir / name)
        if state is not None:
            return state
    raise FileNotFoundError(f"No trained model in {run_dir}")


def build_trained_model(run_dir: Path, device: torch.device):
    """Rebuild the network of a finished run and load its trained weights. Returns (model, config, state)."""
    from ..models import build_model
    state = load_final(run_dir)
    cfg = state["config"]
    if state.get("iteration", 0) < cfg["train"]["iterations"]:
        print(f"[warn] {run_dir} is not finished ({state.get('iteration')}/{cfg['train']['iterations']})")
    model = build_model({**cfg, "model": {**cfg["model"], "pretrained": False}}).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), cfg, state
