"""Seeding, checkpointing with automatic resume, and CSV logging."""
from __future__ import annotations

import csv
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Full determinism makes training much slower on GPU; seeds alone make runs repeatable
    # up to small numerical differences, which seed-to-seed statistics already cover.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Write atomically: a Colab/Kaggle disconnect during saving cannot corrupt the last good file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


class CSVLogger:
    """Appends rows to a CSV file; survives restarts (header written only once)."""

    def __init__(self, path: str | Path, fields: list[str]):
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


def code_version(repo_dir: str | Path | None = None) -> str:
    """Git commit of the code if available, else 'unversioned' (the code is uploaded as a zip)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unversioned"
