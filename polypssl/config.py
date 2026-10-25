"""Experiment configuration.

Each experiment is one YAML file in configs/. Values not given in the file fall back to DEFAULTS
below, so a config only needs to state what differs. The fully merged config is saved next to
every run, so any result can be traced to the exact settings that produced it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": None,                 # run name; results go to runs/<name>/seed<seed>/
    "seed": 0,
    "method": "supervised",       # supervised | mean_teacher
    "model": {
        "name": "fulnet_c",       # fulnet_c | polyp_pvt | unet_r34
        "encoder": "mscan_t",     # mscan_t for FULNet-C; ignored by the others
        "ual": "corrected",       # corrected | faithful | off  (polyp_pvt: corrected | off)
        "pretrained": True,
    },
    "data": {
        "img_size": 320,
        "split": "full",          # "full" = all 1,450 training images labelled, else a split file name
        "augment": "fulnet",      # fulnet (FULNet ver=2 augmentations) | none (Polyp-PVT official default)
        "num_workers": 2,
    },
    "fal": {
        "mode": "corrected",      # corrected | faithful | off
        "beta": 0.01,             # low-frequency window half-width = floor(img_size * beta)
        "lam": 0.5,               # amplitude mixing weight (FULNet uses 0.5)
        "aug_weight": 0.1,        # loss weight of the FAL view (FULNet: 0.9 * clean + 0.1 * FAL)
        "prob": 0.5,              # probability of adding the FAL loss in an iteration (FULNet: 0.5)
    },
    "loss": {
        "type": "ari",            # ari (FULNet deep supervision) | structure (Polyp-PVT)
        "mode": "corrected",      # corrected | faithful
    },
    "ssl": {                      # used by method = mean_teacher (all values fixed in advance, no tuning)
        "unlabelled_batch": 8,    # unlabelled images per step (labelled: train.batch_size)
        "ema": 0.99,              # teacher = EMA of the student (Mean Teacher; RD-Net CVPR 2025 uses 0.99)
        "threshold": 0.85,        # pixel pseudo-label used only if teacher confidence >= 0.85 (as RD-Net)
        "weight": 1.0,            # unlabelled loss weight (FixMatch-style lambda_u = 1)
        "rampup": 0.1,            # weight ramps up (sigmoid) over the first 10 % of iterations
    },
    "zoom": {                     # ZoomMatch (Stage E); see polypssl/ssl_views.py
        "mode": "off",            # off | targeted (ZoomMatch) | fixed2 (control: MixPL-style 2x2 mosaic)
        "prob": 0.5,              # fraction of images turned into a zoomed-out mosaic
        "area_min": 0.003,        # target polyp area fraction sampled log-uniformly in [0.3 %, 15 %]
        "area_max": 0.15,
        "max_grid": 4,            # at most a 4 x 4 mosaic (zoom factor 1/4)
        "labelled": True,         # also zoom labelled images (with their exact masks)
    },
    "train": {
        "iterations": 18125,      # FULNet default (= 100 epochs of 1,450 images at batch 8)
        "batch_size": 8,
        "lr": 5.0e-5,
        "weight_decay": 1.0e-4,
        "grad_clip": 2.0,
        "grad_clip_mode": "norm", # norm (FULNet: clip_grad_norm_) | value (Polyp-PVT: clamp each gradient)
        "multiscale": [1.0],      # Polyp-PVT: [0.75, 1.0, 1.25], one optimiser step per scale
        "amp": True,
        "ckpt_every": 500,
        "log_every": 50,
        "val_every": 1000,        # only used when the split has a validation subset
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    """path: a YAML config file, or None to build the config from DEFAULTS + overrides only."""
    user = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    cfg = _merge(cfg, overrides or {})
    for section, key, disabled_value in (
        ("model", "ual", "off"),
        ("fal", "mode", "off"),
        ("data", "augment", "none"),
        ("zoom", "mode", "off"),
    ):
        if cfg[section][key] is False:
            cfg[section][key] = disabled_value
    if not cfg.get("name"):
        if path is None:
            raise ValueError("a config built without a file needs a 'name'")
        cfg["name"] = Path(path).stem
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    m = cfg["model"]
    assert m["name"] in {"fulnet_c", "polyp_pvt", "unet_r34"}, m["name"]
    if not isinstance(m["ual"], str) or m["ual"] not in {"corrected", "faithful", "off"}:
        raise ValueError(f"model.ual must be 'corrected', 'faithful', or 'off'; got {m['ual']!r}")
    if m["name"] == "polyp_pvt":
        if m["ual"] not in {"corrected", "off"}:
            raise ValueError("Polyp-PVT supports only model.ual='corrected' or 'off'")
    if not isinstance(cfg["fal"]["mode"], str) or cfg["fal"]["mode"] not in {
        "corrected", "faithful", "off"
    }:
        raise ValueError(
            f"fal.mode must be 'corrected', 'faithful', or 'off'; got {cfg['fal']['mode']!r}")
    assert cfg["loss"]["type"] in {"ari", "structure"}, cfg["loss"]["type"]
    assert cfg["loss"]["mode"] in {"corrected", "faithful"}, cfg["loss"]["mode"]
    if not isinstance(cfg["data"]["augment"], str) or cfg["data"]["augment"] not in {"fulnet", "none"}:
        raise ValueError(f"data.augment must be 'fulnet' or 'none'; got {cfg['data']['augment']!r}")
    assert cfg["train"]["grad_clip_mode"] in {"norm", "value"}, cfg["train"]["grad_clip_mode"]
    if cfg["method"] not in {"supervised", "mean_teacher"}:
        raise ValueError(f"method must be 'supervised' or 'mean_teacher'; got {cfg['method']!r}")
    if cfg["zoom"]["mode"] not in {"off", "targeted", "fixed2"}:
        raise ValueError(f"zoom.mode must be 'off', 'targeted' or 'fixed2'; got {cfg['zoom']['mode']!r}")
    if cfg["zoom"]["mode"] != "off" and cfg["method"] != "mean_teacher":
        raise ValueError("ZoomMatch (zoom.mode) requires method = mean_teacher")


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
