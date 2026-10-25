"""Supervised training (Stage A: reproduction runs and supervised baselines).

FULNet training schedule (train.py of the released code, first_phase_iterations = 0):
    every iteration: loss_clean = DS-loss(model(x), y)
    with probability fal.prob (0.5): loss = 0.9 * loss_clean + 0.1 * DS-loss(model(FAL(x)), y)
FAL modes: see fal.py. In faithful mode the FAL view is forwarded EVERY iteration (as released, which
also updates BatchNorm statistics with FAL views); in corrected mode it is forwarded only when its loss
is used. With fal.mode = off the model trains on clean views only.

Polyp-PVT recipe (train.multiscale = [0.75, 1, 1.25]): each batch is rescaled to every rate
(round(size*rate/32)*32, bilinear, align_corners=True, as in the official Train.py) and the optimiser takes
one step per rate. The FAL view of the batch (if used this iteration) is rescaled the same way.

Robustness: checkpoint every `ckpt_every` iterations (atomic write) and automatic resume, so a Colab or
Kaggle disconnect loses at most `ckpt_every` iterations.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import save_config
from ..data import LabelledDataset, normalize, pranet_train
from ..data.splits import check_split, full_split, load_split
from ..fal import fal
import torch.nn.functional as F

from ..losses import segmentation_loss
from ..models import build_model, final_logits
from ..paths import Locations
from ..utils import CSVLogger, code_version, rng_state, set_rng_state, set_seed
from .common import finalize, is_finished, load_resume, save_resume, trim_log


def _infinite(dataset: LabelledDataset, batch_size: int, num_workers: int, seed: int, start_epoch: int):
    """Yields batches forever; each pass over the data reshuffles and re-draws augmentations."""
    epoch = start_epoch
    while True:
        dataset.epoch = epoch
        g = torch.Generator().manual_seed(seed * 100_003 + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for batch in loader:
            yield epoch, batch
        epoch += 1


CODE_SPLITS = Path(__file__).resolve().parents[2] / "splits"   # split files shipped with the code


def _rescale(x_n, y, x_fal_n, rate: float, size: int):
    """Polyp-PVT multi-scale training: resize to round(size*rate/32)*32 (bilinear, align_corners=True)."""
    if rate == 1.0:
        return x_n, y, x_fal_n
    new = int(round(size * rate / 32) * 32)
    r = lambda t: None if t is None else F.interpolate(t, size=(new, new), mode="bilinear", align_corners=True)
    return r(x_n), r(y), r(x_fal_n)


def resolve_split(cfg: dict, loc: Locations, samples: list[dict]) -> dict:
    """Split files are generated once (on the real data) and shipped inside the code folder, so Colab and
    Kaggle always use identical labelled images. Falls back to outputs/splits for new splits."""
    name = cfg["data"]["split"]
    if name == "full":
        split = full_split(samples)
    else:
        for folder in (CODE_SPLITS, loc.splits):
            if (folder / f"{name}.json").exists():
                split = load_split(folder / f"{name}.json")
                break
        else:
            raise FileNotFoundError(f"Split '{name}' not found in {CODE_SPLITS} or {loc.splits}")
    check_split(split, samples)
    return split


@torch.no_grad()
def _val_dice(model, samples, img_size, device, amp) -> float:
    """Sanity check only (detects broken runs). Never used for model selection."""
    if not samples:
        return float("nan")
    model.eval()
    ds = LabelledDataset(samples, img_size, augment=False)
    dices = []
    for x, y in DataLoader(ds, batch_size=8, shuffle=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = (torch.sigmoid(final_logits(model, model(normalize(x))).float()) >= 0.5).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        dices += (2 * inter / (pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-8)).tolist()
    model.train()
    return float(np.mean(dices))


def train(cfg: dict, loc: Locations, device: torch.device | None = None, run_dir: Path | None = None) -> Path:
    """Train one run. run_dir defaults to outputs/runs/<name>/seed<k> (run.py passes its own folder)."""
    if cfg["method"] == "mean_teacher":
        from .semi import train_mean_teacher
        return train_mean_teacher(cfg, loc, device, run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    run_dir = Path(run_dir) if run_dir else loc.runs / cfg["name"] / f"seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_finished(run_dir):
        print(f"[done] {run_dir} already has model_final.pth; not retraining")
        return run_dir
    cfg["code_version"] = code_version()
    save_config(cfg, run_dir / "config.yaml")
    set_seed(cfg["seed"])

    samples = pranet_train(loc.data)
    split = resolve_split(cfg, loc, samples)
    by_id = {s["id"]: s for s in samples}
    train_samples = [by_id[i] for i in split["labelled_train"]]
    val_samples = [by_id[i] for i in split["labelled_val"]]
    print(f"[data] {len(train_samples)} labelled train, {len(val_samples)} labelled val, "
          f"{len(split['unlabelled'])} unlabelled (unused in supervised training)")

    model = build_model(cfg, weights_dir=loc.weights).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_it, start_epoch = 0, 0
    state = load_resume(run_dir)
    if state is not None:
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        start_it, start_epoch = state["iteration"], state["epoch"]
        trim_log(run_dir, start_it)
        print(f"[resume] from iteration {start_it}")
    if start_it >= t["iterations"]:
        finalize(run_dir, model, cfg, start_it)
        print(f"[done] finished run converted to {run_dir / 'model_final.pth'}")
        return run_dir

    size = cfg["data"]["img_size"]
    dataset = LabelledDataset(train_samples, size, augment=cfg["data"]["augment"] == "fulnet", seed=cfg["seed"])
    batches = _infinite(dataset, t["batch_size"], cfg["data"]["num_workers"], cfg["seed"], start_epoch)
    log = CSVLogger(run_dir / "log.csv", ["iteration", "epoch", "loss", "loss_clean", "loss_fal",
                                           "val_dice", "sec_per_it", "gpu_mem_gb"])
    fal_cfg, loss_cfg, w = cfg["fal"], cfg["loss"], cfg["fal"]["aug_weight"]
    model.train()
    tic, epoch = time.time(), start_epoch

    for it in range(start_it, t["iterations"]):
        epoch, (x, y) = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        gen = torch.Generator().manual_seed(cfg["seed"] * 1_000_003 + it)
        use_fal = fal_cfg["mode"] != "off" and torch.rand(1, generator=gen).item() < fal_cfg["prob"]
        # the FAL view is built once per iteration from the clean batch
        x_fal_n = None
        if fal_cfg["mode"] == "faithful":                # forwarded every iteration, as released
            x_fal_n = fal(normalize(x), "faithful", gen, lam=fal_cfg["lam"])
        elif fal_cfg["mode"] == "corrected" and use_fal:
            x_fal_n = normalize(fal(x, "corrected", gen, beta=fal_cfg["beta"], lam=fal_cfg["lam"]))
        x_n = normalize(x)

        for rate in t["multiscale"]:                     # [1.0] = one step; Polyp-PVT: 3 steps
            xs, ys, xfs = _rescale(x_n, y, x_fal_n, rate, size)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss_clean = segmentation_loss(model(xs), ys, loss_cfg)
                loss_fal = torch.zeros((), device=device)
                if xfs is not None:
                    out_fal = model(xfs)
                    if use_fal:
                        loss_fal = segmentation_loss(out_fal, ys, loss_cfg)
                loss = (1 - w) * loss_clean + w * loss_fal if use_fal else loss_clean
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if t["grad_clip_mode"] == "norm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            else:
                torch.nn.utils.clip_grad_value_(model.parameters(), t["grad_clip"])
            scaler.step(optim)
            scaler.update()

        step = it + 1
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss became {loss.item()} at iteration {step}; stopping this run")
        val = _val_dice(model, val_samples, cfg["data"]["img_size"], device, use_amp) \
            if val_samples and step % t["val_every"] == 0 else float("nan")
        if step % t["log_every"] == 0 or step == t["iterations"]:
            sec = (time.time() - tic) / t["log_every"]
            tic = time.time()
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            log.log({"iteration": step, "epoch": epoch, "loss": loss.item(), "loss_clean": loss_clean.item(),
                     "loss_fal": loss_fal.item(), "val_dice": val, "sec_per_it": sec, "gpu_mem_gb": mem})
            print(f"[{step}/{t['iterations']}] loss={loss.item():.4f} sec/it={sec:.3f}"
                  + ("" if np.isnan(val) else f" val_dice={val:.4f}"))
        if step % t["ckpt_every"] == 0 and step < t["iterations"]:
            save_resume(run_dir, {"model": model.state_dict(), "optim": optim.state_dict(),
                                  "scaler": scaler.state_dict(), "rng": rng_state(),
                                  "iteration": step, "epoch": epoch, "config": cfg})
    final = finalize(run_dir, model, cfg, t["iterations"])
    print(f"[done] final model: {final}")
    return run_dir
