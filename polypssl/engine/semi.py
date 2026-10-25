"""Mean Teacher semi-supervised training (run B2) with optional ZoomMatch (runs E1, E2).

Every iteration:
  labelled batch  (8 images + masks): supervised loss exactly as in supervised training (structure loss,
                  corrected FAL view with prob 0.5). With ZoomMatch, labelled images may first be turned into
                  zoomed-out mosaics using their exact masks.
  unlabelled batch (8 images, masks never read):
      1. teacher (EMA copy of the student, eval mode) predicts on the clean image -> probability p;
      2. pseudo-label = (p >= 0.5); confidence weight = 1 where max(p, 1 - p) >= 0.85, else 0
         (confidence-thresholded pseudo-labels as in RD-Net, CVPR 2025);
      3. ZoomMatch (if on): image, pseudo-label and confidence are placed into a zoomed-out mosaic with the
         same geometry (ssl_views.zoom_mosaic), so the student sees SMALL polyps whose labels come from
         the teacher's prediction at the ORIGINAL scale;
      4. the student sees a strong photometric perturbation of that image (same geometry);
      5. unlabelled loss = confidence-weighted BCE(student, pseudo-label).
  total = supervised + weight(t) * unlabelled, weight ramped up (sigmoid) over the first 10 %.
  The teacher is updated as an exponential moving average (0.99) of the student after every step.
Final model = the EMA TEACHER (Mean Teacher convention: the averaged model is the more accurate one);
the student weights are also stored in model_final.pth under "student".
All hyper-parameters are fixed in config.py DEFAULTS before any run (no tuning on any test data).
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import save_config
from ..data import LabelledDataset, UnlabelledDataset, normalize, pranet_train
from ..fal import fal
from ..losses import segmentation_loss
from ..models import build_model, final_logits
from ..paths import Locations
from ..ssl_views import strong_photometric, zoom_mosaic
from ..utils import CSVLogger, code_version, rng_state, set_rng_state, set_seed
from .common import finalize, is_finished, load_resume, save_resume, trim_log
from .supervised import _infinite, _val_dice, resolve_split


def _rampup(step: int, total: int, fraction: float) -> float:
    span = max(1.0, fraction * total)
    t = min(step / span, 1.0)
    return math.exp(-5.0 * (1.0 - t) ** 2)


@torch.no_grad()
def _ema_update(teacher, student, step: int, decay: float) -> None:
    alpha = min(1.0 - 1.0 / (step + 1), decay)
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(alpha).add_(sp.detach(), alpha=1.0 - alpha)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def train_mean_teacher(cfg: dict, loc: Locations, device: torch.device | None = None,
                       run_dir: Path | None = None) -> Path:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, ssl, zc = cfg["train"], cfg["ssl"], cfg["zoom"]
    if list(t["multiscale"]) != [1.0]:
        raise ValueError("mean_teacher uses single-scale training (train.multiscale must be [1.0])")
    if cfg["fal"]["mode"] == "faithful":
        raise ValueError("mean_teacher supports fal.mode 'corrected' or 'off' (faithful FAL is for reproductions)")
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
    lab = [by_id[i] for i in split["labelled_train"]]
    val = [by_id[i] for i in split["labelled_val"]]
    unl = [by_id[i] for i in split["unlabelled"]]
    if not unl:
        raise ValueError("mean_teacher needs unlabelled images (use a 2.5/5/10/20% split, not 'full')")
    print(f"[data] {len(lab)} labelled train, {len(val)} labelled val (run-check only), "
          f"{len(unl)} unlabelled (images only) | split {split['method']} {split['budget_pct']}% seed {split['seed']}")
    print(f"[method] Mean Teacher | ZoomMatch: {zc['mode']}")

    student = build_model(cfg, weights_dir=loc.weights).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    optim = torch.optim.AdamW(student.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_it, ep_l, ep_u = 0, 0, 0
    state = load_resume(run_dir)
    if state is not None:
        student.load_state_dict(state["model"])
        teacher.load_state_dict(state["teacher"])
        optim.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        start_it, ep_l, ep_u = state["iteration"], state["epoch"], state["epoch_u"]
        trim_log(run_dir, start_it)
        print(f"[resume] from iteration {start_it}")

    size = cfg["data"]["img_size"]
    lab_ds = LabelledDataset(lab, size, augment=cfg["data"]["augment"] == "fulnet", seed=cfg["seed"])
    unl_ds = UnlabelledDataset(unl, size, seed=cfg["seed"])
    lab_batches = _infinite(lab_ds, t["batch_size"], cfg["data"]["num_workers"], cfg["seed"], ep_l)
    unl_batches = _infinite(unl_ds, ssl["unlabelled_batch"], cfg["data"]["num_workers"], cfg["seed"] + 7, ep_u)
    log = CSVLogger(run_dir / "log.csv", ["iteration", "epoch", "epoch_unlabelled", "loss", "loss_labelled",
                                           "loss_unlabelled", "unsup_weight", "confident_px", "zoomed_unlabelled",
                                           "val_dice", "sec_per_it", "gpu_mem_gb"])
    fal_cfg, loss_cfg, w_fal = cfg["fal"], cfg["loss"], cfg["fal"]["aug_weight"]
    student.train()
    tic = time.time()

    for it in range(start_it, t["iterations"]):
        ep_l, (x, y) = next(lab_batches)
        ep_u, xu = next(unl_batches)
        x, y, xu = x.to(device, non_blocking=True), y.to(device, non_blocking=True), xu.to(device, non_blocking=True)
        gen = torch.Generator().manual_seed(cfg["seed"] * 1_000_003 + it)

        # ---- labelled branch (identical to supervised training, plus optional zoom with exact masks)
        if zc["mode"] != "off" and zc["labelled"]:
            x, y, _, _ = zoom_mosaic(x, y, None, zc["mode"], zc, gen)
        use_fal = fal_cfg["mode"] == "corrected" and torch.rand(1, generator=gen).item() < fal_cfg["prob"]
        x_fal_n = normalize(fal(x, "corrected", gen, beta=fal_cfg["beta"], lam=fal_cfg["lam"])) if use_fal else None

        # ---- unlabelled branch: teacher pseudo-labels at the original scale
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            p_t = torch.sigmoid(final_logits(teacher, teacher(normalize(xu))).float())
        conf = (torch.maximum(p_t, 1 - p_t) >= ssl["threshold"]).float()
        conf_ratio = conf.mean().item()                     # share of confident pixels (before any zoom)
        zoomed = 0.0
        if zc["mode"] != "off":
            xu, p_t, conf, grids = zoom_mosaic(xu, p_t, conf, zc["mode"], zc, gen)
            zoomed = float(np.mean([g > 1 for g in grids]))
        pseudo = (p_t >= 0.5).float()
        xu_s = strong_photometric(xu, gen)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_l = segmentation_loss(student(normalize(x)), y, loss_cfg)
            if x_fal_n is not None:
                loss_l = (1 - w_fal) * loss_l + w_fal * segmentation_loss(student(x_fal_n), y, loss_cfg)
            logit_u = final_logits(student, student(normalize(xu_s)))
        with torch.autocast(device_type=device.type, enabled=False):
            bce = F.binary_cross_entropy_with_logits(logit_u.float(), pseudo, reduction="none")
            loss_u = (bce * conf).sum() / conf.sum().clamp_min(1.0)
            lam = ssl["weight"] * _rampup(it, t["iterations"], ssl["rampup"])
            loss = loss_l.float() + lam * loss_u

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        if t["grad_clip_mode"] == "norm":
            torch.nn.utils.clip_grad_norm_(student.parameters(), t["grad_clip"])
        else:
            torch.nn.utils.clip_grad_value_(student.parameters(), t["grad_clip"])
        scaler.step(optim)
        scaler.update()
        _ema_update(teacher, student, it, ssl["ema"])

        step = it + 1
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss became {loss.item()} at iteration {step}; stopping this run")
        v = _val_dice(teacher, val, size, device, use_amp) if val and step % t["val_every"] == 0 else float("nan")
        teacher.eval()
        if step % t["log_every"] == 0 or step == t["iterations"]:
            sec = (time.time() - tic) / t["log_every"]
            tic = time.time()
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            log.log({"iteration": step, "epoch": ep_l, "epoch_unlabelled": ep_u, "loss": loss.item(),
                     "loss_labelled": loss_l.item(), "loss_unlabelled": loss_u.item(), "unsup_weight": lam,
                     "confident_px": conf_ratio, "zoomed_unlabelled": zoomed, "val_dice": v,
                     "sec_per_it": sec, "gpu_mem_gb": mem})
            print(f"[{step}/{t['iterations']}] loss={loss.item():.4f} (lab {loss_l.item():.4f}, unlab "
                  f"{loss_u.item():.4f} x{lam:.2f}) conf_px={conf_ratio:.2f} sec/it={sec:.3f}"
                  + ("" if np.isnan(v) else f" val_dice={v:.4f}"))
        if step % t["ckpt_every"] == 0 and step < t["iterations"]:
            save_resume(run_dir, {"model": student.state_dict(), "teacher": teacher.state_dict(),
                                  "optim": optim.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                                  "iteration": step, "epoch": ep_l, "epoch_u": ep_u, "config": cfg})
    final = finalize(run_dir, teacher, cfg, t["iterations"], extra={"student": student.state_dict()})
    print(f"[done] final model (EMA teacher): {final}")
    return run_dir
