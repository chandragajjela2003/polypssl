"""Measure seconds per iteration and GPU memory for a config, WITHOUT saving anything.
The result decides the final number of training iterations that fits the free GPU budget.

Usage:
    python speed_test.py --config configs/stageA/fulnet_c_mscan_full.yaml --iterations 100
"""
import argparse
import time

import torch

from polypssl.config import load_config
from polypssl.data import normalize
from polypssl.fal import fal
from polypssl.losses import segmentation_loss
from polypssl.models import build_model
from polypssl.paths import get_locations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config(args.config)
    loc = get_locations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, weights_dir=loc.weights).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bs, size = cfg["train"]["batch_size"], cfg["data"]["img_size"]
    x = torch.rand(bs, 3, size, size, device=device)
    y = (torch.rand(bs, 1, size, size, device=device) > 0.9).float()
    gen = torch.Generator().manual_seed(0)
    for i in range(args.iterations + 10):          # first 10 iterations are warm-up, not timed
        if i == 10:
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = segmentation_loss(model(normalize(x)), y, cfg["loss"])
            if cfg["fal"]["mode"] != "off":
                xf = fal(x, "corrected", gen) if cfg["fal"]["mode"] == "corrected" else fal(normalize(x), "faithful", gen)
                xf = normalize(xf) if cfg["fal"]["mode"] == "corrected" else xf
                loss = 0.9 * loss + 0.1 * segmentation_loss(model(xf), y, cfg["loss"])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - t0) / args.iterations
    mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    sec *= len(cfg["train"]["multiscale"])          # Polyp-PVT recipe: one step per scale (approximation)
    print(f"[speed] {cfg['name']} on {gpu}: {sec:.3f} s/iteration (worst case: FAL view every iteration), "
          f"peak memory {mem:.2f} GB")
    for iters in (10_000, 15_000, 18_125, 20_000):
        print(f"        {iters:>6} iterations ~ {iters * sec / 3600:.1f} h")


if __name__ == "__main__":
    main()
