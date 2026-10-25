"""Evaluate a trained model on every test set and write:

    results/<run>/per_image.csv   one row per test image (dataset, id, size bin, all metrics)
    results/<run>/summary.csv     mean per dataset (all images, and per polyp-size bin)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import TestDataset, normalize, test_collate
from ..models import final_logits
from .metrics import image_metrics

METRICS = ["dice", "iou", "precision", "recall", "hd95_pct", "mdice_thr"]


@torch.no_grad()
def predict_probabilities(model, batch_images: torch.Tensor, sizes: list[tuple[int, int]],
                          device, amp: bool = True) -> list[np.ndarray]:
    x = normalize(batch_images.to(device))
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = final_logits(model, model(x))
    probs = []
    for i, (h, w) in enumerate(sizes):
        p = F.interpolate(logits[i:i + 1].float(), size=(h, w), mode="bilinear", align_corners=False)
        probs.append(torch.sigmoid(p)[0, 0].cpu().numpy())
    return probs


def evaluate_model(model, test_sets: dict[str, list[dict]], img_size: int, device,
                   batch_size: int = 8, num_workers: int = 2, amp: bool = True, transform=None,
                   verbose: bool = True, full_metrics: bool = True) -> pd.DataFrame:
    model.eval()
    rows = []
    for name, samples in test_sets.items():
        loader = DataLoader(TestDataset(samples, img_size, transform), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=test_collate)
        for batch in loader:
            sizes = [m.shape for m in batch["mask"]]
            probs = predict_probabilities(model, batch["image"], sizes, device, amp)
            for prob, gt, sid in zip(probs, batch["mask"], batch["id"]):
                rows.append({"dataset": name, "id": sid, **image_metrics(prob, gt, full_metrics)})
        if verbose:
            print(f"  {name:20s} n={len(samples):4d}  dice={np.mean([r['dice'] for r in rows if r['dataset'] == name]):.4f}")
    return pd.DataFrame(rows)


MISS_DICE = 0.1   # an image with Dice < 0.1 counts as a MISSED polyp (C1 finding: misses dominate the gap)


def summarise(per_image: pd.DataFrame) -> pd.DataFrame:
    """Means per dataset (all images and per size bin), plus
    miss_rate     = share of images with Dice < 0.1 (polyp essentially not found), and
    dice_detected = mean Dice over the other images (boundary quality of found polyps)."""
    per_image = per_image.assign(miss_rate=(per_image["dice"] < MISS_DICE).astype(float),
                                 dice_detected=per_image["dice"].where(per_image["dice"] >= MISS_DICE))
    metrics = METRICS + ["miss_rate", "dice_detected"]
    overall = per_image.groupby("dataset")[metrics].mean()
    overall["n"] = per_image.groupby("dataset").size()
    overall["empty_pred"] = per_image.groupby("dataset")["empty_pred"].sum()
    overall.insert(0, "size_bin", "all")
    by_size = per_image.groupby(["dataset", "size_bin"])[metrics].mean()
    by_size["n"] = per_image.groupby(["dataset", "size_bin"]).size()
    by_size = by_size.reset_index(level="size_bin")
    return pd.concat([overall, by_size]).reset_index().sort_values(["dataset", "size_bin"])


def save_results(per_image: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    summary = summarise(per_image)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary
