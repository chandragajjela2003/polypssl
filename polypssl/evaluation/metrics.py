"""Per-image segmentation metrics, computed at the ORIGINAL mask resolution.

Definitions (fixed here so every number in the thesis/paper means the same thing):
- Prediction: network probability upsampled (bilinear) to the original mask size, threshold 0.5.
- dice, iou, precision, recall: standard, per image. If the prediction is empty while the polyp is not:
  dice = iou = precision = recall = 0. (Every test image in the protocol contains a polyp.)
- hd95_pct: 95th-percentile symmetric surface distance divided by the image diagonal, x100.
  Normalising makes ETIS (1225x966) and ClinicDB (384x288) comparable. Empty prediction -> 100 (worst).
- mdice_thr: Dice averaged over 256 thresholds (0..255 on the 8-bit probability), the convention of the
  PraNet MATLAB evaluation toolbox used by PraNet / Polyp-PVT. Reported ONLY to compare our
  reproductions with published numbers; our own results use the fixed-threshold `dice`.
- area_pct and size_bin: polyp area as % of the image, binned < 2 % (small), 2-10 % (medium), > 10 % (large).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SIZE_BINS = ((0.0, 2.0, "small"), (2.0, 10.0, "medium"), (10.0, 100.01, "large"))


def size_bin(area_pct: float) -> str:
    for lo, hi, name in SIZE_BINS:
        if lo <= area_pct < hi:
            return name
    return "large"


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile symmetric surface distance in pixels. Both masks must be non-empty."""
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def threshold_averaged_dice(prob: np.ndarray, gt: np.ndarray) -> float:
    """Mean Dice over thresholds t = 0..255 applied to the 8-bit map, PraNet-toolbox style.
    As in the PraNet / Polyp-PVT test scripts, the map is first min-max normalised to [0, 1]
    (they save it that way before the toolbox reads it). This reproduces the published convention
    approximately; it is used only to compare reproductions with published numbers."""
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    q = np.clip(np.round(prob * 255), 0, 255).astype(np.int64)
    fg_hist = np.bincount(q[gt], minlength=256)
    bg_hist = np.bincount(q[~gt], minlength=256)
    tp = np.cumsum(fg_hist[::-1])[::-1]          # pixels with value >= t that are polyp
    fp = np.cumsum(bg_hist[::-1])[::-1]
    n_gt = gt.sum()
    dice = 2 * tp / np.maximum(tp + fp + n_gt, 1)
    return float(dice.mean())


def image_metrics(prob: np.ndarray, gt: np.ndarray, full: bool = True) -> dict:
    """prob: float probabilities at original resolution, gt: binary {0,1} at the same resolution.
    full=False skips the two slow metrics (HD95, threshold-averaged Dice; set to NaN), used by the many
    re-evaluations of the C1 diagnosis, which only need Dice."""
    gt = gt.astype(bool)
    pred = prob >= 0.5
    tp = np.logical_and(pred, gt).sum()
    n_pred, n_gt = pred.sum(), gt.sum()
    h, w = gt.shape
    diag = float(np.hypot(h, w))
    area_pct = 100.0 * n_gt / gt.size
    out = {
        "dice": float(2 * tp / (n_pred + n_gt)) if (n_pred + n_gt) > 0 else 1.0,
        "iou": float(tp / (n_pred + n_gt - tp)) if (n_pred + n_gt - tp) > 0 else 1.0,
        "precision": float(tp / n_pred) if n_pred > 0 else 0.0,
        "recall": float(tp / n_gt) if n_gt > 0 else 1.0,
        "hd95_pct": (100.0 * hd95(pred, gt) / diag if (n_pred > 0 and n_gt > 0) else 100.0) if full
        else float("nan"),
        "mdice_thr": threshold_averaged_dice(prob, gt) if (n_gt > 0 and full) else float("nan"),
        "empty_pred": bool(n_pred == 0),
        "area_pct": float(area_pct),
        "size_bin": size_bin(area_pct),
        "height": h,
        "width": w,
    }
    return out
