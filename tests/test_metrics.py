"""Metrics against hand-computed values."""
import numpy as np
import pytest

from polypssl.evaluation.metrics import hd95, image_metrics, size_bin, threshold_averaged_dice


def square(h, w, y, x, s):
    m = np.zeros((h, w), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_perfect_prediction():
    gt = square(100, 100, 20, 20, 30)
    r = image_metrics(gt.astype(float), gt)
    assert r["dice"] == r["iou"] == r["precision"] == r["recall"] == 1.0
    assert r["hd95_pct"] == 0.0


def test_half_overlap_dice_and_iou():
    gt = square(100, 100, 0, 0, 20)            # 400 px
    pred = np.zeros((100, 100))
    pred[0:20, 10:30] = 1.0                     # 400 px, 200 overlapping
    r = image_metrics(pred, gt)
    assert r["dice"] == pytest.approx(0.5)
    assert r["iou"] == pytest.approx(200 / 600)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)


def test_empty_prediction_is_worst_case():
    gt = square(50, 50, 10, 10, 10)
    r = image_metrics(np.zeros((50, 50)), gt)
    assert r["dice"] == 0 and r["recall"] == 0 and r["precision"] == 0
    assert r["hd95_pct"] == 100.0 and r["empty_pred"]


def _brute_force_hd95(pred, gt):
    """Reference: explicit pairwise distances between surface pixels (slow, exact)."""
    from scipy import ndimage
    def surf(m):
        return np.argwhere(m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0))
    sp, sg = surf(pred), surf(gt)
    d = np.sqrt(((sp[:, None, :] - sg[None, :, :]) ** 2).sum(-1))
    return float(np.percentile(np.concatenate([d.min(1), d.min(0)]), 95))


def test_hd95_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        gt = square(60, 70, int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(8, 25)))
        pred = square(60, 70, int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(8, 25)))
        assert hd95(pred, gt) == pytest.approx(_brute_force_hd95(pred, gt), abs=1e-6)


def test_threshold_averaged_dice_on_binary_map():
    gt = square(40, 40, 5, 5, 10)              # 100 of 1600 px
    # thresholds 1..255 give the exact mask (Dice 1); threshold 0 selects every pixel
    expected = (255 * 1.0 + 2 * 100 / (1600 + 100)) / 256
    assert threshold_averaged_dice(gt.astype(float), gt) == pytest.approx(expected)


def test_size_bins():
    assert size_bin(0.5) == "small" and size_bin(2.0) == "medium" and size_bin(25) == "large"
    gt = square(100, 100, 0, 0, 10)            # 1 % of the image
    assert image_metrics(gt.astype(float), gt)["size_bin"] == "small"
