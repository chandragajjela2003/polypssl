"""Shared test fixtures: a tiny synthetic dataset with the exact PraNet folder layout.

Each synthetic image is a dark background with one bright square (the "polyp"); its mask is that square.
This lets tests check that augmentations keep image and mask aligned, without real data.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_pair(img_dir: Path, mask_dir: Path, stem: str, size=(96, 120), seed=0):
    rng = np.random.default_rng(seed)
    h, w = size
    img = (rng.uniform(0, 60, (h, w, 3))).astype(np.uint8)
    y0, x0 = rng.integers(10, h // 2), rng.integers(10, w // 2)
    s = int(rng.integers(15, 30))
    img[y0:y0 + s, x0:x0 + s] = 230
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y0 + s, x0:x0 + s] = 255
    cv2.imwrite(str(img_dir / f"{stem}.png"), img)
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)


@pytest.fixture()
def fake_data(tmp_path: Path) -> Path:
    """data/TrainDataset/image + masks (6 Kvasir-style, 4 ClinicDB-style) and TestDataset/<set>/images + masks."""
    train_img, train_mask = tmp_path / "TrainDataset" / "image", tmp_path / "TrainDataset" / "masks"
    train_img.mkdir(parents=True)
    train_mask.mkdir(parents=True)
    for i in range(6):
        _write_pair(train_img, train_mask, f"cju{i:03d}abc", seed=i)
    for i in range(1, 5):
        _write_pair(train_img, train_mask, str(i), seed=100 + i)
    for name in ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]:
        d_img, d_mask = tmp_path / "TestDataset" / name / "images", tmp_path / "TestDataset" / name / "masks"
        d_img.mkdir(parents=True)
        d_mask.mkdir(parents=True)
        for i in range(2):
            _write_pair(d_img, d_mask, f"{name}_{i}", size=(80, 100), seed=200 + i)
    return tmp_path
