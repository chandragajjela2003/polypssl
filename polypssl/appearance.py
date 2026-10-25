"""Appearance descriptor and appearance statistics (shared by ACLS selection and the C1 diagnosis).

Descriptor (57 numbers per image, format-controlled as decided in the advisor review, NOVELTY.md A4/A5):
- take the CENTRAL 60 % square crop (removes black borders, text overlays and aspect-ratio effects,
  which otherwise identify the dataset by its file format rather than its appearance);
- resize to 320 x 320;
- per RGB channel: log of the radially averaged Fourier amplitude at radii 0..16 (low-frequency
  "style": overall colour, illumination, smooth shading)  -> 3 x 17 = 51 numbers;
- per RGB channel: mean and standard deviation of the pixel values  -> 6 numbers.
Uses images only, never masks.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE, RADII = 320, 16
_yy, _xx = np.mgrid[:SIZE, :SIZE]
_RAD = np.sqrt((_yy - SIZE // 2) ** 2 + (_xx - SIZE // 2) ** 2).astype(int)
_RING = [(_RAD == r) for r in range(RADII + 1)]


def central_crop(img: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = img.shape[:2]
    s = int(frac * min(h, w))
    t, l = (h - s) // 2, (w - s) // 2
    return img[t:t + s, l:l + s]


def descriptor(img: np.ndarray) -> np.ndarray:
    """img: RGB float [0, 1] at any size -> 57-dim appearance descriptor."""
    x = cv2.resize(central_crop(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float64)
    feats = []
    for c in range(3):
        amp = np.abs(np.fft.fftshift(np.fft.fft2(x[..., c]))) / (SIZE * SIZE)
        feats.append(np.log(np.array([amp[ring].mean() for ring in _RING]) + 1e-8))
    flat = x.reshape(-1, 3)
    feats += [flat.mean(0), flat.std(0)]
    return np.concatenate(feats).astype(np.float32)


def descriptors(paths: list[str], load_image) -> np.ndarray:
    return np.stack([descriptor(load_image(p)) for p in paths])


class Standardiser:
    """z-score using statistics of the TRAINING pool only."""

    def __init__(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
