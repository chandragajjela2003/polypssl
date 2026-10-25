"""Image loading and augmentations (OpenCV + NumPy only, so behaviour is identical on Colab, Kaggle
and locally, independent of albumentations versions).

The training augmentation reproduces FULNet's `get_train_augmentation(ver=2)` (dataset/Segmentation.py)
with albumentations' default parameter ranges:
    OneOf(HorizontalFlip, VerticalFlip, RandomRotate90)           p=0.5
    OneOf(RandomBrightnessContrast(0.2, 0.2), RandomGamma(80-120)) p=0.5
    OneOf(MotionBlur(<=5), MedianBlur(<=5), GaussianBlur(<=5),
          GaussNoise(var 5-20 on the 0-255 scale))                p=0.5
    Resize(img_size)
Images stay in [0, 1] (NOT normalised) so that the Fourier augmentation (fal.py) can work on real
colours; ImageNet normalisation is applied afterwards with `normalize()`.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str) -> np.ndarray:
    """RGB float32 in [0, 1], shape (H, W, 3)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_mask(path: str, colour_mask: bool = False) -> np.ndarray:
    """Binary uint8 mask {0, 1}, shape (H, W).

    colour_mask=True is for BKAI-IGH, whose masks are red (neoplastic) / green (non-neoplastic):
    both classes count as polyp."""
    if colour_mask:
        m = cv2.imread(path, cv2.IMREAD_COLOR)
        if m is None:
            raise IOError(f"Cannot read mask {path}")
        b, g, r = m[..., 0], m[..., 1], m[..., 2]
        return ((r > 127) | (g > 127)).astype(np.uint8)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Cannot read mask {path}")
    return (m > 127).astype(np.uint8)


# ---------------------------------------------------------------- individual augmentations

def _geometric(img, mask, rng: random.Random):
    choice = rng.randrange(3)
    if choice == 0:
        return img[:, ::-1], mask[:, ::-1]
    if choice == 1:
        return img[::-1, :], mask[::-1, :]
    k = rng.randrange(4)  # RandomRotate90: 0, 90, 180 or 270 degrees
    return np.rot90(img, k), np.rot90(mask, k)


def _photometric(img, rng: random.Random):
    if rng.random() < 0.5:  # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
        alpha = 1.0 + rng.uniform(-0.2, 0.2)
        beta = rng.uniform(-0.2, 0.2)
        return np.clip(img * alpha + beta, 0.0, 1.0)
    gamma = rng.uniform(80, 120) / 100.0  # RandomGamma(gamma_limit=(80, 120))
    return np.power(np.clip(img, 0.0, 1.0), gamma)


def _motion_kernel(k: int, rng: random.Random) -> np.ndarray:
    kernel = np.zeros((k, k), np.float32)
    x1, y1, x2, y2 = (rng.randrange(k) for _ in range(4))
    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)
    if kernel.sum() == 0:
        kernel[k // 2, k // 2] = 1.0
    return kernel / kernel.sum()


def _blur_or_noise(img, rng: random.Random):
    choice = rng.randrange(4)
    k = rng.choice([3, 5])
    if choice == 0:
        return cv2.filter2D(img, -1, _motion_kernel(k, rng))
    if choice == 1:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.medianBlur(u8, k).astype(np.float32) / 255.0
    if choice == 2:
        return cv2.GaussianBlur(img, (k, k), 0)
    var = rng.uniform(5.0, 20.0)  # GaussNoise(var_limit=(5, 20)) on the 0-255 scale
    noise = np.random.default_rng(rng.randrange(2**31)).normal(0.0, np.sqrt(var) / 255.0, img.shape)
    return np.clip(img + noise.astype(np.float32), 0.0, 1.0)


# ---------------------------------------------------------------- pipelines

def train_transform(img: np.ndarray, mask: np.ndarray, size: int, rng: random.Random):
    if rng.random() < 0.5:
        img, mask = _geometric(img, mask, rng)
    if rng.random() < 0.5:
        img = _photometric(img, rng)
    if rng.random() < 0.5:
        img = _blur_or_noise(np.ascontiguousarray(img), rng)
    return resize_pair(img, mask, size)


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int):
    img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(np.ascontiguousarray(mask), (size, size), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32), mask.astype(np.uint8)


def to_tensor(img: np.ndarray, mask: np.ndarray | None = None):
    t_img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    if mask is None:
        return t_img
    return t_img, torch.from_numpy(np.ascontiguousarray(mask))[None].float()


def normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation of a [B, 3, H, W] batch in [0, 1] (applied on the GPU after FAL)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
