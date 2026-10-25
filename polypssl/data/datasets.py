"""PyTorch datasets.

LabelledDataset   : training images with masks (augmented). Returns image in [0, 1] and binary mask.
UnlabelledDataset : training images WITHOUT masks (the mask file is never opened), with geometric
                    augmentation only (flip / rotate-90). Used by semi-supervised methods.
TestDataset     : resized image for the network + the ORIGINAL-resolution mask, so metrics are
                  computed at native resolution (the PSRC / "Metrics or Mirage?" convention).
"""
from __future__ import annotations

import random

import numpy as np
from torch.utils.data import Dataset

from .transforms import load_image, load_mask, resize_pair, to_tensor, train_transform


def _is_colour_mask(sample: dict) -> bool:
    return sample["dataset"] == "BKAI"


class LabelledDataset(Dataset):
    def __init__(self, samples: list[dict], img_size: int, augment: bool = True, seed: int = 0):
        self.samples = samples
        self.size = img_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0  # changed by the training loop so augmentations differ between passes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s["image"])
        mask = load_mask(s["mask"], _is_colour_mask(s))
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {s['id']}: {img.shape[:2]} vs {mask.shape[:2]}")
        if self.augment:
            # per-sample, per-pass RNG: reproducible from (seed, epoch, idx), independent of worker count
            rng = random.Random(hash((self.seed, self.epoch, idx)) & 0xFFFFFFFF)
            img, mask = train_transform(img, mask, self.size, rng)
        else:
            img, mask = resize_pair(img, mask, self.size)
        return to_tensor(img, mask)


class UnlabelledDataset(Dataset):
    def __init__(self, samples: list[dict], img_size: int, seed: int = 0):
        self.samples = samples
        self.size = img_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = load_image(self.samples[idx]["image"])               # image only: no mask is read
        # integers only: Python randomises str hashes per process, which would break reproducibility
        rng = random.Random(hash((self.seed, self.epoch, idx, 7_919)) & 0xFFFFFFFF)
        if rng.random() < 0.5:
            choice = rng.randrange(3)
            img = img[:, ::-1] if choice == 0 else img[::-1, :] if choice == 1 else np.rot90(img, rng.randrange(4))
        img, _ = resize_pair(img, np.zeros(img.shape[:2], np.uint8), self.size)
        return to_tensor(img)


class TestDataset(Dataset):
    """`transform`: optional function applied to the RESIZED image (float RGB in [0, 1]) before it is given
    to the network; used only by the C1 diagnosis to re-colour test images. Masks are never changed."""

    def __init__(self, samples: list[dict], img_size: int, transform=None):
        self.samples = samples
        self.size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s["image"])
        mask = load_mask(s["mask"], _is_colour_mask(s))
        small, _ = resize_pair(img, np.zeros(img.shape[:2], np.uint8), self.size)
        if self.transform is not None:
            small = np.clip(self.transform(small), 0.0, 1.0).astype(np.float32)
        return {"image": to_tensor(small), "mask": mask, "id": s["id"], "dataset": s["dataset"]}


def test_collate(batch: list[dict]) -> dict:
    """Images are stacked; native-resolution masks differ in size, so they stay a list."""
    import torch
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": [b["mask"] for b in batch],
        "id": [b["id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }
