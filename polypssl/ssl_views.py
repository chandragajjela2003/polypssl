"""Views used by semi-supervised training: the student's strong appearance perturbation and the
ZoomMatch mosaic (Stage E). Everything works on GPU batches of images in [0, 1].

strong_photometric : per-image brightness / contrast / saturation jitter, Gaussian blur and Gaussian
                     noise. Purely photometric, so the image stays pixel-aligned with its pseudo-label.

zoom_mosaic (ZoomMatch, NOVELTY.md v7 P3 / M1):
  Problem: unseen hospitals have far smaller polyps relative to the frame (ETIS: 51 % of polyps < 2 % of
  the image vs 3 % in Kvasir training). Rescaling the whole input (DEC-Seg, ScaleMatch) keeps that
  fraction unchanged; ZoomMatch changes it at a FIXED canvas size.
  For every image i chosen for zooming (probability `prob`):
    1. area estimate a_i = fraction of pixels in its (pseudo-)label (teacher prediction >= 0.5, or the
       real mask for labelled images);
    2. targeted mode: draw a target fraction f ~ log-uniform[area_min, area_max] and set the zoom factor
       s = sqrt(f / a_i), i.e. the grid size g = round(1 / s) clipped to [1, max_grid];
       fixed2 mode (control E2, MixPL "Pseudo Mosaic" style): always g = 2;
    3. the canvas (same size as the input) is a g x g grid of tiles of size H // g: image i in a random
       cell, other images of the batch in the remaining cells; any leftover border is black (like the
       endoscope frame) and labelled background;
    4. the label canvas is built with EXACTLY the same placement and the same resizing operator
       (area averaging, i.e. anti-aliased shrinking, for images, labels and weights), so image and
       (pseudo-)label stay aligned pixel for pixel.
  g = 1 means "no zoom" for that image.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def strong_photometric(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    b = x.shape[0]
    dev = x.device
    r = lambda lo, hi: (lo + (hi - lo) * torch.rand(b, 1, 1, 1, generator=gen)).to(dev)
    x = x * r(0.6, 1.4)                                            # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * r(0.6, 1.4) + mean                            # contrast
    grey = x.mean(dim=1, keepdim=True)
    x = (x - grey) * r(0.6, 1.4) + grey                            # saturation
    x = x.clamp(0, 1)
    blur = (torch.rand(b, generator=gen) < 0.5).to(dev)
    if blur.any():
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
        k = (k[:, None] * k[None, :]) / 256.0
        xb = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
        x = torch.where(blur.view(b, 1, 1, 1), xb, x)
    noise = torch.randn(x.shape, generator=gen).to(dev) * r(0.0, 0.03)
    return (x + noise).clamp(0, 1)


def _grid_sizes(area: torch.Tensor, mode: str, cfg: dict, gen: torch.Generator) -> list[int]:
    sizes = []
    for a in area.tolist():
        if torch.rand(1, generator=gen).item() >= cfg["prob"]:
            sizes.append(1)
        elif mode == "fixed2":
            sizes.append(2)
        else:                                                      # targeted
            lo, hi = math.log(cfg["area_min"]), math.log(cfg["area_max"])
            f = math.exp(lo + (hi - lo) * torch.rand(1, generator=gen).item())
            if a <= 0:                                             # nothing predicted: random moderate zoom
                g = int(torch.randint(1, 3, (1,), generator=gen).item())
            else:
                g = round(1.0 / math.sqrt(f / a)) if f < a else 1
            sizes.append(int(min(max(g, 1), cfg["max_grid"])))
    return sizes


def zoom_mosaic(images: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None,
                mode: str, cfg: dict, gen: torch.Generator):
    """images [B,3,H,W] in [0,1]; labels [B,1,H,W] (probabilities or masks); weights [B,1,H,W] or None
    (per-pixel loss weights, e.g. pseudo-label confidence). Returns (images, labels, weights, grid_sizes).
    Area for the targeted zoom is measured on `labels` (>= 0.5)."""
    b, _, h, w = images.shape
    area = (labels >= 0.5).float().mean(dim=(1, 2, 3))
    grids = _grid_sizes(area, mode, cfg, gen)
    if weights is None:
        weights = torch.ones_like(labels)
    out_x, out_y, out_w = images.clone(), labels.clone(), weights.clone()
    for i, g in enumerate(grids):
        if g == 1:
            continue
        th, tw = h // g, w // g
        canvas_x = torch.zeros_like(images[i])
        canvas_y = torch.zeros_like(labels[i])
        canvas_w = torch.ones_like(weights[i])                     # black border = certain background
        cells = g * g
        own_cell = int(torch.randint(0, cells, (1,), generator=gen).item())
        others = [j for j in range(b) if j != i] or [i]
        for c in range(cells):
            j = i if c == own_cell else others[int(torch.randint(0, len(others), (1,), generator=gen).item())]
            r0, c0 = (c // g) * th, (c % g) * tw
            canvas_x[:, r0:r0 + th, c0:c0 + tw] = F.interpolate(images[j:j + 1], size=(th, tw), mode="area")[0]
            canvas_y[:, r0:r0 + th, c0:c0 + tw] = F.interpolate(labels[j:j + 1], size=(th, tw), mode="area")[0]
            canvas_w[:, r0:r0 + th, c0:c0 + tw] = F.interpolate(weights[j:j + 1], size=(th, tw), mode="area")[0]
        out_x[i], out_y[i], out_w[i] = canvas_x, canvas_y, canvas_w
    return out_x, out_y, out_w, grids
