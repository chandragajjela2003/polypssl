"""Frequency-Augmented Learning (FAL) from FULNet, in two versions.

Both versions build, for every image a in the batch, two re-styled copies using two other images b, c
of the same batch as "donors", and stitch original + copies into three stripes (FULNet
`combine_three_images`): first third = original, middle third = 0.5*original + 0.5*copy_b,
last third = copy_c, split horizontally or vertically at random. The label is a's own mask.

faithful  : exactly what the released FULNet code does (train.py, verified 2026-10-10):
            amplitude-only swap with L = 0.001, i.e. window half-width b = floor(320*0.001) = 0, so ONLY
            the DC coefficient is blended (= a per-channel brightness shift); applied to ImageNet-NORMALISED
            images; lam = 0.5.
corrected : what the FULNet paper describes: a real low-frequency band (half-width floor(size*beta),
            default beta = 0.01 -> 3 px, a 7x7 window at 320 px) of the amplitude spectrum is blended, on
            images in [0, 1] (real colours), phase kept, result clipped to [0, 1].
"""
from __future__ import annotations

import torch


def _partners(batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Two donor indices per image, both different from the image itself and from each other."""
    if batch_size < 3:
        raise ValueError("FAL needs a batch of at least 3 images (FULNet uses 8)")
    first, second = [], []
    for i in range(batch_size):
        others = [j for j in range(batch_size) if j != i]
        perm = torch.randperm(len(others), generator=generator)[:2].tolist()
        first.append(others[perm[0]])
        second.append(others[perm[1]])
    return torch.tensor(first), torch.tensor(second)


def _amplitude_mix(src: torch.Tensor, donor: torch.Tensor, half_width: int, lam: float) -> torch.Tensor:
    """Blend the centred (2b+1)x(2b+1) low-frequency amplitude window of src with donor's; keep src phase."""
    f_src = torch.fft.fft2(src.float(), dim=(-2, -1))
    f_don = torch.fft.fft2(donor.float(), dim=(-2, -1))
    amp_src, pha_src = f_src.abs(), f_src.angle()
    amp_don = f_don.abs()
    amp_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
    amp_don = torch.fft.fftshift(amp_don, dim=(-2, -1))
    h, w = src.shape[-2:]
    ch, cw = h // 2, w // 2
    b = half_width
    win = (..., slice(ch - b, ch + b + 1), slice(cw - b, cw + b + 1))
    amp_src = amp_src.clone()
    amp_src[win] = lam * amp_don[win] + (1 - lam) * amp_src[win]
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2, -1))
    return torch.real(torch.fft.ifft2(amp_src * torch.exp(1j * pha_src), dim=(-2, -1)))


def _stripe_weights(h: int, w: int, vertical: bool, device) -> torch.Tensor:
    """FULNet's 3-stripe mask: 0 | 0.5 | 1 along height (vertical) or width (horizontal)."""
    if vertical:
        sizes = [h // 3, h // 3, h - 2 * (h // 3)]
        col = torch.cat([torch.full((s,), v) for s, v in zip(sizes, (0.0, 0.5, 1.0))])
        return col.view(h, 1).expand(h, w).to(device)
    sizes = [w // 3, w // 3, w - 2 * (w // 3)]
    row = torch.cat([torch.full((s,), v) for s, v in zip(sizes, (0.0, 0.5, 1.0))])
    return row.view(1, w).expand(h, w).to(device)


def _combine(orig: torch.Tensor, copy_b: torch.Tensor, copy_c: torch.Tensor, vertical: bool) -> torch.Tensor:
    m = _stripe_weights(orig.shape[-2], orig.shape[-1], vertical, orig.device)
    # identical to FULNet: (1-m)*orig + m*copy_b*(m==0.5) + m*copy_c*(m==1)
    return (1 - m) * orig + m * copy_b * (m == 0.5) + m * copy_c * (m == 1.0)


def fal(x: torch.Tensor, mode: str, generator: torch.Generator, beta: float = 0.01, lam: float = 0.5) -> torch.Tensor:
    """Return the FAL view of batch x [B, 3, H, W].

    mode='faithful' expects x already ImageNet-normalised (as in FULNet);
    mode='corrected' expects x in [0, 1] and returns values in [0, 1]."""
    if mode not in {"faithful", "corrected"}:
        raise ValueError(mode)
    # faithful: L = 0.001 as in FULNet train.py -> 0 at 320 px (DC term only)
    half_width = int(min(x.shape[-2:]) * (0.001 if mode == "faithful" else beta))
    idx_b, idx_c = _partners(x.shape[0], generator)
    idx_b, idx_c = idx_b.to(x.device), idx_c.to(x.device)
    copy_b = _amplitude_mix(x, x[idx_b], half_width, lam)
    copy_c = _amplitude_mix(x, x[idx_c], half_width, lam)
    out = torch.empty_like(x, dtype=torch.float32)
    for i in range(x.shape[0]):
        vertical = bool(torch.randint(0, 2, (1,), generator=generator).item())
        out[i] = _combine(x[i].float(), copy_b[i], copy_c[i], vertical)
    if mode == "corrected":
        out = out.clamp(0.0, 1.0)
    return out.to(x.dtype)
