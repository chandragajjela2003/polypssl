"""FULNet segmentation loss with deep supervision, in two versions.

FULNet's loss per output = "adaptive pixel intensity" (ARI) loss + LogitMarginL1, summed over 5
deep-supervision outputs with the ground truth halved in size (bilinear, align_corners=True) per level.

faithful  : exact copy of the released code (loss/segmentation.py, loss/logit_margin_l1.py):
            - F.binary_cross_entropy(..., reduce=None) and F.l1_loss(..., reduce=None) keep the default
              reduction='mean', so the pixel weights omega multiply a SCALAR (no pixel weighting of BCE/MAE);
            - LogitMarginL1 on a 1-channel output is identically zero (max over 1 channel minus itself = 0).
corrected : BCE and MAE are genuinely pixel-weighted (reduction='none'); the always-zero LogitMarginL1
            term is dropped. IoU and Dice terms are identical in both versions.

Model outputs are LOGITS; probabilities are computed here in float32 (binary cross-entropy is not
safe under mixed precision).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _omega(mask: torch.Tensor) -> torch.Tensor:
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    return 1 + 0.5 * (w1 + w2 + w3) * mask


def ari_loss(prob: torch.Tensor, mask: torch.Tensor, mode: str, return_parts: bool = False):
    omega = _omega(mask)
    if mode == "faithful":
        bce = F.binary_cross_entropy(prob, mask)          # scalar (the released-code behaviour)
        mae = F.l1_loss(prob, mask)                        # scalar
    else:
        bce = F.binary_cross_entropy(prob, mask, reduction="none")
        mae = torch.abs(prob - mask)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))
    inter = ((prob * mask) * omega).sum(dim=(2, 3))
    union = ((prob + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)
    adice = 1 - (2 * inter + 1) / (union + 1)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))
    if return_parts:
        return {"bce": abce.mean(), "iou": aiou.mean(), "dice": adice.mean(), "mae": amae.mean()}
    return (0.7 * abce + 0.7 * aiou + 0.7 * adice + 0.7 * amae).mean()


def logit_margin_l1_faithful(prob: torch.Tensor, margin: float = 10.0) -> torch.Tensor:
    """Exact LogitMarginL1 of the released code. With one output channel it is always 0; kept only so
    the faithful mode is a literal reproduction (and the unit test can prove it is zero)."""
    x = prob.view(prob.size(0), prob.size(1), -1).transpose(1, 2).reshape(-1, prob.size(1))
    diff = x.max(dim=1, keepdim=True).values - x
    return F.relu(diff - margin).mean()


def structure_loss(logits: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Polyp-PVT / PraNet 'structure loss' (weighted BCE + weighted IoU) on LOGITS.

    faithful : as released, F.binary_cross_entropy_with_logits(..., reduce='none'). The legacy argument
               reduce='none' is a truthy string, so PyTorch uses reduction='mean': the BCE is a scalar and
               the boundary weights `weit` do not weight it pixel-wise (same issue as FULNet's loss).
    corrected: genuinely pixel-weighted BCE (reduction='none').
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    if mode == "faithful":
        wbce = F.binary_cross_entropy_with_logits(logits, mask)
    else:
        wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def segmentation_loss(logits: list[torch.Tensor], mask: torch.Tensor, loss_cfg: dict) -> torch.Tensor:
    """Dispatch on loss.type: 'ari' (FULNet deep supervision) or 'structure' (Polyp-PVT, summed over outputs)."""
    if loss_cfg.get("type", "ari") == "ari":
        return deep_supervision_loss(logits, mask, loss_cfg["mode"])
    with torch.autocast(device_type=mask.device.type, enabled=False):
        total = torch.zeros((), device=mask.device)
        for lg in logits:
            gt = mask.float() if lg.shape[-2:] == mask.shape[-2:] else \
                F.interpolate(mask.float(), size=lg.shape[-2:], mode="bilinear", align_corners=True)
            total = total + structure_loss(lg.float(), gt, loss_cfg["mode"])
        return total


def deep_supervision_loss(logits: list[torch.Tensor], mask: torch.Tensor, mode: str) -> torch.Tensor:
    """logits: model outputs, full resolution first (FULNet: 5 outputs at 1, 1/2, 1/4, 1/8, 1/16).
    A model with a single output (e.g. UNet) gets the same loss on that output only."""
    with torch.autocast(device_type=mask.device.type, enabled=False):
        gt = mask.float()
        total = torch.zeros((), device=mask.device)
        for level, lg in enumerate(logits):
            if level > 0:
                gt = F.interpolate(gt, scale_factor=0.5, mode="bilinear", align_corners=True)
            if lg.shape[-2:] != gt.shape[-2:]:
                raise ValueError(f"Output {level} has size {tuple(lg.shape[-2:])}, target {tuple(gt.shape[-2:])}")
            prob = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
            total = total + ari_loss(prob, gt, mode)
            if mode == "faithful":
                total = total + logit_margin_l1_faithful(prob)
        return total
