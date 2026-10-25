"""Model factory. Every model returns a LIST of logit maps (training outputs); `final_logits(model, outputs)`
gives the map used for prediction and evaluation:
    FULNet-C : 5 deep-supervision outputs (full, 1/2, 1/4, 1/8, 1/16); final = full-resolution output
    Polyp-PVT: [P1, P2] at full resolution; final = P1 + P2 (official Test.py)
    UNet-R34 : one output; final = that output
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .fulnet import FULNetC
from .polyp_pvt import PolypPVT

WEIGHT_FILES = {"fulnet_c": "mscan_t.pth", "polyp_pvt": "pvt_v2_b2.pth"}


class UNetR34(nn.Module):
    """UNet with an ImageNet ResNet-34 encoder (segmentation_models_pytorch), the backbone used by RD-Net."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None,
                            in_channels=3, classes=1)

    def forward(self, x):
        return [self.net(x)]


def final_logits(model: nn.Module, outputs: list):
    fn = getattr(model, "final_logits", None)
    return fn(outputs) if fn is not None else outputs[0]


def _weights(m: dict, weights_dir) -> str | None:
    if not m.get("pretrained", True):
        return None
    path = Path(weights_dir or ".") / WEIGHT_FILES[m["name"]]
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {path} (setup_env.py downloads them)")
    return str(path)


def build_model(cfg: dict, weights_dir: str | Path | None = None) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "fulnet_c":
        return FULNetC(encoder=m["encoder"], ual=m["ual"], pretrained_path=_weights(m, weights_dir))
    if m["name"] == "polyp_pvt":
        return PolypPVT(ual=m["ual"], pretrained_path=_weights(m, weights_dir))
    if m["name"] == "unet_r34":
        return UNetR34(pretrained=m.get("pretrained", True))
    raise ValueError(m["name"])
