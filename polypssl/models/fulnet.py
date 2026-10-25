"""FULNet decoder and the full FULNet-C model, ported from FULNet's model/Segmentation_Decoder.py.

The decoder is kept layer-for-layer (shifted-MLP blocks, conv/BN/upsample stages, 5 deep-supervision
heads). Two changes:

1. Outputs are LOGITS (the original applies sigmoid inside the model). The loss applies sigmoid in
   float32, which is numerically identical but safe under mixed precision.
2. The UAL attention ("AFM") has three modes:
   faithful  : as released. AFM(x_skip, pred) is fed the multi-channel ReLU decoder FEATURE map out_k,
               so sigmoid(feature) >= 0.5 and the weight (1 - 2|sigmoid - 0.5|)^2 simply decays with
               feature magnitude (not uncertainty).
   corrected : as the paper describes. The weight is computed from the 1-channel PREDICTION of the
               decoder stage (its deep-supervision head, final_k(out_k)), so it is high where the
               prediction is near 0.5, i.e. uncertain. Broadcast over the skip feature's channels.
   off       : no attention; the skip feature is added unchanged (ablation).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

from .mscan import build_mscan, load_mscan_weights


class DWConvTokens(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.dwconv(x).flatten(2).transpose(1, 2)


def _init(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class ShiftMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0.0, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConvTokens(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(_init)

    def _shift(self, x, h, w, dim):
        b, n, c = x.shape
        xn = x.transpose(1, 2).view(b, c, h, w).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, h)
        x_s = torch.narrow(x_cat, 3, self.pad, w)
        return x_s.reshape(b, c, h * w).contiguous().transpose(1, 2)

    def forward(self, x, h, w):
        x = self.fc1(self._shift(x, h, w, dim=2))
        x = self.drop(self.act(self.dwconv(x, h, w)))
        x = self.fc2(self._shift(x, h, w, dim=3))
        return self.drop(x)


class ShiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ShiftMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.apply(_init)

    def forward(self, x, h, w):
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


def uncertainty_weight(signal: torch.Tensor) -> torch.Tensor:
    """FULNet Eq. 5-6: att = 4*d^2 - 4*d + 1 = (1 - 2d)^2 with d = |sigmoid(signal) - 0.5|."""
    d = torch.abs(torch.sigmoid(signal) - 0.5)
    return 4 * d ** 2 - 4 * d + 1


class FULNetDecoder(nn.Module):
    """Expects MSCAN-T features [x2 (16 ch), x4 (32), x8 (64), x16 (160), x32 (256)]."""

    def __init__(self, ual: str = "corrected", num_classes: int = 1):
        super().__init__()
        assert ual in {"corrected", "faithful", "off"}
        self.ual = ual
        self.dnorm3 = nn.LayerNorm(160)
        self.dnorm4 = nn.LayerNorm(64)
        self.dblock1 = nn.ModuleList([ShiftedBlock(dim=160, mlp_ratio=1)])
        self.dblock2 = nn.ModuleList([ShiftedBlock(dim=64, mlp_ratio=1)])
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 64, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(64)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(16, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(160, num_classes, kernel_size=1)

    def _fuse(self, skip: torch.Tensor, dec: torch.Tensor, head: nn.Module):
        """Returns (fused feature, side logits). FULNet: out = dec + AFM(skip, signal)."""
        side = head(dec)
        if self.ual == "off":
            att_skip = skip
        else:
            signal = dec if self.ual == "faithful" else side
            att_skip = skip * uncertainty_weight(signal) + skip
        return dec + att_skip, side

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, feats):
        x2, x4, x8, x16, x32 = feats
        # stage 4
        out4 = F.relu(self._up(self.dbn1(self.decoder1(x32))))
        out, side4 = self._fuse(x16, out4, self.final4)
        b, _, h, w = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, h, w)
        # stage 3
        out = self.dnorm3(out).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out3 = F.relu(self._up(self.dbn2(self.decoder2(out))))
        out, side3 = self._fuse(x8, out3, self.final3)
        _, _, h, w = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, h, w)
        # stage 2
        out = self.dnorm4(out).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out2 = F.relu(self._up(self.dbn3(self.decoder3(out))))
        out, side2 = self._fuse(x4, out2, self.final2)
        # stage 1
        out1 = F.relu(self._up(self.dbn4(self.decoder4(out))))
        out, side1 = self._fuse(x2, out1, self.final1)
        out = F.relu(self._up(self.decoder5(out)))
        # full resolution first, then 1/2, 1/4, 1/8, 1/16 (deep supervision order of FULNet)
        return [self.final(out), side1, side2, side3, side4]


class FULNetC(nn.Module):
    def __init__(self, encoder: str = "mscan_t", ual: str = "corrected", pretrained_path: str | None = None):
        super().__init__()
        if encoder != "mscan_t":
            raise NotImplementedError("Only the MSCAN-T encoder is implemented in Stage A "
                                      "(the PVTv2-B2 design is pending a decision; see NOVELTY.md).")
        self.encoder = build_mscan(encoder)
        self.load_report = load_mscan_weights(self.encoder, pretrained_path) if pretrained_path else None
        self.decoder = FULNetDecoder(ual=ual)

    def forward(self, x):
        return self.decoder(self.encoder(x))
