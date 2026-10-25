"""Polyp-PVT (Dong et al., CAAI AIR 2023), ported from the official lib/pvt.py, with FULNet's UAL as an
optional plug-in ("Option A" of the project: Polyp-PVT + FULNet parts; user decision 2026-10-18).

Official network, unchanged: PVTv2-B2 backbone -> CIM (channel + spatial attention on the stride-4
feature) -> CFM (cascaded fusion of strides 8/16/32) -> SAM (graph reasoning that fuses the CFM feature
with the low-level CIM feature). Two logit maps P1 (from CFM) and P2 (from SAM), upsampled x8.
Final prediction = sigmoid(P1 + P2), exactly as in the official Test.py.

UAL insertion (ual = "corrected"): FULNet's UAL re-weights a feature with the uncertainty of a coarser
prediction:  f <- f * att(P) + f,  att = (1 - 2|sigmoid(P) - 0.5|)^2  (FULNet Eq. 5-6).
In Polyp-PVT it is applied to the fused CFM feature that SAM refines into P2, using the CFM prediction
P1 (stride 8, before upsampling):  cfm <- cfm * att(P1) + cfm.  No new parameters.
Why here and not on the low-level feature T2 (the closer analogue of FULNet's encoder skip): in Polyp-PVT,
T2 enters SAM only as a channel-softmax gate, and re-weighting it changed P2 by ~1e-6 (measured
2026-10-18 with the official ImageNet backbone), i.e. UAL would have had no effect there.
ual="off" gives the original Polyp-PVT bit-for-bit. (A "faithful" UAL does not exist for Polyp-PVT:
FULNet released no code for its Polyp-PVT variant.)

Only API changes: deprecated F.upsample -> F.interpolate with the same default mode (nearest).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fulnet import uncertainty_weight
from .pvtv2 import load_pvt_weights, pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)       # defined but not applied, as in the official code

    def forward(self, x):
        return self.bn(self.conv(x))


class CFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.relu(self.conv2(h - x))


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]))      # official F.upsample default: nearest
        n, c, h, w = x.size()
        edge = torch.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.softmax(torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1)), dim=1)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1.0 / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_state = torch.matmul(x_n_rel, x_proj_reshaped).view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class PolypPVT(nn.Module):
    def __init__(self, channel: int = 32, ual: str = "off", pretrained_path: str | None = None):
        super().__init__()
        if ual not in {"off", "corrected"}:
            raise ValueError("Polyp-PVT supports ual='off' (original) or 'corrected' (FULNet UAL plug-in)")
        self.ual = ual
        self.backbone = pvt_v2_b2()                      # [64, 128, 320, 512]
        self.load_report = load_pvt_weights(self.backbone, pretrained_path) if pretrained_path else None
        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        self.down05 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x1 = self.ca(x1) * x1                            # CIM
        cim_feature = self.sa(x1) * x1
        cfm_feature = self.CFM(self.Translayer4_1(x4), self.Translayer3_1(x3), self.Translayer2_1(x2))
        prediction1 = self.out_CFM(cfm_feature)          # coarse prediction, stride 8
        t2 = self.down05(self.Translayer2_0(cim_feature))
        sam_input = cfm_feature
        if self.ual == "corrected":                      # FULNet UAL plug-in (see module docstring)
            sam_input = cfm_feature * uncertainty_weight(prediction1) + cfm_feature
        sam_feature = self.SAM(sam_input, t2)
        prediction2 = self.out_SAM(sam_feature)
        p1 = F.interpolate(prediction1, scale_factor=8, mode="bilinear")
        p2 = F.interpolate(prediction2, scale_factor=8, mode="bilinear")
        return [p1, p2]

    @staticmethod
    def final_logits(outputs):
        return outputs[0] + outputs[1]                   # official Test.py: sigmoid(P1 + P2)
