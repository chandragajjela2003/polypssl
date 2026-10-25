"""MSCAN encoder (SegNeXt), ported from FULNet's model/SegNeXt_Encoder.py WITHOUT mmcv.

Changes from the original, none of which alter the computation:
- mmcv BaseModule -> nn.Module; build_norm_layer(BN) -> nn.BatchNorm2d; mmcv DropPath -> timm DropPath.
  (FULNet passes norm_cfg=BN everywhere: MSCAN's own default overrides the SyncBN defaults of the
  sub-modules, so plain BatchNorm2d is the exact equivalent on one GPU.)
- Module and parameter NAMES are kept identical so the released checkpoint pretrained/mscan_t.pth
  loads without key remapping (verified by tests/test_models.py when the weights are present).
- The stage-1 stem also returns its intermediate GELU output (stride 2), as FULNet's get_blocks() does;
  FULNet's decoder uses that as its highest-resolution skip feature.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from timm.layers import DropPath

MSCAN_SHAPES = {
    # name: (embed_dims, depths, drop_path_rate)  -- from FULNet get_mscan_model_shape
    "mscan_t": ([32, 64, 160, 256], [3, 3, 5, 2], 0.1),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        gelu_out = None
        for i, layer in enumerate(self.proj):
            x = layer(x)
            if i == 2:
                gelu_out = x
        _, _, h, w = x.shape
        return gelu_out, x.flatten(2).transpose(1, 2), h, w


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2)
        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x.view(b, c, n).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)
        return x.flatten(2).transpose(1, 2), h, w


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MSCAN(nn.Module):
    def __init__(self, embed_dims, depths, mlp_ratios=(8, 8, 4, 4), drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.depths = depths
        self.num_stages = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                                         drop_path=dpr[cur + j]) for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        """Returns 5 feature maps (FULNet get_blocks): strides 2, 4, 8, 16, 32."""
        b = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            if i == 0:
                gelu_out, x, h, w = patch_embed(x)
                outs.append(gelu_out)
            else:
                x, h, w = patch_embed(x)
            for blk in getattr(self, f"block{i + 1}"):
                x = blk(x, h, w)
            x = getattr(self, f"norm{i + 1}")(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


def build_mscan(name: str = "mscan_t") -> MSCAN:
    dims, depths, dpr = MSCAN_SHAPES[name]
    return MSCAN(embed_dims=dims, depths=depths, drop_path_rate=dpr)


def load_mscan_weights(model: MSCAN, path: str | Path) -> dict:
    """Load the released SegNeXt ImageNet checkpoint. Returns a report of missing/unexpected keys.
    Raises if any ENCODER parameter is missing (the classification head keys are allowed to be extra)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state.items():
        for prefix in ("module.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    result = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"MSCAN checkpoint is missing {len(missing)} encoder keys, e.g. {missing[:5]}")
    return {"missing": missing, "unexpected": list(result.unexpected_keys)}
