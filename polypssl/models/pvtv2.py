"""PVTv2-B2 backbone, ported from Polyp-PVT's lib/pvtv2.py (itself the official PVTv2 code).

Unchanged computation; only unused parts (classification head helpers, timm registry) are dropped and
`timm.models.layers` -> `timm.layers` (the old path is deprecated). Module names are identical, so the
official ImageNet checkpoint pvt_v2_b2.pth (github.com/whai362/PVT release v2, 101,501,299 bytes) loads
directly (verified by tests/test_models.py when the file is present).
Returns 4 feature maps at strides 4, 8, 16, 32 with 64, 128, 320, 512 channels.
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


def _init_weights(m):
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


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x, h, w):
        x = self.drop(self.act(self.dwconv(self.fc1(x), h, w)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.norm(self.sr(x_).reshape(b, c, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.apply(_init_weights)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class PVTv2(nn.Module):
    def __init__(self, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4),
                 qkv_bias=True, depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for s in range(4):
            setattr(self, f"block{s + 1}", nn.ModuleList([
                Block(dim=embed_dims[s], num_heads=num_heads[s], mlp_ratio=mlp_ratios[s], qkv_bias=qkv_bias,
                      drop=drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[s])
                for i in range(depths[s])]))
            setattr(self, f"norm{s + 1}", norm_layer(embed_dims[s]))
            cur += depths[s]
        self.apply(_init_weights)

    def forward(self, x):
        b = x.shape[0]
        outs = []
        for s in range(1, 5):
            x, h, w = getattr(self, f"patch_embed{s}")(x)
            for blk in getattr(self, f"block{s}"):
                x = blk(x, h, w)
            x = getattr(self, f"norm{s}")(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


def pvt_v2_b2() -> PVTv2:
    return PVTv2()


def load_pvt_weights(model: PVTv2, path: str | Path) -> dict:
    """Polyp-PVT's loading rule (keep checkpoint keys that exist in the backbone), plus a hard check that
    EVERY backbone parameter was found in the checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state.get("model", state))
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in matched]
    if missing:
        raise RuntimeError(f"PVTv2 checkpoint is missing {len(missing)} backbone keys, e.g. {missing[:5]}")
    model.load_state_dict(matched, strict=True)
    return {"missing": missing, "unused": [k for k in state if k not in own]}
