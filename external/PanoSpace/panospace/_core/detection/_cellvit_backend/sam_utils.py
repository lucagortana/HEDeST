# -*- coding: utf-8 -*-
#
# PanoSpace-adapted SAM ViT encoder utilities
#
# @ Adapted by PanoSpace Team
from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Simple MLP block: Linear -> Activation -> Linear"""

    def __init__(self, embed_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """LayerNorm applied per channel for 2D feature maps"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings (B, C, H, W) -> (B, H', W', C')"""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # BCHW -> BHWC


class Attention(nn.Module):
    """Multi-Head Attention with optional relative positional bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "input_size must be given for relative position."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, self.head_dim).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 3, 1, 4)
        return self.proj(x.reshape(B, H, W, -1))


class MLPAttentionBlock(nn.Module):
    """Single Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    """ViT encoder with optional absolute/relative positional embedding for SAM."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            [
                MLPAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
                for i in range(depth)
            ]
        )

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, 1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


# --- Window partition utils ---
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :]


# --- Relative position utils ---
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = (
            F.interpolate(rel_pos[None, :, :].permute(0, 2, 1), size=max_rel_dist, mode="linear")
            .squeeze(0)
            .permute(1, 0)
        )
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1)
    rel_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1)
    return rel_pos[rel_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = attn.view(B, q_h, q_w, k_h, k_w) + rel_h[..., None] + rel_w[..., None, :]
    return attn.view(B, q_h * q_w, k_h * k_w)
