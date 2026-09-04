# -*- coding: utf-8 -*-
# Basic convolutional blocks for CellViT U-Net decoders in PanoSpace
from __future__ import annotations

import torch.nn as nn


class Conv2DBlock(nn.Module):
    """2D Conv Block: Conv2d → BatchNorm → ReLU → Dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Default: 3.
        dropout (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """2D Deconv Block: ConvTranspose2d → Conv2d → BatchNorm → ReLU → Dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size for the Conv2d after upsampling. Default: 3.
        dropout (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
