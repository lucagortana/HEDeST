"""Minimal CellViT backend for PanoSpace detection module.

This subpackage contains:
- The `CellViT` model definition (ViT encoder + multi-branch decoder)
- The ViT backbone implementation (`ViTCellViT`)
- Basic building blocks (`Conv2DBlock`, `Deconv2DBlock`)
- The post-processing to extract instance maps (`DetectionCellPostProcessor`)

Internal modules should be imported relatively. Only public symbols are re-exposed below.
"""
from __future__ import annotations

from .backbones import ViTCellViT
from .blocks import Conv2DBlock
from .blocks import Deconv2DBlock
from .cellvit import CellViT
from .postprocessing import DetectionCellPostProcessor

__all__ = [
    "CellViT",
    "ViTCellViT",
    "Conv2DBlock",
    "Deconv2DBlock",
    "DetectionCellPostProcessor",
]
