"""PanoSpace core backend implementations.

This package contains the actual implementation code for all high-level
functionality exposed through :mod:`panospace.tl`, including:

- Cell detection (CellViT, StarDist)
- Cell-type deconvolution (RCTD, cell2location, spatialDWLS, EnDecon)
- Super-resolution refinement
- Cell-type annotation

All backends are lazily loaded to avoid importing heavy dependencies
(e.g., PyTorch, TensorFlow) until actually needed.
"""
from __future__ import annotations
