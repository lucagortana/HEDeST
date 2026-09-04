"""Backend implementations for cell-type annotation and deconvolution.

Supports multiple algorithms:
- RCTD: Robust Cell Type Decomposition
- cell2location: Probabilistic cell-type mapping
- spatialDWLS: Dampened Weighted Least Squares
- EnDecon: Ensemble integration
- Super-resolution refinement
- Cell-type annotation for segmented cells
"""
from __future__ import annotations
