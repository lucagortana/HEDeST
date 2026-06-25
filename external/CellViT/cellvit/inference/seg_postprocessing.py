# -*- coding: utf-8 -*-
# Segmentation post-processing for CellViT
#
# Makes CellViT export the same per-cell artefacts as the (adapted) HoVer-Net
# pipeline, so the two segmentation backends can be used interchangeably:
#   * a ``<wsi>.json`` file with the HoVer-Net ``{"mag": ..., "nuc": {...}}``
#     shape, with the cell ids re-ordered from 0 to n_cells-1,
#   * an optional QuPath-compatible GeoJSON file (one feature per cell),
#   * an optional dictionary of cell crops extracted around each nucleus
#     centroid, with user-chosen pixel size / micron size / mpp,
#   * an optional filtering of cells based on their distance to the closest
#     spatial-transcriptomics spot (e.g. Visium v2), given an AnnData file.
#
# The logic mirrors ``external/hovernet/seg_postprocessing.py``.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PanNuke label / colour definitions (identical to the HoVer-Net type_info.json
# and to the HoVer-Net seg_postprocessing.TYPE_COLORS). CellViT and HoVer-Net
# both use the canonical PanNuke ordering, so the integer type ids match
# 1-to-1:
#   0: background / unlabelled   1: neoplastic        2: inflammatory
#   3: connective                4: dead / necrosis   5: non-neoplastic epithelial
# ---------------------------------------------------------------------------
PANNUKE_TYPE_INFO = {
    0: ["nolabe", [0, 0, 0]],
    1: ["neopla", [255, 0, 0]],
    2: ["inflam", [0, 255, 0]],
    3: ["connec", [0, 0, 255]],
    4: ["necros", [255, 255, 0]],
    5: ["no-neo", [255, 165, 0]],
}

# QuPath ``colorRGB`` packed integers, matching the HoVer-Net geojson exporter.
TYPE_COLORS = {
    0: -1,  # white (unknown)
    1: -3670016,  # red
    2: -16711936,  # green
    3: -16776961,  # blue
    4: -16711681,  # cyan
    5: -65536,  # magenta
}

# Keys kept in the HoVer-Net ``nuc`` entries, in HoVer-Net order.
NUC_KEYS = ("bbox", "centroid", "contour", "type_prob", "type")


def _to_builtin(value):
    """Convert numpy scalars/arrays to plain python so ``json`` can serialise."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def cells_to_nuc_dict(cells: List[dict]) -> Dict[str, dict]:
    """Convert a CellViT ``cells`` list into a HoVer-Net style ``nuc`` dict.

    Only the HoVer-Net keys (bbox, centroid, contour, type_prob, type) are
    kept and the cell ids are re-indexed from 0 to ``len(cells) - 1`` in the
    list order, exactly like the adapted HoVer-Net ``run_infer.py`` does.

    Args:
        cells: List of CellViT cell dictionaries (the ``"cells"`` entry of
            ``cells.json``). Each must expose at least the keys in
            :data:`NUC_KEYS`.

    Returns:
        Ordered ``{"0": {...}, "1": {...}, ...}`` dictionary.
    """
    nuc: Dict[str, dict] = {}
    for i, cell in enumerate(cells):
        nuc[str(i)] = {
            "bbox": _to_builtin(cell["bbox"]),
            "centroid": _to_builtin(cell["centroid"]),
            "contour": _to_builtin(cell["contour"]),
            "type_prob": _to_builtin(cell.get("type_prob")),
            "type": int(cell["type"]),
        }
    return nuc


def reindex_nuc(nuc: Dict[str, dict]) -> Dict[str, dict]:
    """Re-key a ``nuc`` dict so the ids range from 0 to n_cells-1 (order kept)."""
    return {str(i): value for i, value in enumerate(nuc.values())}


def filter_by_st_proximity(
    nuc: Dict[str, dict],
    adata_path: str,
    mpp: float,
    dist_thresh_um: float = 300,
    logger=None,
) -> Dict[str, dict]:
    """Remove cells further than a threshold from the nearest ST spot.

    Ported from the adapted HoVer-Net ``seg_postprocessing.filter_by_st_proximity``.
    Cells whose centroid is further than ``dist_thresh_um`` micrometers from the
    closest spatial-transcriptomics spot (read from ``adata.obsm['spatial']``,
    e.g. Visium v2) are discarded.

    Args:
        nuc: ``nuc`` dictionary (as returned by :func:`cells_to_nuc_dict`).
        adata_path: Path to an AnnData ``.h5ad`` file with spot coordinates in
            ``obsm['spatial']`` (in the same level-0 pixel space as the centroids).
        mpp: Microns per pixel, used to convert the micron threshold to pixels.
        dist_thresh_um: Distance threshold in micrometers. Defaults to 300.
        logger: Optional logger with an ``info`` method.

    Returns:
        Filtered ``nuc`` dictionary (ids preserved; re-index afterwards).
    """
    import anndata as ad
    from scipy.spatial import KDTree

    if logger is not None:
        logger.info(f"Loading AnnData from {adata_path} for spatial filtering...")
    adata = ad.read_h5ad(adata_path)

    # Extract spot coordinates from obsm['spatial']
    spot_coords = np.array(adata.obsm["spatial"]).astype("int64")
    tree = KDTree(spot_coords)

    # Convert microns to pixels: threshold_px = microns / microns_per_pixel
    dist_thresh_px = dist_thresh_um / mpp

    filtered_nuc: Dict[str, dict] = {}
    original_count = len(nuc)

    for nuc_id, nuc_info in nuc.items():
        # Centroid orientation must match adata.obsm['spatial'] ([x, y] here).
        centroid = np.array(nuc_info["centroid"])
        dist, _ = tree.query(centroid)
        if dist <= dist_thresh_px:
            filtered_nuc[nuc_id] = nuc_info

    if logger is not None:
        logger.info(
            f"ST Filtering: Kept {len(filtered_nuc)}/{original_count} cells "
            f"(Threshold: {dist_thresh_um}um / {dist_thresh_px:.2f}px)"
        )
    return filtered_nuc


def save_nuc_json(
    nuc: Dict[str, dict],
    output_json_path: Union[str, Path],
    mag: Optional[float] = None,
) -> str:
    """Write a HoVer-Net compatible ``{"mag": ..., "nuc": {...}}`` json file.

    Args:
        nuc: ``nuc`` dictionary as returned by :func:`cells_to_nuc_dict`.
        output_json_path: Destination path.
        mag: Magnification stored under the top-level ``"mag"`` key (HoVer-Net
            protocol). May be ``None``.

    Returns:
        The path that was written (as a string).
    """
    json_dict = {"mag": _to_builtin(mag), "nuc": nuc}
    with open(output_json_path, "w") as handle:
        json.dump(json_dict, handle, indent=2)
    return str(output_json_path)


def nuc_to_geojson(
    nuc: Dict[str, dict],
    geojson_output_path: Union[str, Path],
) -> int:
    """Export a ``nuc`` dict as a QuPath-compatible GeoJSON file.

    One ``Feature`` (Polygon) is written per cell, with the same property
    schema as the HoVer-Net ``hovernet_to_geojson`` exporter.

    Args:
        nuc: ``nuc`` dictionary as returned by :func:`cells_to_nuc_dict`.
        geojson_output_path: Destination path.

    Returns:
        Number of exported features.
    """
    features = []
    skipped = 0

    for cell_id, cell_info in nuc.items():
        contour = cell_info.get("contour", [])
        if len(contour) < 3:
            skipped += 1
            continue

        coords = [[float(p[0]), float(p[1])] for p in contour]
        poly = Polygon(coords)

        if not poly.is_valid:
            poly = make_valid(poly)

        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        elif poly.geom_type != "Polygon":
            poly = poly.convex_hull

        cell_type = cell_info.get("type", 0)

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)],
                },
                "properties": {
                    "object_type": "detection",
                    "classification": {
                        "name": f"Type_{cell_type}",
                        "colorRGB": TYPE_COLORS.get(cell_type, -1),
                    },
                    "isLocked": False,
                    "cell_id": str(cell_id),
                    "cell_type": cell_type,
                    "type_prob": cell_info.get("type_prob", None),
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    with open(geojson_output_path, "w") as f:
        json.dump(geojson, f)

    return len(features)


def extract_cell_images(
    slide_path: Union[str, Path],
    nuc: Dict[str, dict],
    level: int = 0,
    size_px: int = 64,
    size_um: Optional[float] = None,
    mpp: Optional[float] = None,
    save_dict: Optional[str] = None,
    logger=None,
) -> Dict[str, torch.Tensor]:
    """Extract one crop per nucleus centroid from a WSI.

    Mirrors the adapted HoVer-Net ``extract_images_hn``: a ``size_um`` x
    ``size_um`` micron window (converted to pixels through ``mpp``) is cropped
    around every centroid and resized to ``size_px`` x ``size_px``. If
    ``size_um`` is not provided, a ``size_px`` x ``size_px`` window is cropped
    directly (no resize).

    The returned/saved dictionary maps ``str(cell_id)`` -> ``uint8`` tensor of
    shape ``(C, H, W)``. The cell ids match the (re-indexed) json, so the
    image_dict and the json stay aligned.

    Args:
        slide_path: Path to the WSI (any OpenSlide-readable format).
        nuc: ``nuc`` dictionary as returned by :func:`cells_to_nuc_dict`.
        level: WSI level to read from (centroids are in level-0 coordinates).
        size_px: Output crop size in pixels.
        size_um: Crop size in micrometers (requires ``mpp``).
        mpp: Microns per pixel of the WSI at ``level`` (level-0 resolution).
        save_dict: If given, the image_dict is ``torch.save``-d to this path.
        logger: Optional logger with an ``info`` method.

    Returns:
        Dictionary of extracted crops as uint8 tensors.
    """
    import openslide

    slide = openslide.open_slide(str(slide_path))

    if size_um is not None:
        if mpp is None:
            raise ValueError(
                "If size_um is provided, `mpp` must also be provided "
                "(pass --mpp or make sure the WSI exposes its resolution)."
            )
        crop_px = int(round(size_um / mpp))
    else:
        crop_px = size_px

    image_dict: Dict[str, torch.Tensor] = {}

    iterator = tqdm(nuc.items(), total=len(nuc), desc="Extracting cell crops")
    for cell_id, cell_info in iterator:
        centroid = cell_info["centroid"]
        x = int(round(float(centroid[0])))
        y = int(round(float(centroid[1])))

        img_cell = slide.read_region(
            (x - crop_px // 2, y - crop_px // 2),
            level,
            (crop_px, crop_px),
        )
        img_cell = img_cell.convert("RGB")

        if size_um is not None and crop_px != size_px:
            img_cell = img_cell.resize((size_px, size_px))

        img_tensor = torch.tensor(np.array(img_cell)).permute(2, 0, 1)
        image_dict[str(cell_id)] = img_tensor

    if save_dict is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_dict)), exist_ok=True)
        torch.save(image_dict, save_dict)
        if logger is not None:
            logger.info(f"-> image_dict saved to {save_dict}")

    return image_dict
