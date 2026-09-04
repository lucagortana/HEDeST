# -*- coding: utf-8 -*-
# Core utilities for CellViT backend in PanoSpace.
#
# Includes bounding box extraction, small object removal,
# dictionary flattening/unflattening, and miscellaneous helpers.
#
# @ PanoSpace adaptation
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from scipy import ndimage


def get_bounding_box(img: np.ndarray) -> List[int]:
    """Get tight bounding box of a binary mask.

    Args:
        img (np.ndarray): 2D binary mask.

    Returns:
        List[int]: [rmin, rmax, cmin, cmax] coordinates.
    """
    rows, cols = np.any(img, axis=1), np.any(img, axis=0)
    if not rows.any() or not cols.any():
        return []
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax + 1, cmin, cmax + 1]


def remove_small_objects(label_map, min_size=64, connectivity=1):
    """Remove small connected components.

    Args:
        label_map (np.ndarray): Labeled array.
        min_size (int): Minimum object size.
        connectivity (int): Pixel connectivity.

    Returns:
        np.ndarray: Cleaned label map.
    """
    if min_size == 0:
        return label_map

    if label_map.dtype == bool:
        selem = ndimage.generate_binary_structure(label_map.ndim, connectivity)
        ccs = np.zeros_like(label_map, dtype=np.int32)
        ndimage.label(label_map, selem, output=ccs)
    else:
        ccs = label_map

    counts = np.bincount(ccs.ravel())
    mask = counts < min_size
    label_map[mask[ccs]] = 0

    return label_map


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary with a separator.

    Args:
        d (dict): Nested dict.
        parent_key (str): Used internally for recursion.
        sep (str): Key separator.

    Returns:
        dict: Flattened dict.
    """
    items = []
    for k, v in d.items():
        k = str(k)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Convert flattened dict back to nested form.

    Args:
        d (dict): Flattened dict.
        sep (str): Separator.

    Returns:
        dict: Nested dict.
    """
    result = {}
    for flat_key, value in sorted(d.items(), key=lambda x: len(x[0].split(sep))):
        keys = flat_key.split(sep)
        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return result


def get_size_of_dict(d: dict) -> int:
    """Estimate memory footprint of a dict (rough).

    Args:
        d (dict)

    Returns:
        int: Size in bytes.
    """
    size = sys.getsizeof(d)
    for k, v in d.items():
        size += sys.getsizeof(k) + sys.getsizeof(v)
    return size


def load_wsi_files_from_csv(csv_path: Union[Path, str], wsi_extension: str) -> List[str]:
    """Load WSI filenames from a CSV (expects a 'Filename' column).

    Args:
        csv_path (Path or str): Path to CSV.
        wsi_extension (str): File extension, e.g. 'svs'.

    Returns:
        List[str]: Filtered filenames.
    """
    df = pd.read_csv(csv_path)
    return [f for f in df["Filename"].tolist() if Path(f).suffix == f".{wsi_extension}"]


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
