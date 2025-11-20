from __future__ import annotations

import json
import os
from typing import Dict
from typing import Optional
from typing import Tuple
import numpy as np
import openslide
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from hedest.config import TqdmToLogger

tqdm_out = TqdmToLogger(logger, level="INFO")


def extract_images_hn(
    image_path: str,
    json_path: str,
    level: int = 0,
    size: Tuple[int, int] = (64, 64),
    dict_types: Optional[Dict[int, str]] = None,
    save_images: Optional[str] = None,
    save_dict: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts tiles from a whole slide image (WSI) given a JSON file with cell centroids.

    Args:
        image_path: Path to the WSI file.
        json_path: Path to the JSON file with cell centroids.
        level: Level of the WSI to extract the tiles.
        size: Size of the tiles (width, height).
        dict_types: Optional dictionary mapping cell types to names.
        save_images: Path to save extracted image tiles.
        save_dict: Path to save the dictionary of extracted tiles.
        
    Returns:
        Dictionary containing extracted tiles as tensors.
    """

    slide = openslide.open_slide(image_path)
    centroid_list_wsi = []
    type_list_wsi = []
    image_dict = {}

    # Extract nuclear info
    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data["nuc"]
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = inst_info["centroid"]
            centroid_list_wsi.append(inst_centroid)
            if dict_types is not None:
                inst_type = inst_info["type"]
                type_list_wsi.append(inst_type)

    cell_table = pd.DataFrame(centroid_list_wsi, columns=["x", "y"])
    if dict_types is not None:
        cell_table["class"] = type_list_wsi

    for i in tqdm(range(len(cell_table)), file=tqdm_out, desc="Extracting tiles"):
        cell_line = cell_table[cell_table.index == i]
        img_cell = slide.read_region(
            (int(cell_line["x"].values[0]) - size[0] // 2, int(cell_line["y"].values[0]) - size[1] // 2), level, size
        )
        img_cell = img_cell.convert("RGB")

        if save_images is not None:
            if dict_types is not None:
                cell_class = dict_types[cell_line["class"].values[0]]
                save_dir = os.path.join(save_images, cell_class)
            else:
                save_dir = save_images

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img_cell.save(os.path.join(save_dir, f"cell{i}.jpg"))

        img_tensor = torch.tensor(np.array(img_cell)).permute(2, 0, 1)
        image_dict[str(i)] = img_tensor

    if save_images is not None:
        logger.info("-> Tile images saved.")

    if save_dict is not None:
        torch.save(image_dict, save_dict)
        logger.info("-> image_dict saved.")

    return image_dict