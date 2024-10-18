from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
from numpy import ndarray
from openslide import OpenSlide
from scipy.spatial import KDTree
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.morphology import opening
from skimage.morphology import square
from skimage.transform import resize
from tqdm import tqdm


def make_auto_mask(slide: OpenSlide, mask_level: int, save: Optional[str] = None) -> ndarray:
    """make_auto_mask. Create a binary mask from a downsampled version
    of a WSI. Uses the Otsu algorithm and a morphological opening.

    :param slide: WSI. Accepted extension *.tiff, *.svs, *ndpi.
    :param mask_level: level of the pyramidal WSI used to create the mask.
    :return mask: ndarray. Binary mask of the WSI. Dimensions are the one of the
    dowsampled image.
    """

    assert mask_level >= 0, "mask_level must be a positive integer"
    slide = openslide.open_slide(slide) if isinstance(slide, str) else slide

    im = slide.read_region((0, 0), 0, slide.level_dimensions[0])
    im = np.array(im)[:, :, :3]

    desired_dims = (
        slide.level_dimensions[0][1] // (2**mask_level),
        slide.level_dimensions[0][0] // (2**mask_level),
    )
    im = resize(im, desired_dims, anti_aliasing=True)

    im_gray = rgb2gray(im)
    im_gray = clear_border(im_gray, prop=30)
    size = im_gray.shape
    im_gray = im_gray.flatten()
    pixels_int = im_gray[np.logical_and(im_gray > 0.1, im_gray < 0.98)]
    t = threshold_otsu(pixels_int)
    mask = opening(
        closing(np.logical_and(im_gray < t, im_gray > 0.1).reshape(size), footprint=square(32)), footprint=square(32)
    )
    mask = (np.stack([mask] * 3, axis=-1).astype(np.uint8)) * 255
    if save is not None:
        cv2.imwrite(save, mask)
    return mask


def clear_border(mask: ndarray, prop: int):
    r, c = mask.shape
    pr, pc = r // prop, c // prop
    mask[:pr, :] = 0
    mask[r - pr :, :] = 0
    mask[:, :pc] = 0
    mask[:, c - pc :] = 0
    return mask


def get_x_y(slide: OpenSlide, point_l: Tuple[int, int], level: int, integer: bool = True):
    """
    Code @PeterNaylor from useful_wsi.
    Given a point point_l = (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point point_0 = (x_0, y_0).
    Args:
        slide : Openslide object from which we extract.
        point_l : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level of the associated point.
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_0.
    """
    x_l, y_l = point_l
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])

    x_0 = x_l * size_x_0 / size_x_l
    y_0 = y_l * size_y_0 / size_y_l
    if integer:
        point_0 = (int(x_0), int(y_0))
    else:
        point_0 = (x_0, y_0)
    return point_0


def extract_tiles_hovernet(
    image_path, json_path, level=0, size=(64, 64), dict_types=None, save_images=None, save_dict=None
):  # must adapt for segmentation without classification
    """
    Extracts tiles from a WSI given a json file with the cell centroids.
    Args:
        image_path: Path to the WSI.
        json_path: Path to the json file with the cell centroids.
        level: Level of the WSI to extract the tiles.
        size: Size of the tiles.
        dict_types: Dictionary with the cell types.
        save_images: Path to save the images.
        save_dict: Path to save the dictionary.
    Returns:
        image_dict: Dictionary with the extracted tiles.
    """

    slide = openslide.open_slide(image_path)
    centroid_list_wsi = []
    type_list_wsi = []
    image_dict = {}

    # add results to individual lists
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

    for i in tqdm(range(len(cell_table))):
        cell_line = cell_table[cell_table.index == i]
        img_cell = slide.read_region(
            (int(cell_line["x"].values[0]) - size[0] // 2, int(cell_line["y"].values[0]) - size[1] // 2), level, size
        )
        img_cell = img_cell.convert("RGB")

        if save_images is not None:
            if dict_types is not None:
                cell_class = dict_types[cell_line["class"].values[0]]
                save_dir = save_images + cell_class + "/"
            else:
                save_dir = save_images

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img_cell.save(save_dir + f"cell{i}.jpg")

        img_tensor = torch.tensor(np.array(img_cell)).permute(2, 0, 1)  # Convert image to tensor (C, H, W)
        image_dict[str(i)] = img_tensor

    if save_images is not None:
        print("-> Tile images saved.")

    if save_dict is not None:
        torch.save(image_dict, save_dict)
        print("-> image_dict saved.")

    return image_dict


def map_cells_to_spots(adata, adata_name, json_path):
    """
    Maps cells to spots based on the centroids of the cells and spots.
    Args:
        adata: Anndata object containing the Visium dataset.
        adata_name: Name of the dataset in the adata object.
        json_path: Path to the json file with the cell centroids.
    Returns:
        dict_cells_spots: Dictionary with the mapping of cells to spots.
    Examples:
        dict_cells_spots = map_cells_to_spots(adata, adata_name, json_path)
    """
    centroid_list = []
    _, _, spots_coordinates, diameter = get_visium_infos(adata, adata_name)
    spots_ids = adata.obs.index

    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data["nuc"]
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = inst_info["centroid"]
            centroid_list.append(inst_centroid)

    # Convert to numpy array for KDTree
    centroid_array = np.array(centroid_list)
    spots_array = np.array(spots_coordinates)

    # Create a KDTree for the spots
    tree = KDTree(spots_array)

    # Create a dictionary to hold the mapping
    dict_cells_spots = defaultdict(list)

    # Query the KDTree to find the nearest spot for each cell within the diameter
    for i, cell in enumerate(tqdm(centroid_array, desc="Mapping cells to spots")):
        indices = tree.query_ball_point(cell, r=diameter / 2)
        for idx in indices:
            dict_cells_spots[spots_ids[idx]].append(str(i))

    return dict_cells_spots


def get_visium_infos(adata, adata_name):

    centers = adata.obsm["spatial"].astype("int64")
    diameter = adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"]

    mpp = 55 / diameter
    mag = get_mag(mpp)

    return mag, mpp, centers, diameter


def get_xenium_infos():
    mpp = 0.2125
    # from https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    mag = get_mag(mpp)
    return mag, mpp


def get_mag(mpp):
    """Returns the magnification of the image based on the mpp.

    from HEST
    """

    if mpp <= 0.1:
        mag = 60
    elif 0.1 < mpp and mpp <= 0.25:
        mag = 40
    elif 0.25 < mpp and mpp <= 0.5:
        mag = 40
    elif 0.5 < mpp and mpp <= 1:
        mag = 20
    elif 1 < mpp and mpp <= 4:
        mag = 10
    elif 4 < mpp:
        mag = 5  # not sure about that one

    return mag
