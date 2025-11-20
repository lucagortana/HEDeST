from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import openslide
from numpy import ndarray
from openslide import OpenSlide
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.morphology import opening
from skimage.morphology import square
from skimage.transform import resize

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def make_auto_mask(slide: Union[OpenSlide, str],
                   mask_level: int, 
                   save: Optional[str] = None) -> ndarray:
    """
    Creates a binary mask from a downsampled version of a WSI. Uses the Otsu algorithm 
    and a morphological opening.

    Args:
        slide: OpenSlide object or path to a WSI.
        mask_level: Level at which to create the mask.
        save: Path to save the mask.

    Returns:
        Binary mask.
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

    print("-> (1/3) RGB to Gray conversion...")
    im_gray = rgb2gray(im)
    print("-> (2/3) Clearing border...")
    im_gray = clear_border(im_gray, prop=30)
    size = im_gray.shape
    im_gray = im_gray.flatten()
    pixels_int = im_gray[np.logical_and(im_gray > 0.1, im_gray < 0.98)]
    print("-> (3/3) Otsu thresholding...")
    t = threshold_otsu(pixels_int)
    mask = opening(
        closing(np.logical_and(im_gray < t, im_gray > 0.1).reshape(size), footprint=square(32)), footprint=square(32)
    )
    mask = (np.stack([mask] * 3, axis=-1).astype(np.uint8)) * 255
    if save is not None:
        cv2.imwrite(save, mask)
        print(f"Mask saved at {save}")
    return mask


def clear_border(mask: ndarray, prop: int) -> ndarray:
    """
    Clears the border of a binary mask.
    
    Args:
        mask: Binary mask.
        prop: Proportion of the border to clear.
    
    Returns:
        Mask with cleared border
    """

    r, c = mask.shape
    pr, pc = r // prop, c // prop
    mask[:pr, :] = 0
    mask[r - pr :, :] = 0
    mask[:, :pc] = 0
    mask[:, c - pc :] = 0
    return mask


def get_x_y(slide: OpenSlide, 
            point_l: Tuple[int, int], 
            level: int, 
            integer: bool = True) -> Tuple[int, int]:
    """
    Code @PeterNaylor from useful_wsi.
    Given a point point_l = (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point point_0 = (x_0, y_0).

    Args:
        slide: Openslide object from which we extract.
        point_l: A tuple, or tuple like object of size 2 with integers.
        level: Integer, level of the associated point.
        integer: Boolean, by default True. Wether or not to round
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
