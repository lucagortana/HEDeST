from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
import openslide


class OpenSlideHandler:
    """Class for handling OpenSlide supported whole-slide images.
    From hovernet code."""

    def __init__(self, file_path, mag, mpp):
        """file_path (string): path to single whole-slide image."""
        self.file_ptr = openslide.OpenSlide(file_path)  # load OpenSlide object
        self.metadata = self.__load_metadata(mag, mpp)

        # only used for cases where the read magnification is different from
        self.image_ptr = None  # the existing modes of the read file
        self.read_level = None
        self._prepare_reading(mag)

    def __load_metadata(self, mag, mpp):
        metadata = {}

        level_0_magnification = float(mag)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        mpp_array = np.array([mpp, mpp])

        metadata = [
            ("available_mag", magnification_level),  # highest to lowest mag
            ("base_mag", magnification_level[0]),
            ("mpp  ", mpp_array),
            ("base_shape", np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)

    def _prepare_reading(self, read_mag, read_mpp=None, cache_path=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """
        read_lv, scale_factor = self._get_read_info(read_mag=read_mag, read_mpp=read_mpp)

        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self._get_full_img(read_mag=read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode="r")
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], "Not supported uneven `read_mpp`"
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]

        hires_mag = read_mag
        scale_factor = None
        if read_mag not in self.metadata["available_mag"]:
            if read_mag > self.metadata["base_mag"]:
                scale_factor = read_mag / self.metadata["base_mag"]
                hires_mag = self.metadata["base_mag"]
            else:
                mag_list = np.array(self.metadata["available_mag"])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_factor = read_mag / hires_mag

        hires_lv = self.metadata["available_mag"].index(hires_mag)
        return hires_lv, scale_factor

    def _get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """

        read_lv, scale_factor = self._get_read_info(read_mag=read_mag, read_mpp=read_mpp)

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(wsi_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp)
        return wsi_img

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.

        Args:
            coords (tuple): (dims_x, dims_y),
                            top left coordinates of image region at selected
                            `read_mag` or `read_mpp` from `prepare_reading`
            size (tuple): (dims_x, dims_y)
                            width and height of image region at selected
                            `read_mag` or `read_mpp` from `prepare_reading`

        """
        if self.image_ptr is None:
            # convert coord from read lv to lv zero
            lv_0_shape = np.array(self.file_ptr.level_dimensions[0])
            lv_r_shape = np.array(self.file_ptr.level_dimensions[self.read_lv])
            up_sample = (lv_0_shape / lv_r_shape)[0]
            new_coord = [0, 0]
            new_coord[0] = int(coords[0] * up_sample)
            new_coord[1] = int(coords[1] * up_sample)
            region = self.file_ptr.read_region(new_coord, self.read_lv, size)
        else:
            region = self.image_ptr[coords[1] : coords[1] + size[1], coords[0] : coords[0] + size[0]]
        return np.array(region)[..., :3]


def get_slide_object(path, backend, mag, mpp):
    if backend in [
        ".svs",
        ".tif",
        ".vms",
        ".vmu",
        ".ndpi",
        ".scn",
        ".mrxs",
        ".tiff",
        ".svslide",
        ".bif",
    ]:
        return OpenSlideHandler(path, mag, mpp)
    else:
        assert False, "Unknown WSI format `%s`" % backend
