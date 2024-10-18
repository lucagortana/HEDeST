from __future__ import annotations

import json
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.basics import check_json_classification
from tools.hovernet_tools import get_visium_infos
from tools.hovernet_tools import get_xenium_infos
from tools.openslide_handler import get_slide_object


class SlideVisualizer:
    def __init__(
        self,
        image_path,
        vis_data=None,
        vis_dataname=None,
        dict_cells=None,
        dict_types_colors=None,
        window="full",
        figsize=(18, 15),
    ):

        # input variables
        self.image_path = image_path
        self.adata = vis_data
        self.adata_name = vis_dataname
        self.dict_types_colors = dict_types_colors
        self.window = window
        self.figsize = figsize

        # initialize variables
        self.spots_center = None
        self.spots_diameter = None

        # check adata
        if self.adata is not None and self.adata_name is not None:
            try:
                self.mag_info, self.mpp, self.spots_center, self.spots_diameter = get_visium_infos(
                    self.adata, self.adata_name
                )
                print("Visium data found.")
            except ValueError:
                raise ValueError(
                    f"Impossible to retrieve information. Either this is Visium data, but"
                    f"‘{self.adata_name}’ was not found, or it's not Visium data."
                    "Please check the format and annotations of the data."
                )
        elif self.adata is None:
            print("No Visium data provided. We guess it's an H&E slide from Xenium data.")
            self.mag_info, self.mpp = get_xenium_infos()

        elif self.adata is not None and self.adata_name is None:
            raise ValueError("Please provide the name of the dataset in the adata object.")

        # check dict_cells and open it
        if isinstance(dict_cells, str) and os.path.isfile(dict_cells):
            with open(dict_cells) as json_file:
                self.data = json.load(json_file)
        elif isinstance(dict_cells, dict) or dict_cells is None:
            self.data = dict_cells
        else:
            raise ValueError("dict_cells must be a path to a JSON file or a dictionary.")

        # check dict_types_colors
        if self.data is not None:
            if self.dict_types_colors is None:
                if check_json_classification(self.data):
                    raise ValueError(
                        "dict_types_colors must be provided if dict_cells is a JSON file with classified cells."
                    )
                else:
                    self.dict_types_colors = {"None": ("Unkwnown", (0, 0, 0))}

            elif self.dict_types_colors is not None:
                if not check_json_classification(self.data):
                    print("Warning : You gave a dict_types_colors but the JSON file gives no classification.")
                    self.dict_types_colors = {"None": ("Unkwnown", (0, 0, 0))}

        # extract nuclear information
        if self.data is not None:
            self.nuc_info = self.data["nuc"]
            self.contour_list_wsi = []
            self.type_list_wsi = []
            self.centroid_list_wsi = []
            for inst in self.nuc_info:
                inst_info = self.nuc_info[inst]
                self.contour_list_wsi.append(inst_info["contour"])
                self.type_list_wsi.append(inst_info["type"])
                self.centroid_list_wsi.append(inst_info["centroid"])

        # Initialize slide handler
        self.wsi_obj = get_slide_object(self.image_path, pathlib.Path(self.image_path).suffix, self.mag_info, self.mpp)
        self.original_size = self.wsi_obj.file_ptr.level_dimensions[-1]
        if self.window == "full":
            self.window = (0, 0), self.original_size

        self.x, self.y = self.window[0]
        self.w, self.h = self.window[1]

        # Placeholder for plot objects
        self.region = self.wsi_obj.read_region(self.window[0], self.window[1])
        self.overlaid_output = None

    def plot_slide(self, show_visium=False, title=None, display=True):
        """Adds the histological slide to the plot."""
        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.region)

        if show_visium:
            self._add_visium()

        ax.axis("off")
        ax.set_title([f"Slide - {self.adata_name}", title][title is not None])

        if display:
            plt.show()
        else:
            plt.close(fig)
            return fig

    def plot_seg(self, draw_dot=False, show_visium=False, title=None, display=True):
        """Adds segmentation contours to the slide."""
        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        if self.data is None:
            raise ValueError("You must create a SlideVisualizer object with segmentation info to apply add_seg()")

        tile_info_dict = {}
        for idx, cnt in enumerate(self.contour_list_wsi):
            cnt_tmp = np.array(cnt)
            cnt_tmp = cnt_tmp[
                (cnt_tmp[:, 0] >= self.x)
                & (cnt_tmp[:, 0] <= self.x + self.w)
                & (cnt_tmp[:, 1] >= self.y)
                & (cnt_tmp[:, 1] <= self.y + self.h)
            ]
            label = str(self.type_list_wsi[idx])
            centroid_x = self.centroid_list_wsi[idx][0] - self.x
            centroid_y = self.centroid_list_wsi[idx][1] - self.y
            if cnt_tmp.shape[0] > 0:
                cnt_adj = np.round(cnt_tmp - np.array([self.x, self.y])).astype("int")
                tile_info_dict[idx] = {"contour": cnt_adj, "type": label, "centroid": [centroid_x, centroid_y]}

        self.overlaid_output = visualize_instances_dict(
            self.region, tile_info_dict, type_colour=self.dict_types_colors, draw_dot=draw_dot
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.overlaid_output)

        if show_visium:
            self._add_visium()

        ax.axis("off")
        ax.set_title([f"Segmentation overlay - {self.adata_name}", title][title is not None])

        if display:
            plt.show()
        else:
            plt.close(fig)
            return fig

    def _add_visium(self):
        """Adds Visium spots to the slide."""
        ext_vis = self.spots_diameter / 2
        for spot in self.spots_center:
            spot_x = spot[0] - self.x
            spot_y = spot[1] - self.y
            if -ext_vis <= spot_x <= self.w + ext_vis and -ext_vis <= spot_y <= self.h + ext_vis:
                circle = plt.Circle((spot_x, spot_y), self.spots_diameter / 2, color="black", fill=False, linewidth=3)
                plt.gca().add_patch(circle)


def plot_specific_spot(
    image_path, adata, adata_name, spot_id=None, dict_cells=None, dict_types_colors=None, figsize=(12, 10), display=True
):
    """Plots a specific spot with Visium circles and segmentation."""
    _, _, centers, diameter = get_visium_infos(adata, adata_name)
    spots_coordinates = pd.DataFrame(centers, columns=["x", "y"])
    spots_coordinates["id"] = adata.obs.index

    if spot_id is None:
        spot_id = np.random.choice(spots_coordinates["id"])
        print(f"Randomly selected spot_id: {spot_id}")

    spot_x, spot_y = spots_coordinates[spots_coordinates["id"] == spot_id][["x", "y"]].values[0]
    img_diam = int(diameter + 50)
    window = ((spot_x - img_diam / 2, spot_y - img_diam / 2), (img_diam, img_diam))

    plotter = SlideVisualizer(image_path, adata, adata_name, dict_cells, dict_types_colors, window, figsize=figsize)

    if dict_cells is not None:
        fig = plotter.plot_seg(show_visium=True, title=f"Spot ID: {spot_id}", draw_dot=True, display=display)
    else:
        fig = plotter.plot_slide(show_visium=True, title=f"Spot ID: {spot_id}", display=display)

    return fig


def visualize_instances_dict(input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} ,
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours

    from hovernet
    """
    overlay = np.copy((input_image))

    for _, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        inst_colour = type_colour[inst_info["type"]][1]
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, inst_colour, -1)
    return overlay
