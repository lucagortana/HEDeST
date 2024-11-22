from __future__ import annotations

import base64
import io
import json
import os
import pathlib
from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import plotly.graph_objects as go
import torch
from misc.wsi_handler import get_file_handler
from PIL import Image
from plotly.subplots import make_subplots
from scipy.spatial import KDTree
from tqdm import tqdm

from deconvplugin.basics import check_json_classification
from deconvplugin.basics import remove_empty_keys


class SlideVisualizer(ABC):
    def __init__(
        self,
        image_path,
        vis_data=None,
        vis_dataname=None,
        dict_cells=None,
        dict_types_colors=None,
    ):
        # input variables
        self.image_path = image_path
        self.adata = vis_data
        self.adata_name = vis_dataname
        self.dict_cells = dict_cells
        self.dict_types_colors = dict_types_colors

        # initialize variables
        self.spots_center = None
        self.spots_diameter = None

        # change : adata must be provided
        if self.adata is not None and self.adata_name is not None:
            try:
                self.spots_center = self.adata.obsm["spatial"].astype("int64")
                self.spots_diameter = self.adata.uns["spatial"][self.adata_name]["scalefactors"][
                    "spot_diameter_fullres"
                ]
                print("Visium data found.")
            except ValueError:
                raise ValueError(
                    f"Impossible to retrieve information. Either this is Visium data, but"
                    f"‘{self.adata_name}’ was not found, or it's not Visium data."
                    "Please check the format and annotations of the data."
                )
        elif self.adata is None:
            print("No Visium data provided. You won't be able to plot Visium spots.")

        elif self.adata is not None and self.adata_name is None:
            raise ValueError("Please provide the name of the dataset in the adata object.")

        # check dict_cells and open it
        if isinstance(dict_cells, str) and os.path.isfile(dict_cells):
            if dict_cells.endswith(".json"):
                with open(dict_cells) as json_file:
                    self.data = json.load(json_file)
            else:
                raise ValueError("dict_cells must be a JSON file if given as a path.")
        elif isinstance(dict_cells, dict) or dict_cells is None:
            self.data = dict_cells
        else:
            raise ValueError("dict_cells must be a path to a JSON file or a dictionary.")

        # check dict_types_colors
        if self.data is not None:
            if self.dict_types_colors is None:
                if check_json_classification(self.data):
                    error1 = "dict_types_colors must be provided if dict_cells is a JSON file with classified cells.\n"
                    error2 = "You can create one with basics.generate_color_dict()."
                    raise ValueError(error1 + error2)
                else:
                    self.dict_types_colors = {"None": ("Unkwnown", (0, 0, 0))}

            elif self.dict_types_colors is not None:
                if not check_json_classification(self.data):
                    print("Warning : You gave a dict_types_colors but the JSON file gives no classification.")
                    self.dict_types_colors = {"None": ("Unkwnown", (0, 0, 0))}

        # Extract nuclear info
        if self.data is not None:
            self.nuc_info = self.data["nuc"]
            self.contour_list_wsi = []
            self.type_list_wsi = []
            self.id_list_wsi = []
            self.centroid_list_wsi = []
            for inst in self.nuc_info:
                inst_info = self.nuc_info[inst]
                self.contour_list_wsi.append(inst_info["contour"])
                self.type_list_wsi.append(inst_info["type"])
                self.id_list_wsi.append(inst)
                self.centroid_list_wsi.append(inst_info["centroid"])

        # Initialize slide handler
        self.wsi_obj = get_file_handler(self.image_path, pathlib.Path(self.image_path).suffix)
        self.mag_info = self.wsi_obj.metadata["base_mag"]
        self.wsi_obj.prepare_reading(read_mag=self.mag_info)
        self.original_size = self.wsi_obj.file_ptr.level_dimensions[-1]
        self.overlaid_output = None

    def plot_slide(self, window, show_visium=False, title=None, display=True, figsize=(18, 15)):
        """Adds the histological slide to the plot."""

        self._set_window(window)

        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        fig, ax = plt.subplots(figsize=figsize)
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

    def plot_specific_spot(self, spot_id=None, figsize=(12, 10), display=True):
        """Plots a specific spot with Visium circles and segmentation."""

        spots_coordinates = pd.DataFrame(self.spots_center, columns=["x", "y"])
        spots_coordinates["id"] = self.adata.obs.index

        if spot_id is None:
            spot_id = np.random.choice(spots_coordinates["id"])
            print(f"Randomly selected spot_id: {spot_id}")

        spot_x, spot_y = spots_coordinates[spots_coordinates["id"] == spot_id][["x", "y"]].values[0]
        img_diam = int(self.spots_diameter + 50)
        window = ((spot_x - img_diam / 2, spot_y - img_diam / 2), (img_diam, img_diam))

        if self.dict_cells is not None:
            fig = self.plot_seg(window, show_visium=True, title=f"Spot ID: {spot_id}", display=display, figsize=figsize)
        else:
            fig = self.plot_slide(
                window, show_visium=True, title=f"Spot ID: {spot_id}", display=display, figsize=figsize
            )

        return fig

    def _set_window(self, window):
        """Sets the window to a specific region of the slide."""
        self.window = window
        if self.window == "full":
            self.window = (0, 0), self.original_size
        self.x, self.y = self.window[0]
        self.w, self.h = self.window[1]
        self.region = self.wsi_obj.read_region(self.window[0], self.window[1])

    @abstractmethod
    def plot_seg(self):
        pass

    @abstractmethod
    def _add_visium(self):
        pass


class StdVisualizer(SlideVisualizer):
    def plot_seg(self, window, draw_dot=False, show_visium=False, title=None, display=True, figsize=(18, 15)):
        """Adds segmentation contours to the slide."""

        self._set_window(window)

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

        self.overlaid_output = self._visualize_instances_dict(
            self.region, tile_info_dict, type_colour=self.dict_types_colors, draw_dot=draw_dot
        )

        fig, ax = plt.subplots(figsize=figsize)
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

    def _visualize_instances_dict(self, input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2):
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


class IntVisualizer(SlideVisualizer):
    def plot_seg(self, window, show_visium=False, line_width=2, title=None, display=True, figsize=(18, 15)):
        """Adds segmentation contours to the slide using Plotly."""

        self._set_window(window)

        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        if self.data is None:
            raise ValueError("You must create a SlideVisualizer object with segmentation info to apply plot_seg()")

        fig = make_subplots()

        # Convert the region (NumPy array) to a base64 string
        img_str = self._convert_array_to_base64(self.region)

        # Add the slide image
        fig.add_layout_image(
            dict(
                source=img_str,
                xref="x",
                yref="y",
                x=0,
                y=self.region.shape[0],
                sizex=self.region.shape[1],
                sizey=self.region.shape[0],
                sizing="stretch",
                layer="below",
            )
        )

        # Plot each cell
        for idx, cnt in enumerate(self.contour_list_wsi):
            cnt_tmp = np.array(cnt)
            cnt_tmp = cnt_tmp[
                (cnt_tmp[:, 0] >= self.x)
                & (cnt_tmp[:, 0] <= self.x + self.w)
                & (cnt_tmp[:, 1] >= self.y)
                & (cnt_tmp[:, 1] <= self.y + self.h)
            ]
            label = str(self.type_list_wsi[idx])
            if cnt_tmp.shape[0] > 0:
                cnt_adj = np.round(cnt_tmp - np.array([self.x, self.y])).astype("int")
                color_rgb = self.dict_types_colors[label][1]
                color = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"

                fig.add_trace(
                    go.Scatter(
                        x=cnt_adj[:, 0],
                        y=[self.region.shape[0] - y for y in cnt_adj[:, 1]],  # Flipping the y-coordinates
                        mode="lines",
                        name=self.dict_types_colors[label][0],
                        hoverinfo="name",
                        line=dict(color=color, width=line_width),
                    )
                )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        fig.update_layout(
            title=[f"Segmentation overlay - {self.adata_name}", title][title is not None],
            title_x=0.5,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            hovermode="closest",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        if show_visium:
            self._add_visium(fig)

        if display:
            fig.show()
        else:
            return fig

    def _add_visium(self, fig=None):
        """Adds Visium spots to the plot."""

        ext_vis = self.spots_diameter / 2

        if fig is None:
            for spot in self.spots_center:
                spot_x = spot[0] - self.x
                spot_y = spot[1] - self.y
                if -ext_vis <= spot_x <= self.w + ext_vis and -ext_vis <= spot_y <= self.h + ext_vis:
                    circle = plt.Circle(
                        (spot_x, spot_y), self.spots_diameter / 2, color="black", fill=False, linewidth=1
                    )
                    plt.gca().add_patch(circle)

        else:
            rad_visium = (self.spots_diameter / 2) * (self.original_size[0] / self.w) / 20  # to be changed
            for i, spot in enumerate(self.spots_center):
                spot_x = spot[0] - self.x
                spot_y = spot[1] - self.y
                if -ext_vis <= spot_x <= self.w + ext_vis and -ext_vis <= spot_y <= self.h + ext_vis:
                    fig.add_shape(
                        type="circle",
                        x0=spot_x - self.spots_diameter / 2,
                        y0=self.region.shape[0] - (spot_y + self.spots_diameter / 2),
                        x1=spot_x + self.spots_diameter / 2,
                        y1=self.region.shape[0] - (spot_y - self.spots_diameter / 2),
                        line=dict(color="black", width=1),
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[spot_x],
                            y=[self.region.shape[0] - spot_y],
                            mode="markers",
                            marker=dict(size=rad_visium, color="rgba(0,0,0,0)"),
                            hoverinfo="text",
                            text=self.adata.obs.index[i],
                        )
                    )

    def _convert_array_to_base64(self, array):
        """Converts a NumPy array to a base64 string."""
        image = Image.fromarray(array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return "data:image/png;base64," + img_str


# Tool functions


def count_cell_types(hovernet_dict, ct_list):
    cell_type_counts = {}
    nuc = hovernet_dict["nuc"]
    for cell_id in nuc.keys():
        label = nuc[cell_id]["type"]
        cell_type = ct_list[int(label)]
        if cell_type not in cell_type_counts.keys():
            cell_type_counts[cell_type] = 1
        else:
            cell_type_counts[cell_type] += 1
    df = pd.DataFrame([cell_type_counts])

    return df


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


def map_cells_to_spots(adata, adata_name, json_path, only_in=True):
    """
    Maps cells to spots based on the centroids of the cells and spots.
    Args:
        adata: Anndata object containing the Visium dataset.
        adata_name: Name of the dataset in the adata object.
        json_path: Path to the json file with the cell centroids.
        only_in: If True, map cells only to spots within the spot's diameter.
                 If False, map each cell to its nearest spot regardless of distance.
    Returns:
        dict_cells_spots: Dictionary with the mapping of cells to spots.
    Examples:
        dict_cells_spots = map_cells_to_spots(adata, adata_name, json_path, only_in=True)
    """
    centroid_list = []
    spots_coordinates = adata.obsm["spatial"].astype("int64")
    diameter = adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"]
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

    # Query the KDTree based on only_in parameter
    for i, cell in enumerate(tqdm(centroid_array, desc="Mapping cells to spots")):
        if only_in:
            # "only in" method: find spots within the diameter
            indices = tree.query_ball_point(cell, r=diameter / 2)
            for idx in indices:
                dict_cells_spots[spots_ids[idx]].append(str(i))
        else:
            # "not only in" method: find the closest spot regardless of distance
            idx = tree.query(cell)[1]
            dict_cells_spots[spots_ids[idx]].append(str(i))

    return remove_empty_keys(dict_cells_spots)
