from __future__ import annotations

import base64
import io
import json
import math
import os
import pathlib
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from anndata import AnnData
from loguru import logger
from PIL import Image
from plotly.subplots import make_subplots
from scipy.spatial import KDTree
from tqdm import tqdm

from external.hovernet.misc.wsi_handler import get_file_handler
from hedest.config import TqdmToLogger
from hedest.utils import check_json_classification
from hedest.utils import remove_empty_keys
from hedest.utils import seg_colors_compatible

tqdm_out = TqdmToLogger(logger, level="INFO")


class SlideVisualizer(ABC):
    def __init__(
        self,
        image_path: str,
        adata: Optional[AnnData] = None,
        adata_name: Optional[str] = None,
        seg_dict: Optional[Union[str, Dict[str, Any]]] = None,
        color_dict: Optional[Dict[str, Tuple[str, Tuple[int, int, int]]]] = None,
    ) -> None:
        """
        Initializes the SlideVisualizer.

        Args:
            image_path (str): Path to the histological image.
            adata (AnnData, optional): Optional Anndata object containing Visium data.
            adata_name (str, optional): Name of the dataset in the adata object.
            seg_dict(Union[str, Dict[str, Any]], optional): Segmentation dictionary or path to a JSON file.
            color_dict(Dict, optional): Dictionary mapping cell types to color tuples.
        """

        self.image_path = image_path
        self.adata = adata
        self.adata_name = adata_name
        self.seg_dict = seg_dict
        self.color_dict = color_dict
        self.spots_center = None
        self.spots_diameter = None

        # Check Visium data
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

        # check seg_dict and open it
        if isinstance(seg_dict, str) and os.path.isfile(seg_dict):
            if seg_dict.endswith(".json"):
                with open(seg_dict) as json_file:
                    self.data = json.load(json_file)
            else:
                raise ValueError("seg_dict must be a JSON file if given as a path.")
        elif isinstance(seg_dict, dict) or seg_dict is None:
            self.data = seg_dict
        else:
            raise ValueError("seg_dict must be a path to a JSON file or a dictionary.")

        # check color_dict
        if self.data is not None:
            if self.color_dict is None:
                if check_json_classification(self.data):
                    error1 = "color_dict must be provided if seg_dict is a JSON file with classified cells.\n"
                    error2 = "You can create one with basics.generate_color_dict()."
                    raise ValueError(error1 + error2)
                else:
                    self.color_dict = {"None": ("Unkwnown", (0, 0, 0))}

            elif self.color_dict is not None:
                if not check_json_classification(self.data):
                    print("Warning : You gave a color_dict but the JSON file gives no classification.")
                    self.color_dict = {"None": ("Unkwnown", (0, 0, 0))}

                elif not seg_colors_compatible(self.data, self.color_dict):
                    raise ValueError("Some labels found in your data have not been found in the color dictionnary.")

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

    def plot_specific_spot(
        self,
        spot_id: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        display: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plots a specific Visium spot with its segmentation contours.

        Args:
            spot_id (str, optional): ID of the Visium spot to plot. Randomly selected if None.
            figsize (Tuple[int, int]): Size of the plot.
            display (bool): Whether to display the plot or return the figure.
        Returns:
            Optional[plt.Figure]: The plotted figure if display is False.
        """

        assert self.adata is not None, "Please create a SlideVisualizer object with adata."
        spots_coordinates = pd.DataFrame(self.spots_center, columns=["x", "y"])
        spots_coordinates["id"] = self.adata.obs.index

        if spot_id is None:
            spot_id = np.random.choice(spots_coordinates["id"])
            print(f"Randomly selected spot_id: {spot_id}")

        spot_x, spot_y = spots_coordinates[spots_coordinates["id"] == spot_id][["x", "y"]].values[0]
        img_diam = int(self.spots_diameter + 50)
        window = ((spot_x - img_diam / 2, spot_y - img_diam / 2), (img_diam, img_diam))

        if self.seg_dict is not None:
            fig = self.plot_seg(window, show_visium=True, title=f"Spot ID: {spot_id}", display=display, figsize=figsize)
        else:
            fig = self.plot_slide(
                window, show_visium=True, title=f"Spot ID: {spot_id}", display=display, figsize=figsize
            )

        return fig

    def _set_window(self, window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        """Sets the current viewing window for the slide."""

        self.window = window
        if self.window == "full":
            self.window = (0, 0), self.original_size
        self.x, self.y = self.window[0]
        self.w, self.h = self.window[1]
        self.region = self.wsi_obj.read_region(self.window[0], self.window[1])

    @abstractmethod
    def plot_slide(self) -> None:
        """Abstract method to plot histological slide."""
        pass

    @abstractmethod
    def plot_seg(self) -> None:
        """Abstract method to plot segmentation overlays."""
        pass


class StdVisualizer(SlideVisualizer):
    def plot_slide(
        self,
        window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]],
        spot_prop_df: Optional[pd.DataFrame] = None,
        show_visium: bool = False,
        title: Optional[str] = None,
        display: bool = True,
        figsize: Tuple[int, int] = (18, 15),
    ) -> Optional[plt.Figure]:
        """
        Plots the histological slide with an optional overlay of Visium spots.

        Args:
            window (Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]): Region of the slide to plot.
                    "full" for the entire slide.
            spot_prop_df (pd.DataFrame, optional): DataFrame containing proportions for each Visium spot.
            show_visium (bool): Whether to overlay Visium spots.
            title (str, optional): Optional title for the plot.
            display (bool): Whether to display the plot or return the figure.
            figsize (Tuple[int, int]): Size of the plot.
        Returns:
            Optional[plt.Figure]: The plotted figure if display is False.
        """

        self._set_window(window)

        if show_visium and spot_prop_df is not None:
            if self.w != self.h:
                print("Warning: The selected window is not square. Pie chart placement may be inaccurate.")

        if not show_visium and spot_prop_df is not None:
            print(
                "Warning: You provided a spot_prop_df but did not enable Visium spot drawing (show_visium=False).",
                "No pie charts will be plotted.",
            )
            spot_prop_df = None  # Prevent drawing if not allowed

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.region)

        if show_visium:
            self._add_visium(ax=ax, spot_prop_df=spot_prop_df)

        ax.axis("off")
        ax.set_title([f"Slide - {self.adata_name}", title][title is not None])

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    def plot_seg(
        self,
        window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]],
        draw_dot: bool = False,
        show_visium: bool = False,
        title: Optional[str] = None,
        display: bool = True,
        figsize: Tuple[int, int] = (18, 15),
    ) -> Optional[Any]:
        """
        Adds segmentation contours to the slide.

        Args:
            window (Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]): The (x, y, width, height) of the
                    region to display. "full" for the entire slide.
            draw_dot (bool): Whether to draw centroids as dots.
            show_visium (bool): Whether to overlay Visium spots.
            title (str, optional): Optional title for the plot.
            display (bool): Whether to display the plot. If False, returns the figure object.
            figsize (Tuple[int, int]): Size of the figure.

        Returns:
            The matplotlib figure object if display is False, otherwise None.
        """

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
            self.region, tile_info_dict, color_dict=self.color_dict, draw_dot=draw_dot
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.overlaid_output)

        if show_visium:
            self._add_visium()

        ax.axis("off")
        ax.set_title([f"Segmentation overlay - {self.adata_name}", title][title is not None])

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    def plot_seg_overlays(
        self,
        window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]],
        draw_dot: bool = False,
        figsize: Tuple[int, int] = (20, 12),
        max_cols: int = 4,
        display: bool = True,
        separated: bool = True,
        scale_cells: float = 1.0,
    ) -> plt.Figure:
        """
        Plots segmentation overlays.

        If separated=True → mosaic with one subplot per cell type.
        If separated=False → one plot with all types shown together, filled in colors.

        Args:
            window: Viewing window for all plots.
            draw_dot: Whether to draw centroids as dots.
            figsize: Size of the figure.
            max_cols: Max number of columns (for separated view).
            display: Whether to display the plot.
            separated: If True, make a mosaic; if False, combine all types in one panel.
            scale_cells: Scaling factor for cell size (1.0 = normal, >1 = bigger cells, <1 = smaller).

        Returns:
            plt.Figure: The figure object.
        """

        self._set_window(window)

        if self.data is None:
            raise ValueError("Segmentation data not found.")

        # Build the instance dict for this window
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

                # --- NEW: scale contours around centroid ---
                if scale_cells != 1.0:
                    centroid = np.array([centroid_x, centroid_y])
                    cnt_adj = ((cnt_adj - centroid) * scale_cells + centroid).astype("int")

                tile_info_dict[idx] = {
                    "contour": cnt_adj,
                    "type": label,
                    "centroid": [centroid_x, centroid_y],
                }

        if separated:
            # same mosaic code as before ...
            unique_types = sorted({inst["type"] for inst in tile_info_dict.values()})
            n_types = len(unique_types)
            n_cols = min(n_types, max_cols)
            n_rows = math.ceil(n_types / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()

            for i, cell_type in enumerate(unique_types):
                ax = axes[i]
                custom_color_dict = {}
                for k, v in self.color_dict.items():
                    if k == cell_type:
                        custom_color_dict[k] = v
                    else:
                        custom_color_dict[k] = ("Other", (160, 160, 160))

                overlay = self._visualize_instances_dict(
                    input_image=np.ones_like(self.region) * 255,
                    inst_dict=tile_info_dict,
                    draw_dot=draw_dot,
                    color_dict=custom_color_dict,
                    line_thickness=2,
                    filled_types=[cell_type],
                )

                ax.imshow(overlay)
                ax.set_title(self.color_dict[cell_type][0])
                ax.axis("off")

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.tight_layout()

        else:
            # All types together
            fig, ax = plt.subplots(figsize=figsize)
            overlay = self._visualize_instances_dict(
                input_image=np.ones_like(self.region) * 255,
                inst_dict=tile_info_dict,
                draw_dot=draw_dot,
                color_dict=self.color_dict,
                line_thickness=2,
                filled_types=list(self.color_dict.keys()),
            )
            ax.imshow(overlay)
            ax.set_title("All cell types")
            ax.axis("off")

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    def _add_visium(self, ax: Optional[plt.Axes] = None, spot_prop_df: Optional[pd.DataFrame] = None) -> None:
        if ax is None:
            ax = plt.gca()

        ext_vis = self.spots_diameter / 2
        spot_ids = self.adata.obs.index

        if spot_prop_df is not None:
            colors = [value[1] for value in self.color_dict.values() if value[0] in spot_prop_df.columns]
            colors = [tuple(channel / 255 for channel in rgba) for rgba in colors]

        for i, spot in enumerate(self.spots_center):
            spot_id = spot_ids[i]
            spot_x = spot[0] - self.x
            spot_y = spot[1] - self.y

            if spot_prop_df is not None and spot_id in spot_prop_df.index:
                if ext_vis <= spot_x <= self.w - ext_vis and ext_vis <= spot_y <= self.h - ext_vis:
                    proportions = spot_prop_df.loc[spot_id].values

                    # Convert center to display coordinates
                    center_disp = ax.transData.transform((spot_x, spot_y))
                    radius_disp = ax.transData.transform((spot_x + ext_vis, spot_y))[0] - center_disp[0]

                    # Normalize to figure coords
                    fig = ax.figure
                    fig_width, fig_height = fig.bbox.width, fig.bbox.height

                    x0 = (center_disp[0] - radius_disp) / fig_width
                    y0 = (center_disp[1] - radius_disp) / fig_height
                    width = (2 * radius_disp) / fig_width
                    height = (2 * radius_disp) / fig_height

                    pie_ax = fig.add_axes([x0, y0, width, height])
                    pie_ax.pie(proportions, startangle=90, colors=colors)
                    pie_ax.set_aspect("equal")
                    pie_ax.axis("off")
            else:
                # Draw black circle (existing rule)
                if -ext_vis <= spot_x <= self.w + ext_vis and -ext_vis <= spot_y <= self.h + ext_vis:
                    circle = plt.Circle((spot_x, spot_y), ext_vis, color="black", fill=False, linewidth=3)
                    ax.add_patch(circle)

    def _visualize_instances_dict(
        self,
        input_image: np.ndarray,
        inst_dict: Dict[int, Dict[str, Any]],
        draw_dot: bool = False,
        color_dict: Optional[Dict[str, Tuple[str, Tuple[int, int, int]]]] = None,
        line_thickness: int = 2,
        filled_types: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Overlays segmentation results (dictionary) on the image as contours. Adapted from Hovernet.

        Args:
            input_image (np.ndarray): Input image array.
            inst_dict (Dict): Dictionary containing segmentation instance data.
            draw_dot (bool): Whether to draw centroids as dots.
            color_dict (Dict, optional): Dictionary mapping type IDs to (name, color).
            line_thickness (int): Thickness of contour lines.

        Returns:
            Image with overlaid segmentation contours.
        """

        overlay = np.copy((input_image))

        for _, [_, inst_info] in enumerate(inst_dict.items()):
            inst_contour = inst_info["contour"]
            inst_colour = color_dict[inst_info["type"]][1]
            if filled_types is not None and inst_info["type"] in filled_types:
                cv2.drawContours(overlay, [inst_contour], -1, inst_colour, -1)
            else:
                cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

            if draw_dot:
                inst_centroid = inst_info["centroid"]
                inst_centroid = tuple([int(v) for v in inst_centroid])
                overlay = cv2.circle(overlay, inst_centroid, 3, inst_colour, -1)
        return overlay


class IntVisualizer(SlideVisualizer):
    def plot_slide(
        self,
        window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]],
        show_visium: bool = False,
        title: Optional[str] = None,
        display: bool = True,
        figsize: Tuple[int, int] = (18, 15),
    ) -> Optional[plt.Figure]:
        """
        Plots the histological slide with an optional overlay of Visium spots.

        Args:
            window (Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]): Region of the slide to plot.
                    "full" for the entire slide.
            show_visium (bool): Whether to overlay Visium spots.
            title (str, optional): Optional title for the plot.
            display (bool): Whether to display the plot or return the figure.
            figsize (Tuple[int, int]): Size of the plot.
        Returns:
            Optional[plt.Figure]: The plotted figure if display is False.
        """

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
            return None
        else:
            plt.close(fig)
            return fig

    def plot_seg(
        self,
        window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]],
        show_visium: bool = False,
        line_width: int = 2,
        title: Optional[str] = None,
        display: bool = True,
        figsize: Tuple[int, int] = (18, 15),
    ) -> Optional[Any]:
        """
        Adds segmentation contours to the slide using Plotly.

        Args:
            window (Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]): The (x, y, width, height) of the
                    region to display. "full" for the entire slide.
            show_visium (bool): Whether to overlay Visium spots.
            line_width (int): Line width for segmentation contours.
            title (str, optional): Optional title for the plot.
            display (bool): Whether to display the plot. If False, returns the figure object.
            figsize (Tuple[int, int]): Size of the figure.

        Returns:
            The Plotly figure object if display is False, otherwise None.
        """

        self._set_window(window)

        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        if self.data is None:
            raise ValueError("You must create a SlideVisualizer object with segmentation info to apply plot_seg()")

        fig = make_subplots()
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
                color_rgb = self.color_dict[label][1]
                color = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"

                cell_id = self.id_list_wsi[idx]
                cell_type_name = self.color_dict[label][0]

                fig.add_trace(
                    go.Scatter(
                        x=cnt_adj[:, 0],
                        y=[self.region.shape[0] - y for y in cnt_adj[:, 1]],
                        mode="lines",
                        name=f"{cell_id}",
                        hoverinfo="text",
                        text=[f"cell_id: {cell_id}<br>type: {cell_type_name}"] * len(cnt_adj),
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
            return None
        else:
            return fig

    def _add_visium(self, fig: Optional[Any] = None) -> None:
        """
        Adds Visium spots to the plot.

        Args:
            fig: Optional Plotly figure object for visualization.
        """

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
            rad_visium = (self.spots_diameter / 2) * (self.original_size[0] / self.w) / 20  # -> to be changed
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

    def _convert_array_to_base64(self, array: np.ndarray) -> str:
        """
        Converts a NumPy array to a base64 string.

        Args:
            array: NumPy array representing an image.

        Returns:
            Base64-encoded string of the image.
        """

        image = Image.fromarray(array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return "data:image/png;base64," + img_str


def count_cell_types(seg_dict: Dict[str, Any], ct_list: List[str]) -> pd.DataFrame:
    """
    Counts cell types in the segmentation dictionary.

    Args:
        seg_dict (Dict): Dictionary containing segmentation data.
        ct_list (List[str]): List of cell type names.

    Returns:
        DataFrame containing counts of each cell type.
    """

    cell_type_counts = {}
    nuc = seg_dict["nuc"]
    for cell_id in nuc.keys():
        label = nuc[cell_id]["type"]
        cell_type = ct_list[int(label)]
        if cell_type not in cell_type_counts.keys():
            cell_type_counts[cell_type] = 1
        else:
            cell_type_counts[cell_type] += 1
    df = pd.DataFrame([cell_type_counts])

    return df


def map_cells_to_spots(adata: AnnData, adata_name: str, json_path: str, only_in: bool = True) -> Dict[str, List[str]]:
    """
    Maps cells to spots based on centroids of the cells and spots.

    Args:
        adata (AnnData): Anndata object containing the Visium dataset.
        adata_name (str): Name of the dataset in the adata object.
        json_path (str): Path to the JSON file with cell centroids.
        only_in (bool): If True, maps cells only if they are located in a spot.
                        If False, maps cells to the closest spot regardless of distance.

    Returns:
        Dictionary mapping cells to spots.
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
    for i, cell in enumerate(tqdm(centroid_array, file=tqdm_out, desc="Mapping cells to spots")):
        if only_in:
            # "only in" method: find spots within the diameter
            indices = tree.query_ball_point(cell, r=diameter / 2)
            for idx in indices:
                dict_cells_spots[spots_ids[idx]].append(str(i))
        else:
            # "not only in" method: find the closest spot within the diameter
            dist, idx = tree.query(cell)
            if dist <= diameter:
                dict_cells_spots[spots_ids[idx]].append(str(i))

    return remove_empty_keys(dict_cells_spots)
