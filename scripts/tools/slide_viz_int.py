from __future__ import annotations

import base64
import io
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
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

        # Extract nuclear info
        if self.data is not None:
            self.nuc_info = self.data["nuc"]
            self.contour_list_wsi = []
            self.type_list_wsi = []
            self.id_list_wsi = []
            for inst in self.nuc_info:
                inst_info = self.nuc_info[inst]
                self.contour_list_wsi.append(inst_info["contour"])
                self.type_list_wsi.append(inst_info["type"])
                self.id_list_wsi.append(inst)

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

    def _convert_array_to_base64(self, array):
        """Converts a NumPy array to a base64 string."""
        image = Image.fromarray(array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return "data:image/png;base64," + img_str

    def plot_slide(self, show_visium=False, title=None, display=True):
        """Adds the histological slide to the plot."""
        if show_visium and (self.spots_center is None or self.spots_diameter is None):
            print("Warning : You cannot plot Visium spots without Visium data.")
            show_visium = False

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.region)

        if show_visium:
            self._add_visium_plt()

        ax.axis("off")
        ax.set_title([f"Slide - {self.adata_name}", title][title is not None])

        if display:
            plt.show()
        else:
            plt.close(fig)
            return fig

    def plot_seg(self, show_visium=False, line_width=2, title=None, display=True):
        """Adds segmentation contours to the slide using Plotly."""
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
            width=self.figsize[0] * 80,
            height=self.figsize[1] * 80,
            hovermode="closest",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        if show_visium:
            self._add_visium_plotly(fig)

        if display:
            fig.show()
        else:
            return fig

    def _add_visium_plotly(self, fig):
        """Adds Visium spots to the plot."""

        rad_visium = (self.spots_diameter / 2) * (self.original_size[0] / self.w) / 20  # to be changed
        ext_vis = self.spots_diameter / 2

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

    def _add_visium_plt(self):
        """Adds Visium spots to the slide."""

        ext_vis = self.spots_diameter / 2

        for spot in self.spots_center:
            spot_x = spot[0] - self.x
            spot_y = spot[1] - self.y
            if -ext_vis <= spot_x <= self.w + ext_vis and -ext_vis <= spot_y <= self.h + ext_vis:
                circle = plt.Circle((spot_x, spot_y), self.spots_diameter / 2, color="black", fill=False, linewidth=1)
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
        fig = plotter.plot_seg(show_visium=True, line_width=4, title=f"Spot ID: {spot_id}", display=display)
    else:
        fig = plotter.plot_slide(show_visium=True, title=f"Spot ID: {spot_id}", display=display)

    return fig
