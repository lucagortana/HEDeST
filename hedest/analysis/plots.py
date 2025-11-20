from __future__ import annotations

import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hedest.analysis.postseg import StdVisualizer
from hedest.utils import fig_to_array


def plot_pie_chart(
    ax: plt.Axes, data: pd.Series, color_dict: Dict[str, Tuple], plot_labels: bool = False, add_legend: bool = False
) -> None:
    """
    Plots a pie chart for the given data on the provided axis.

    Args:
        ax: The axis to plot the pie chart on.
        data: A Series containing the proportions for each category.
        color_dict: A dictionary mapping category to color.
        plot_labels: Whether to display labels on the chart. Defaults to False.
        add_legend: Whether to add a legend to the chart. Defaults to False.
    """

    labels = data.index
    proportions = data.values

    colors = [color_dict[cell_type] for cell_type in labels if cell_type in color_dict.keys()]

    if plot_labels:
        wedges, _, _ = ax.pie(proportions, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    else:
        wedges, _, _ = ax.pie(proportions, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")

    if add_legend:
        legend_labels = [label for label in labels]
        wedges, legend_labels, _ = zip(
            *sorted(zip(wedges, legend_labels, proportions), key=lambda x: x[2], reverse=True)
        )
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)


def plot_legend(
    color_dict: Dict[str, Union[Tuple, Tuple[str, List[int]]]], ax: plt.Axes = None, savefig: Optional[str] = None
) -> None:
    """
    Plots a legend with an optional axis. If no axis is provided, a new figure is created.

    Args:
        color_dict: A dictionary mapping identifiers to labels and colors.
        ax: Axis to add the legend to. Defaults to None.
        savefig: Filename to save the plot. Defaults to None.
    """

    if isinstance(next(iter(color_dict.values())), tuple):
        # Format 1: {'fibroblast': (r, g, b, a)}
        legend_labels = list(color_dict.keys())
        legend_colors = list(color_dict.values())
    else:
        # Format 2: {'0': ['fibroblast', [r, g, b, a]]}
        legend_labels = [str(v[0]) for v in color_dict.values()]
        legend_colors = [tuple(c / 255 for c in v[1]) for v in color_dict.values()]

    patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=18) for color in legend_colors
    ]

    fig_created = False
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        fig_created = True

    ax.legend(patches, legend_labels, loc="center", fontsize=14)

    if savefig and fig_created:
        plt.savefig(savefig, bbox_inches="tight", dpi=300)

    if fig_created:
        plt.show()


def plot_mosaic_cells(
    spot_dict: Dict[str, List[str]],
    image_dict: Dict[str, torch.Tensor],
    spot_id: Optional[str] = None,
    predicted_labels: Optional[Dict[str, Dict[str, str]]] = None,
    true_labels: Optional[Dict[str, Dict[str, str]]] = None,
    num_cols: int = 8,
    display: bool = True,
) -> Optional[plt.Figure]:
    """
    Plots a grid of cell images for a given spot along with predicted labels.

    Args:
        spot_dict: A dictionary mapping spot IDs to lists of cell IDs.
        image_dict: A dictionary mapping cell IDs to images.
        spot_id: The spot ID to visualize. Defaults to a random spot.
        predicted_labels: Predicted labels for the cells. Defaults to None.
        true_labels: True labels for the cells. Defaults to None.
        num_cols: Number of columns in the grid. Defaults to 8.
        display: Whether to display the plot directly. Defaults to True.

    Returns:
        Figure object if `display` is False.
    """

    m = 4

    if true_labels is not None and predicted_labels is None:
        raise ValueError("If true_labels is provided, predicted_labels must also be provided.")

    # Select a random spot_id if not provided
    if spot_id is None:
        spot_id = random.choice(list(spot_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    cell_ids = spot_dict[spot_id]

    if len(cell_ids) == 0:
        print(f"No individual cells to display for spot_id: {spot_id}")
        return None  # Just return if no cells are found, no need to plot cells

    num_cells = len(cell_ids)
    num_rows = (num_cells + num_cols - 1) // num_cols

    # Create the mosaic plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, (num_cells // num_cols * m) + m))
    axes = axes.flatten()

    for i, cell_id in enumerate(cell_ids):
        cell_image = image_dict[cell_id].cpu().numpy().transpose(1, 2, 0)

        # Plot the cell image
        axes[i].imshow(cell_image)
        axes[i].axis("off")

        if predicted_labels is not None:
            predicted_class = predicted_labels[cell_id]["class"]

            if true_labels is not None:
                true_class = true_labels[cell_id]["class"]

                if predicted_class == true_class:
                    axes[i].set_title(f"Label: {predicted_class}", color="black")
                else:
                    axes[i].set_title(f"Label: {predicted_class} ({true_class})", color="red")
            else:
                axes[i].set_title(f"Label: {predicted_class}", color="black")

    for i in range(len(cell_ids), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if display:
        plt.show()
        return None
    else:
        plt.close(fig)
        return fig


def plot_cell(
    image_dict: Dict[str, torch.Tensor], ax: Optional[plt.Axes] = None, cell_id: Optional[Union[str, int]] = None
) -> None:
    """
    Plots a single cell image from the provided dictionary.

    Args:
        image_dict: A dictionary mapping cell IDs to images.
        ax: Axis to plot the cell on. If None, creates a new plot. Defaults to None.
        cell_id: The ID of the cell to plot. Defaults to a random cell.
    """

    if cell_id is None:
        cell_id = np.random.choice(list(image_dict.keys()))
    else:
        if not (isinstance(cell_id, str) or isinstance(cell_id, int)):
            raise ValueError("cell_id must be either a string or an integer")

    image = image_dict[str(cell_id)].permute(1, 2, 0)

    if ax is None:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(image)
        ax.axis("off")


def plot_history(
    history_train: List[float], history_val: List[float], show: bool = False, savefig: Optional[str] = None
) -> None:
    """
    Plots training and validation loss history.

    Args:
        history_train: List of training loss values.
        history_val: List of validation loss values.
        show: Whether to display the plot. Defaults to False.
        savefig: Filename to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history_train) + 1), history_train, color="blue")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history_val) + 1), history_val, color="blue")
    plt.title("Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_grid_celltype(
    predictions: pd.DataFrame,
    image_dict: Dict[str, np.ndarray],
    cell_type: str,
    n: int = 20,
    selection: str = "random",
    title: str = "",
    show_probs: bool = True,
    display: bool = False,
) -> Optional[plt.Figure]:
    """
    Plots a grid of cell images predicted as a specific cell type.

    Args:
        predictions: DataFrame of predicted probabilities for each cell type.
        image_dict: A dictionary mapping cell IDs to images.
        cell_type: The cell type to filter images by.
        n: Number of images to display. Defaults to 20.
        selection: Selection mode ("max" or "random"). Defaults to "random".
        show_probs: Whether to display probabilities on the images. Defaults to True.
        display: Whether to display the plot. Defaults to False.

    Returns:
        Figure object if `display` is False.
    """

    max_prob_cell_types = predictions.idxmax(axis=1)
    max_prob_indices = max_prob_cell_types[max_prob_cell_types == cell_type].index
    max_probs = predictions.loc[max_prob_indices, cell_type]

    if selection == "max":
        selected_indices = max_probs.nlargest(n).index
    elif selection == "random":
        selected_indices = max_prob_indices.to_series().sample(n=min(n, len(max_prob_indices)))
    else:
        raise ValueError("Invalid selection mode. Choose 'max' or 'random'.")

    selected_images = [image_dict[cell_id] for cell_id in selected_indices if cell_id in image_dict]
    selected_probs = max_probs[selected_indices]

    num_cols = min(10, n)  # max 10 images per row
    num_rows = int(np.ceil(len(selected_images) / num_cols))

    if show_probs:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
        axes = np.atleast_2d(axes)

        for i, ax in enumerate(axes.flat):
            if i < len(selected_images):
                img = selected_images[i].cpu().numpy().transpose((1, 2, 0))
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                prob_text = f"{selected_probs.iloc[i]:.2f}"
                ax.set_title(prob_text, fontsize=8, color="blue")
            else:
                ax.axis("off")

        plt.suptitle(title)
        plt.tight_layout()

    else:
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols, num_rows), gridspec_kw={"wspace": 0.0, "hspace": 0.0}
        )
        axes = np.atleast_2d(axes)

        for i, ax in enumerate(axes.flat):
            if i < len(selected_images):
                img = selected_images[i].cpu().numpy().transpose((1, 2, 0))
                ax.imshow(img, cmap="gray")
            ax.axis("off")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if display:
        plt.show()
        return None
    else:
        plt.close(fig)
        return fig


def plot_predicted_cell_labels_in_spot(
    spot_dict: Dict[str, List[str]],
    adata: object,
    adata_name: str,
    image_path: str,
    image_dict: Dict[str, np.ndarray],
    predicted_labels: Optional[Dict[str, Dict[str, str]]] = None,
    true_labels: Optional[Dict[str, Dict[str, str]]] = None,
    spot_id: Optional[str] = None,
    display: bool = True,
) -> Optional[plt.Figure]:
    """
    Plots a visualization of a spot with cell images and predicted labels.

    Args:
        spot_dict: A dictionary mapping spot IDs to cell IDs.
        adata: Annotated data object.
        adata_name: Name of the annotated dataset.
        image_path: Path to the image data.
        image_dict: A dictionary mapping cell IDs to images.
        predicted_labels: Predicted labels for the cells. Defaults to None.
        true_labels: True labels for the cells. Defaults to None.
        spot_id: Spot ID to visualize. Defaults to a random spot.
        display: Whether to display the plot. Defaults to True.

    Returns:
        Figure object if `display` is False.
    """

    if spot_id is None:
        spot_id = random.choice(list(spot_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    elif spot_id not in spot_dict:
        raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

    plotter = StdVisualizer(image_path, adata, adata_name)
    fig1 = plotter.plot_specific_spot(spot_id=spot_id, display=False)
    fig2 = plot_mosaic_cells(
        spot_dict=spot_dict,
        image_dict=image_dict,
        spot_id=spot_id,
        predicted_labels=predicted_labels,
        true_labels=true_labels,
        display=False,
    )

    img1 = fig_to_array(fig1)
    img2 = fig_to_array(fig2)

    combined_fig, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 2]})

    axs[0].imshow(img1)
    axs[0].axis("off")

    axs[1].imshow(img2)
    axs[1].axis("off")

    plt.tight_layout()

    if display:
        plt.show()
        return None
    else:
        plt.close(combined_fig)
        return combined_fig


def polygon_area(contour):
    """
    Computes the area of a polygon given its contour points using the shoelace formula.

    Args:
        contour: List of (x, y) tuples representing the polygon's vertices.
    """

    x = [p[0] for p in contour]
    y = [p[1] for p in contour]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(contour) - 1)))
