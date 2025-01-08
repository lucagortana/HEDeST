from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
from basics import fig_to_array
from postseg import StdVisualizer


def plot_pie_chart(ax, data, dict_colors, plot_labels=False, add_legend=False):
    """
    Plots a pie chart for the given data on the provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot the pie chart on.
        data (pd.Series): A Series object containing the proportions for each cell type.
        dict_types_colors (dict, optional): A dictionary mapping cell type to a color code.
    """
    labels = data.index
    proportions = data.values

    colors = [dict_colors[cell_type] for cell_type in labels if cell_type in dict_colors.keys()]

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


def plot_legend(dict_colors, ax=None):
    """
    Plot a legend with an optional axis. If `ax` is None, a new figure is created for the legend.

    Parameters:
        dict_colors (dict): Dictionary where keys are identifiers, and values are [label, RGB color].
        ax (matplotlib.axes._axes.Axes, optional): Axis to add the legend to. If None, a new figure is created.
    """

    if isinstance(next(iter(dict_colors.values())), tuple):
        # Format 1: {'fibroblast': (r, g, b, a)}
        legend_labels = list(dict_colors.keys())
        legend_colors = list(dict_colors.values())
    else:
        # Format 2: {'0': ['fibroblast', [r, g, b, a]]}
        legend_labels = [v[0] for v in dict_colors.values()]  # Extract labels
        legend_colors = [tuple(c / 255 for c in v[1]) for v in dict_colors.values()]  # Normalize RGB to [0, 1]

    # Create patches for the legend
    patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=18) for color in legend_colors
    ]

    # If no axis is provided, create a new figure with an axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))  # Adjust `figsize` as needed
        ax.axis("off")  # Disable the axis for a clean legend-only display

    # Add the legend to the axis
    ax.legend(patches, legend_labels, loc="center", fontsize=14)

    # If no axis was provided, show the new figure
    if ax is None:
        plt.show()


def plot_mosaic_cells(spot_dict, image_dict, spot_id=None, predicted_labels=None, num_cols=8, display=True):
    """
    Plots a grid of individual cell images for a given spot_id along with their predicted labels if provided.
    If labels_pred is None, no title is added to the cell images.
    """

    m = 4

    # Select a random spot_id if not provided
    if spot_id is None:
        spot_id = random.choice(list(spot_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    # Get the cell IDs corresponding to the chosen spot_id
    cell_ids = spot_dict[spot_id]

    # Handle case when no cells are found for the spot
    if len(cell_ids) == 0:
        print(f"No individual cells to display for spot_id: {spot_id}")
        return  # Just return if no cells are found, no need to plot cells

    # Calculate the grid dimensions
    num_cells = len(cell_ids)
    num_rows = (num_cells + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Create the mosaic plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, (num_cells // num_cols * m) + m))
    axes = axes.flatten()

    for i, cell_id in enumerate(cell_ids):
        cell_image = image_dict[cell_id].cpu().numpy().transpose(1, 2, 0)  # Convert torch image to numpy

        # Plot the cell image
        axes[i].imshow(cell_image)
        axes[i].axis("off")

        # Add predicted class as title if labels_pred is provided
        if predicted_labels is not None:
            predicted_class = predicted_labels[cell_id]["predicted_class"]
            axes[i].set_title(f"Label: {predicted_class}", color="black")

    # Hide any extra subplots if not enough cells to fill the grid
    for i in range(len(cell_ids), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.close(fig)
        return fig


def plot_cell(image_dict, ax=None, cell_id=None):

    if cell_id is None:
        cell_id = np.random.choice(list(image_dict.keys()))
    else:
        if not isinstance(cell_id, str):
            if isinstance(cell_id, int):
                cell_id = str(cell_id)
            else:
                raise ValueError("cell_id must be either a string or an integer")

    image = image_dict[cell_id].permute(1, 2, 0)

    if ax is None:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(image)
        ax.axis("off")


def plot_history(history_train, history_val, show=False, savefig=None) -> None:
    """
    Plot the training and validation loss history.

    Parameters:
    - history_train (list): A list of training loss values.
    - history_val (list): A list of validation loss values.
    - criterion (str): The criterion used for calculating the loss.
    - show (bool, optional): Whether to display the plot. Defaults to False.
    - savefig (str, optional): The filename to save the plot. Defaults to None.
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


def plot_grid_celltype(predictions, image_dict, cell_type, n=20, selection="random", show_probs=True, display=False):
    """
    Plots a grid with `n` images of cells predicted as a specific cell type, with optional probability labels.

    Args:
        cell_images (dict): A dictionary where each key is a cell ID and each value is a tensor of a cell image.
        predicted_labels_df (DataFrame): A DataFrame of predicted probabilities for each cell type.
        cell_type (str): The cell type to filter images by (e.g., "fibroblast").
        n (int): The number of images to display in the grid.
        selection (str): The selection mode - "max" for top probabilities or "random" for random sampling.
        show_probs (bool): Whether to show the probability on top of each image.
        display (bool): Whether to display the plot directly.

    Returns:
        fig: A matplotlib figure containing the grid.
    """
    # Find cells where `cell_type` has the maximum predicted probability
    max_prob_cell_types = predictions.idxmax(axis=1)
    max_prob_indices = max_prob_cell_types[max_prob_cell_types == cell_type].index
    max_probs = predictions.loc[max_prob_indices, cell_type]

    if selection == "max":
        # Sort by highest probability and take the top `n`
        selected_indices = max_probs.nlargest(n).index
    elif selection == "random":
        # Randomly sample `n` indices
        selected_indices = max_prob_indices.to_series().sample(n=min(n, len(max_prob_indices)))
    else:
        raise ValueError("Invalid selection mode. Choose 'max' or 'random'.")

    # Retrieve images from the dictionary based on selected indices
    selected_images = [image_dict[cell_id] for cell_id in selected_indices if cell_id in image_dict]
    selected_probs = max_probs[selected_indices]

    # Determine grid dimensions (e.g., 4x5 for 20 images)
    num_rows = (n + 9) // 10
    fig, axes = plt.subplots(num_rows, 10, figsize=(15, 2 * num_rows))

    # Plot the selected images
    for i, ax in enumerate(axes.flat):
        if i < len(selected_images):
            img = selected_images[i].cpu().numpy().transpose((1, 2, 0))  # Adjust shape if necessary
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if show_probs:
                prob_text = f"{selected_probs.iloc[i]:.2f}"
                ax.set_title(prob_text, fontsize=8, color="blue")
        else:
            ax.axis("off")  # Hide any remaining empty subplots

    plt.suptitle(cell_type)
    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.close(fig)
        return fig


def plot_predicted_cell_labels_in_spot(
    spot_dict, adata, adata_name, image_path, image_dict, predicted_labels=None, spot_id=None, display=True
):  # dict_cells,
    """
    Plot a spot's visualization along with all cell images arranged in a grid, showing predicted labels.
    Combines the spot and the mosaic of cells into a single figure.
    """

    if spot_id is None:
        spot_id = random.choice(list(spot_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    elif spot_id not in spot_dict:
        raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

    plotter = StdVisualizer(image_path, adata, adata_name)
    fig1 = plotter.plot_specific_spot(spot_id=spot_id, display=False)
    fig2 = plot_mosaic_cells(
        spot_dict=spot_dict, image_dict=image_dict, spot_id=spot_id, predicted_labels=predicted_labels, display=False
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
    else:
        plt.close(combined_fig)
        return combined_fig
