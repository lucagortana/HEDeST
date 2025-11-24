from __future__ import annotations

import random
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def perform_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    verbose: Optional[bool] = False,
) -> np.ndarray:
    """
    Performs UMAP dimensionality reduction on the given embeddings.

    Args:
        embeddings: The input high-dimensional embeddings.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points)
                     used for manifold approximation.
        min_dist: The effective minimum distance between embedded points.
        n_components: The dimension of the space to embed into.
        verbose: Whether to print progress messages.

    Returns:
        The UMAP-reduced embeddings.
    """

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, verbose=verbose)
    umap_embeddings = reducer.fit_transform(embeddings)

    return umap_embeddings


def perform_kmeans(
    embeddings: np.ndarray, n_clusters: int = 5, metric: str = "euclidean", random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs KMeans clustering on the given embeddings.

    Args:
        embeddings: The input embeddings.
        n_clusters: The number of clusters to form.
        metric: The distance metric to use for computing distances.
        random_state: Seed for reproducibility.

    Returns:
        cluster_labels: The labels of each point indicating cluster membership.
        distances: The distances of each point to each cluster centroid.
        centroids: The coordinates of the cluster centroids.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    distances = cdist(embeddings, centroids, metric=metric)

    return cluster_labels, distances, centroids


def find_closest_cells_to_clusters(
    cluster_labels: np.ndarray, distances: np.ndarray, num_per_cluster: int = 5000
) -> np.ndarray:
    """
    Finds the indices of the closest cells to each cluster centroid.

    Args:
        cluster_labels: The labels of each point indicating cluster membership.
        distances: The distances of each point to each cluster centroid.
        num_per_cluster: The number of closest cells to select per cluster.

    Returns:
        The indices of the closest cells to each cluster centroid.
    """

    selected_indices = []
    n_clusters = len(np.unique(cluster_labels))
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_distances = distances[cluster_indices, i]
        closest_indices = cluster_indices[np.argsort(cluster_distances)[:num_per_cluster]]
        selected_indices.extend(closest_indices)

    return selected_indices


def plot_cells_per_cluster(
    image_dict: dict,
    cell_ids: np.ndarray,
    cluster_labels: np.ndarray,
    selection: str = "top",
    nrows: int = 1,
    ncols: int = 1,
    display: bool = False,
) -> Optional[plt.Figure]:
    """
    Plots 8x8 mosaics per cluster in a grid of cluster mosaics (nrows x ncols),
    with thin spacing and properly positioned titles.

    Args:
        image_dict: Dictionary mapping cell IDs to images (as tensors).
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.
        selection: 'top', 'random', or 'bottom' to select which cells to display per cluster.
        nrows: Number of rows in the grid of cluster mosaics.
        ncols: Number of columns in the grid of cluster mosaics.
        display: If True, displays the plot; otherwise returns the figure object.

    Returns:
        If display is False, returns the matplotlib figure object.
    """

    n_clusters = len(np.unique(cluster_labels))
    if n_clusters > nrows * ncols:
        raise ValueError(f"Number of clusters ({n_clusters}) exceeds layout capacity ({nrows}x{ncols})")

    cluster_h = 8  # 8 image rows
    cluster_w = 8  # 8 image columns
    thin_spacer = 0.1

    total_rows = nrows * cluster_h + (nrows - 1)  # Add spacer rows between clusters
    total_cols = ncols * cluster_w + (ncols - 1)  # Add spacer cols between clusters

    # Define thin spacer ratios
    row_heights = []
    for i in range(nrows):
        row_heights.extend([1] * cluster_h)
        if i < nrows - 1:
            row_heights.append(thin_spacer)

    col_widths = []
    for j in range(ncols):
        col_widths.extend([1] * cluster_w)
        if j < ncols - 1:
            col_widths.append(thin_spacer)

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=total_cols,
        figsize=(ncols * 5, nrows * 5),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0, "height_ratios": row_heights, "width_ratios": col_widths},
    )

    if isinstance(axes, np.ndarray):
        axes = axes.reshape(total_rows, total_cols)

    cluster_order = np.unique(cluster_labels)

    for cluster_idx, cluster in enumerate(cluster_order):
        cluster_row = cluster_idx // ncols
        cluster_col = cluster_idx % ncols

        cluster_indices = np.where(cluster_labels == cluster)[0]

        if selection == "top":
            selected_indices = cluster_indices[: min(len(cluster_indices), 64)]
        elif selection == "random":
            selected_indices = np.random.choice(cluster_indices, size=min(len(cluster_indices), 64), replace=False)
        elif selection == "bottom":
            selected_indices = cluster_indices[-min(len(cluster_indices), 64) :]
        else:
            raise ValueError("selection must be 'top', 'random', or 'bottom'.")

        selected_ids = cell_ids[selected_indices]

        # Calculate the top-left position of the cluster block
        row_start = cluster_row * (cluster_h + 1)
        col_start = cluster_col * (cluster_w + 1)

        for i, cell_id in enumerate(selected_ids):
            r = i // 8
            c = i % 8
            ax = axes[row_start + r, col_start + c]
            img = image_dict[cell_id].numpy().transpose(1, 2, 0)
            ax.imshow(img, cmap="gray")
            ax.axis("off")

    # Turn off unused axes
    for r in range(total_rows):
        for c in range(total_cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01)
    if display:
        plt.show()
        return None
    else:
        plt.close(fig)
        return fig


def create_bags(
    cell_ids: np.ndarray,
    cluster_labels: np.ndarray,
    mean_cell_per_bag: int = 10,
    var_cell_per_bag: int = 4,
    balance: Union[List[int], str] = "auto",
    total_number_of_bags: int = 1000,
    random_state: Optional[int] = None,
    not_mixed: Optional[List[int]] = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Creates bags of cell IDs with controlled sampling based on cluster proportions,
    with optional constraint to not mix certain clusters.

    Args:
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.
        mean_cell_per_bag: Mean number of cells per bag.
        var_cell_per_bag: Variance in the number of cells per bag.
        balance: List of weights representing the desired overall distribution of clusters.
               If "auto", the cluster proportions will be used. Default is "auto".
        total_number_of_bags: Total number of bags to create.
        random_state: Seed for reproducibility.
        not_mixed: List of two cluster indices that should not appear together in any bag.

    Returns:
        bags: List of lists, where each inner list contains the cell IDs for a bag.
        cells_per_cluster: Array with the total number of sampled cells per cluster.
    """

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    if not isinstance(balance, list):
        if balance == "auto":
            balance = np.bincount(cluster_labels)
        else:
            raise ValueError("balance must be 'auto' or a list of weights.")
    else:
        if len(balance) != len(np.unique(cluster_labels)):
            raise ValueError("balance must have the same length as the number of clusters.")
        balance = np.array(balance)

    balance = balance / balance.sum()
    print(f"Using cluster proportions: {balance}")

    unique_clusters = np.unique(cluster_labels)
    cluster_to_indices = {cluster: np.where(cluster_labels == cluster)[0] for cluster in unique_clusters}

    total_cells = int(mean_cell_per_bag * total_number_of_bags)
    cells_per_cluster = (balance * total_cells).astype(int)

    sampled_cells_by_cluster = {}
    for cluster, count in zip(unique_clusters, cells_per_cluster):
        indices = cluster_to_indices[cluster]
        if count <= len(indices):
            sampled_indices = np.random.choice(indices, size=count, replace=False)
        else:
            sampled_indices = np.random.choice(indices, size=count, replace=True)
        sampled_cells_by_cluster[cluster] = list(cell_ids[sampled_indices])

    bags = []

    for _ in range(total_number_of_bags):
        n_cells = int(np.random.normal(mean_cell_per_bag, np.sqrt(var_cell_per_bag)))
        n_cells = max(1, n_cells)

        bag = []
        available_clusters = list(unique_clusters)
        random.shuffle(available_clusters)

        while len(bag) < n_cells and available_clusters:
            cluster = random.choice(available_clusters)

            # Check not_mixed constraint
            if not_mixed is not None:
                if (
                    not_mixed[0] in [cluster_labels[cell_ids == cid][0] for cid in bag] and cluster == not_mixed[1]
                ) or (not_mixed[1] in [cluster_labels[cell_ids == cid][0] for cid in bag] and cluster == not_mixed[0]):
                    available_clusters.remove(cluster)
                    continue

            if sampled_cells_by_cluster[cluster]:
                bag.append(sampled_cells_by_cluster[cluster].pop())

            # Remove the cluster from future sampling if it's empty
            if not sampled_cells_by_cluster[cluster]:
                available_clusters.remove(cluster)

        bags.append(bag)

    random.shuffle(bags)

    return bags, cells_per_cluster


def get_bag_proportions(bags: List[List[int]], cell_ids: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Calculates the cluster proportions for each bag.

    Args:
        bags: List of lists, where each inner list contains the cell IDs for a bag.
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.

    Returns:
        bag_prop_df: DataFrame with shape (n_bags, n_clusters) containing cluster proportions per bag.
    """

    id_to_cluster = {}
    for cell_id, cluster in zip(cell_ids, cluster_labels):
        id_to_cluster[cell_id] = cluster

    bag_cluster_proportions = {}

    for bag_id, bag in enumerate(bags):
        clusters_in_bag = [id_to_cluster[cell_id] for cell_id in bag]
        unique_clusters, counts = np.unique(clusters_in_bag, return_counts=True)
        proportions = counts / len(bag)
        bag_cluster_proportions[bag_id] = dict(zip(unique_clusters, proportions))

    bag_prop_df = pd.DataFrame(
        index=sorted(bag_cluster_proportions.keys()), columns=sorted(set(id_to_cluster.values()))
    )

    for bag_id, proportions in bag_cluster_proportions.items():
        for cluster_id, proportion in proportions.items():
            bag_prop_df.at[bag_id, cluster_id] = proportion

    bag_prop_df = bag_prop_df.fillna(0)
    bag_prop_df.columns = [f"Cluster {cluster_id}" for cluster_id in bag_prop_df.columns]
    bag_prop_df.index = bag_prop_df.index.astype(str)

    return bag_prop_df


def add_perturbation(proportions_df: pd.DataFrame, strength: float) -> pd.DataFrame:
    """
    Adds perturbation to a proportion dataframe.

    Args:
        proportions_df: pd.DataFrame with shape (n_spots, n_cell_types), each row sums to 1
        strength: float in [0, 1], 0 = no perturbation, 1 = full random

    Returns:
        perturbed_df: pd.DataFrame with same shape, rows still sum to 1
    """

    if not (0 <= strength <= 1):
        raise ValueError("strength must be between 0 and 1")

    noise = np.random.rand(*proportions_df.shape)
    mixed = (1 - strength) * proportions_df.values + strength * noise
    mixed /= mixed.sum(axis=1, keepdims=True)

    perturbed_df = pd.DataFrame(mixed, index=proportions_df.index, columns=proportions_df.columns)

    return perturbed_df


def plot_cluster_distribution(
    filtered_labels: np.ndarray,
    n_cells_per_cluster: np.ndarray,
    title: str = "",
    savefig: str = None,
    context: str = "talk",
) -> None:
    """
    Plots a bar chart of cell counts per cluster using Seaborn.

    Args:
        filtered_labels: Cluster labels.
        n_cells_per_cluster: Cell counts per cluster.
        title: Title for the plot.
        savefig: If provided, saves the figure to this path.
        context: Seaborn context (e.g., 'paper', 'notebook', 'talk', 'poster').
    """

    sns.set_context(context)
    sns.set_style("whitegrid")

    cluster_ids = np.unique(filtered_labels)
    cluster_names = [f"Cluster {i}" for i in cluster_ids]

    df = pd.DataFrame({"Cluster": cluster_names, "Cell Count": n_cells_per_cluster})

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Cluster", y="Cell Count", palette="viridis")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Cells")
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {savefig}")

    plt.show()


def plot_bags(
    bags: list,
    cell_ids: list,
    cluster_labels: list,
    title: str = "",
    fig_edge_size: int = 20,
    cmap: str = "viridis",
    savefig: str = None,
    dpi: int = 300,
) -> None:
    """
    Visualizes spatial 'bags' of cells, where each bag contains cells colored by cluster.
    Uses Seaborn for styling and Matplotlib for drawing.

    Args:
        bags: Each sublist contains cell IDs belonging to a bag.
        cell_ids: List or array of cell IDs corresponding to cluster_labels.
        cluster_labels: Cluster assignment for each cell ID.
        title: Title for the figure.
        fig_edge_size: Size of the figure edge (controls overall scale).
        cmap: Matplotlib or Seaborn colormap (default: 'viridis').
        savefig: Path to save the figure. If None, the plot is shown.
        dpi: Resolution for saving figures.
    """

    # ---- Seaborn style ----
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # Map each cell ID to its cluster
    id_to_cluster = dict(zip(cell_ids, cluster_labels))
    unique_clusters = np.unique(cluster_labels)

    # Fixed Seaborn palette from viridis
    colors = sns.color_palette(cmap, len(unique_clusters))

    # ---- Layout parameters ----
    num_bags = len(bags)
    grid_size = int(np.ceil(np.sqrt(num_bags)))

    bag_radius = 0.4 * (fig_edge_size / 20) / float(grid_size ** (1 / 3)) * 3
    cell_size = 20 * (fig_edge_size / 20) / grid_size * 32

    fig, ax = plt.subplots(figsize=(fig_edge_size, fig_edge_size))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=22, fontweight="bold")

    # ---- Plot bags and cells ----
    for bag_id, bag in enumerate(bags):
        row = bag_id // grid_size
        col = bag_id % grid_size
        x = col
        y = grid_size - 1 - row

        # Bag boundary
        circle = Circle((x, y), radius=bag_radius, edgecolor="black", facecolor="none", lw=1.5)
        ax.add_patch(circle)

        # Plot each cell in the bag
        for cell_id in bag:
            cluster_id = id_to_cluster[cell_id]
            cluster_idx = np.where(unique_clusters == cluster_id)[0][0]
            color = colors[cluster_idx]

            # Random position within bag
            angle = 2 * np.pi * np.random.rand()
            radius = bag_radius * np.sqrt(np.random.rand())
            cell_x = x + radius * np.cos(angle)
            cell_y = y + radius * np.sin(angle)

            ax.scatter(cell_x, cell_y, color=color, s=cell_size, edgecolor="black", linewidth=0.4)

    # ---- Legend ----
    for cluster_id, color in zip(unique_clusters, colors):
        ax.scatter([], [], color=color, label=f"Cluster {cluster_id}")
    legend = ax.legend(title="", loc="lower right", fontsize=14, title_fontsize=16, frameon=True)
    legend.get_frame().set_edgecolor("black")

    # ---- Final touches ----
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {savefig}")
    else:
        plt.show()
