from __future__ import annotations

import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from tools import basics
from tools import slide_viz
from tqdm import tqdm


def predict_slide(model, image_dict, ct_list, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device)

    model.eval()
    model = model.to(device)
    predictions = []

    dataloader = torch.utils.data.DataLoader(list(image_dict.items()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on cells", unit="batch"):
            cell_ids, images = batch
            images = images.to(device).float() / 255.0

            outputs = model(images)

            for cell_id, prob_vector in zip(cell_ids, outputs):
                predictions.append(
                    {
                        "cell_id": cell_id,
                        **{ct_list[i]: prob for i, prob in enumerate(prob_vector.cpu().tolist())},
                    }
                )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.set_index("cell_id", inplace=True)

    return predictions_df


def get_labels_slide(predictions_df):
    predicted_labels = {}

    for cell_id, row in predictions_df.iterrows():
        predicted_class = row.idxmax()

        predicted_labels[cell_id] = {
            "predicted_class": predictions_df.columns.get_loc(predicted_class),
            "cell_type": predicted_class,
        }

    return predicted_labels


def extract_stats(predictions_df, predicted_labels=None, metric="predicted"):
    """
    Extrait des statistiques sur les prédictions à partir d'un DataFrame de probabilités.

    Args:
        predictions_df (pd.DataFrame): DataFrame contenant les probabilités pour chaque cellule (index = cell_id).
        ct_list (list): Liste des types cellulaires correspondants à chaque classe.
        metric (str): Soit 'predicted' pour les stats des classes prédites, soit 'all' pour les stats de
        toutes les classes.

    Returns:
        pd.DataFrame: DataFrame avec les statistiques pour chaque classe.
    """
    stats = {}
    ct_list = list(predictions_df.columns)

    if predicted_labels is None:
        predicted_labels = get_labels_slide(predictions_df)

    if metric == "predicted":
        for cell_id, pred in predicted_labels.items():
            max_prob = predictions_df.loc[cell_id].max()

            if pred["cell_type"] not in stats:
                stats[pred["cell_type"]] = {"probs": [], "count": 0}

            stats[pred["cell_type"]]["probs"].append(max_prob)
            stats[pred["cell_type"]]["count"] += 1

        data = []
        for ct, class_stats in stats.items():
            class_probs = class_stats["probs"]
            row = [
                predictions_df.columns.get_loc(ct),
                ct,
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs),
                class_stats["count"],
            ]
            data.append(row)

        columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob", "Cell Count"]
        df_stats = pd.DataFrame(data, columns=columns)

    elif metric == "all":
        for ct in ct_list:
            stats[ct] = []

        for _, prob_vector in predictions_df.iterrows():
            for class_id, prob in enumerate(prob_vector):
                stats[ct_list[class_id]].append(prob)

        data = []
        for ct, class_probs in stats.items():
            row = [
                predictions_df.columns.get_loc(ct),
                ct,
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs),
            ]
            data.append(row)

        columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob"]
        df_stats = pd.DataFrame(data, columns=columns)

    else:
        raise ValueError("Invalid metric. Choose 'predicted' or 'all'.")

    df_stats = df_stats.sort_values(by="Class", ascending=True).reset_index(drop=True)
    df_stats = df_stats.set_index("Class")

    return df_stats


def generate_dicts_viz_pred(nuc_dict, labels_pred, ct_list):
    """
    Updates the nuclear dictionary with the predicted classes.
    """
    for i, (key, _) in enumerate(nuc_dict["nuc"].items()):
        if str(i) in labels_pred:
            nuc_dict["nuc"][key]["type"] = labels_pred[str(i)]["predicted_class"]

    color_dict = generate_color_dict(ct_list)

    return nuc_dict, color_dict


def generate_color_dict(list, palette="hsv"):
    """
    Generate a dictionary of colors for each class in the list.
    """
    color_dict = {}
    num_classes = len(list)

    cmap = plt.get_cmap(palette, num_classes)

    for i, class_name in enumerate(list):
        color = cmap(i)[:3]
        color = [int(255 * c) for c in color]
        color_dict[str(i)] = [class_name, color]

    return color_dict


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


def find_cell_max_cell_type(cell_type, pred):
    if cell_type not in pred.columns:
        raise ValueError(f"Cell type '{cell_type}' not found in the prediction DataFrame.")
    max_cell_id = pred[cell_type].idxmax()
    return max_cell_id


def plot_mosaic_cells(spots_dict, cell_images, labels_pred=None, spot_id=None, num_cols=8, display=True):
    """
    Plots a grid of individual cell images for a given spot_id along with their predicted labels if provided.
    If labels_pred is None, no title is added to the cell images.
    """
    if labels_pred is not None:
        m = 4
    else:
        m = 3

    # Select a random spot_id if not provided
    if spot_id is None:
        spot_id = random.choice(list(spots_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    # Get the cell IDs corresponding to the chosen spot_id
    cell_ids = spots_dict[spot_id]

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
        cell_image = cell_images[cell_id].cpu().numpy().transpose(1, 2, 0)  # Convert torch image to numpy

        # Plot the cell image
        axes[i].imshow(cell_image)
        axes[i].axis("off")

        # Add predicted class as title if labels_pred is provided
        if labels_pred is not None:
            predicted_class = labels_pred[cell_id]["predicted_class"]
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


def plot_predicted_cell_labels_in_spot(
    spots_dict, image_path, adata, adata_name, cell_images, labels_pred=None, spot_id=None, display=True  # dict_cells,
):
    """
    Plot a spot's visualization along with all cell images arranged in a grid, showing predicted labels.
    Combines the spot and the mosaic of cells into a single figure.
    """

    # if isinstance(dict_cells, str) and os.path.isfile(dict_cells):
    #     with open(dict_cells) as json_file:
    #         data = json.load(json_file)
    # elif isinstance(dict_cells, dict) or dict_cells is None:
    #     data = dict_cells
    # else:
    #     raise ValueError("dict_cells must be a path to a JSON file or a dictionary.")

    if spot_id is None:
        spot_id = random.choice(list(spots_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    elif spot_id not in spots_dict:
        raise ValueError(f"Spot ID {spot_id} not found in spots_dict.")

    # Générer les deux figures (le spot et la mosaïque)
    fig1 = slide_viz.plot_specific_spot(image_path, adata, adata_name, spot_id=spot_id, display=False)
    fig2 = plot_mosaic_cells(spots_dict, cell_images, labels_pred, spot_id=spot_id, display=False)

    img1 = basics.fig_to_array(fig1)
    img2 = basics.fig_to_array(fig2)

    # Création d'une nouvelle figure pour combiner les deux
    combined_fig, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 2]})

    # Dessiner la première figure (le spot) dans le premier axe
    axs[0].imshow(img1)  # Utilise l'image de `fig1`
    axs[0].axis("off")

    # Dessiner la deuxième figure (mosaïque des cellules) dans le second axe
    axs[1].imshow(img2)  # Utilise l'image de `fig2`
    axs[1].axis("off")

    # Ajuster l'agencement de la figure
    plt.tight_layout()

    # Afficher ou retourner la figure
    if display:
        plt.show()
    else:
        plt.close(combined_fig)
        return combined_fig


def plot_spots_proportions(
    image_path, adata, adata_name, proportions, spot_dict, pred, dict_cells=None, dict_types_colors=None, spot_id=None
):
    if spot_id is None:
        spot_id = random.choice(list(spot_dict.keys()))
        print(f"Randomly selected spot_id: {spot_id}")

    elif spot_id not in spot_dict:
        raise ValueError(f"Spot ID {spot_id} not found in spots_dict.")

    # Gridspec layout
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

    # Spot image
    ax0 = fig.add_subplot(gs[:, 0])
    fig1 = slide_viz.plot_specific_spot(
        image_path,
        adata,
        adata_name,
        spot_id=spot_id,
        dict_cells=dict_cells,
        dict_types_colors=dict_types_colors,
        display=False,
    )
    img1 = basics.fig_to_array(fig1)
    ax0.imshow(img1)
    ax0.axis("off")

    list_cells = spot_dict[spot_id]
    if dict_types_colors is not None:
        dict_pie_colors = {item[0]: np.array(item[1]) / 255.0 for _, item in dict_types_colors.items()}
    else:
        cmap = plt.get_cmap("tab20")
        dict_pie_colors = {proportions.columns[i]: cmap(i % 20) for i in range(len(proportions.columns))}

    # mean predicted probabilities
    ax1 = fig.add_subplot(gs[0, 1])
    mean_prob_ct = pred[pred.index.isin(list_cells)].mean(axis=0)
    plot_pie_chart(ax1, mean_prob_ct, dict_colors=dict_pie_colors)
    ax1.set_title("Mean Predicted Probabilities")

    # predicted cell type proportions
    ax2 = fig.add_subplot(gs[0, 2])
    prop_ct = pred[pred.index.isin(list_cells)].idxmax(axis=1).value_counts() / len(list_cells)
    plot_pie_chart(ax2, prop_ct, dict_colors=dict_pie_colors)
    ax2.set_title("Predicted Cell Type Proportions")

    # true cell type proportions
    ax3 = fig.add_subplot(gs[1, 1])
    true_prop = proportions.loc[spot_id]
    plot_pie_chart(ax3, true_prop, dict_colors=dict_pie_colors)
    ax3.set_title("True Cell Type Proportions")

    # legend
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    plot_legend(ax4, dict_pie_colors)
    ax4.set_title("Legend")

    plt.tight_layout()
    plt.show()


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


def plot_legend(ax, dict_colors):
    legend_labels = [label for label in dict_colors.keys()]
    legend_colors = [dict_colors[label] for label in legend_labels]
    patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=18) for color in legend_colors
    ]

    # Add the legend to the new subplot
    ax.legend(patches, legend_labels, loc="center", fontsize=14)


def get_predicted_proportions(pred, spot_dict):
    """
    Get the predicted proportions of cell types for each spot in the dataset.
    """

    cell_to_spot = {cell: spot for spot, cells in spot_dict.items() for cell in cells}
    spot_series = pd.Series(cell_to_spot, name="spot")

    pred_with_spot = pred.join(spot_series, how="inner")

    proportion_df = pred_with_spot.groupby("spot").mean()

    return proportion_df


def evaluate_spot_predictions(true_proportions, predicted_proportions):
    """
    Compare the true and predicted proportions using multiple metrics.

    Parameters:
    - true_proportions (pd.DataFrame): DataFrame with true proportions per spot.
    - predicted_proportions (pd.DataFrame): DataFrame with predicted proportions per spot.

    Returns:
    - dict: A dictionary with the calculated metrics.
    """
    # Align dataframes and extracting labels
    true_proportions, predicted_proportions = true_proportions.align(predicted_proportions, join="inner", axis=0)
    true_labels = true_proportions.idxmax(axis=1)
    predicted_labels = predicted_proportions.idxmax(axis=1)

    # Computing weights
    class_frequencies = true_proportions.mean(axis=0)
    class_weights = 1 / class_frequencies
    class_weights /= class_weights.sum()
    weights = np.array([class_weights[col] for col in true_proportions.columns])

    # mse
    squared_errors = (true_proportions - predicted_proportions) ** 2
    weighted_mse = (squared_errors * weights).mean().mean()
    mse = squared_errors.mean().mean()

    # mae
    absolute_errors = (true_proportions - predicted_proportions).abs()
    weighted_mae = (absolute_errors * weights).mean().mean()
    mae = absolute_errors.mean().mean()

    # R^2 score
    r2 = r2_score(true_proportions, predicted_proportions)

    # Pearson and Spearman correlation
    spearman_corrs = []
    pearson_corrs = []
    for spot in true_proportions.index:
        true_values = true_proportions.loc[spot]
        pred_values = predicted_proportions.loc[spot]

        spearman_corr, _ = spearmanr(true_values, pred_values)
        spearman_corrs.append(spearman_corr)

        pearson_corr, _ = pearsonr(true_values, pred_values)
        pearson_corrs.append(pearson_corr)

    avg_spearman_corr = np.nanmean(spearman_corrs)
    avg_pearson_corr = np.nanmean(pearson_corrs)

    # balanced accuracy
    balanced_acc = balanced_accuracy_score(
        true_labels, predicted_labels
    )  # sometimes warnings : classes in y_pred not in y_true

    # F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    # Precision
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    # Recall
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    metrics = {
        "Spearman Correlation": avg_spearman_corr,
        "Pearson Correlation": avg_pearson_corr,
        "Weighted MSE": weighted_mse,
        "MSE": mse,
        "Weighted MAE": weighted_mae,
        "MAE": mae,
        "R^2 Score": r2,
        "Balanced Accuracy": balanced_acc,
        "Weighted F1 Score": f1,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
    }

    return metrics


def evaluate_performance(
    table: pd.DataFrame, feature_a, feature_b, metric: str, na_fill=None, **fixed_features
) -> pd.DataFrame:
    # Convert feature_a and feature_b to lists if they are not already
    if not isinstance(feature_a, list):
        feature_a = [feature_a]
    if not isinstance(feature_b, list):
        feature_b = [feature_b]

    # Check if metric exists in table
    if metric not in table.columns:
        raise ValueError(f"Metric '{metric}' not found in the table columns.")

    # Filter the DataFrame based on fixed feature values
    for feature, value in fixed_features.items():
        if feature not in table.columns:
            raise ValueError(f"Feature '{feature}' not found in the table columns.")
        table = table[table[feature] == value]

    # Check if filtering resulted in an empty DataFrame
    if table.empty:
        raise ValueError("No data left after filtering; please check fixed feature values.")

    if na_fill is not None:
        table[metric] = table[metric].fillna(na_fill)

    # Group by feature_a and feature_b, then calculate the mean of the specified metric
    grouped = table.groupby(feature_a + feature_b)[metric].mean()

    # Unstack the feature_b parameters to create a multi-level column index
    performance_df = grouped.unstack(level=feature_b)

    return performance_df
