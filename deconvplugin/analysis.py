from __future__ import annotations

import pickle
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score

from deconvplugin.basics import fig_to_array
from deconvplugin.basics import generate_color_dict
from deconvplugin.basics import require_attributes
from deconvplugin.plots import plot_grid_celltype
from deconvplugin.plots import plot_history
from deconvplugin.plots import plot_legend
from deconvplugin.plots import plot_mosaic_cells
from deconvplugin.plots import plot_pie_chart
from deconvplugin.postseg import StdVisualizer


class PredAnalyzer:
    EXPECTED_VARIABLES = {
        "mtype",
        "spot_dict",
        "proportions",
        "history",
        "preds",
        "image_dict",
        "image_path",
        "adata",
        "adata_name",
        "seg_dict",
    }

    def __init__(self, model_state="best", adjusted=False, model_info=None, **kwargs):
        """
        Initialize PredAnalyzer with variables from:
        - A dictionary.
        - A pickle file path.
        - A binary pickle object.
        - Direct keyword arguments.

        Args:
            info (dict, str, or None): Either:
                - A dictionary with variable data.
                - A path to a pickle file.
                - A binary pickle object.
            **kwargs: Directly specified variables (override info values if provided).
        """
        self.seg_dict_w_class = None
        self.model_state = model_state
        self.adjusted = adjusted
        self.info = {}

        # Load data from pickle if provided
        if model_info:
            if isinstance(model_info, dict):
                self.info = model_info
            elif isinstance(model_info, str):
                with open(model_info, "rb") as file:
                    self.info = pickle.load(file)
            else:
                self.info = pickle.load(model_info)

        # Update with kwargs
        self.info.update(kwargs)

        unexpected_variables = set(self.info.keys()) - self.EXPECTED_VARIABLES
        if unexpected_variables:
            raise ValueError(
                f"Unexpected keys: {unexpected_variables}. " f"Expected keys are: {self.EXPECTED_VARIABLES}"
            )

        # Attribuer dynamiquement les valeurs
        for key in self.EXPECTED_VARIABLES:
            setattr(self, key, self.info.get(key, None))

        assert self.preds is not None, "The 'preds' attribute must be provided and cannot be None."

        if self.model_state == "best":
            if self.adjusted:
                self.predictions = self.preds["pred_best_adjusted"]
            else:
                self.predictions = self.preds["pred_best"]

        elif self.model_state == "final":
            try:
                if self.adjusted:
                    self.predictions = self.preds["pred_final_adjusted"]
                else:
                    self.predictions = self.preds["pred_final"]
            except KeyError:
                raise KeyError("No final model found in the 'preds' attribute.")

        else:
            raise ValueError("Invalid model state. Choose 'best' or 'final'.")

        self.ct_list = list(self.predictions.columns)
        self.color_dict = generate_color_dict(self.ct_list, format="special")

        print("Loading predicted labels...")
        self.predicted_labels = self._get_labels_slide()
        print("-> ok")

        if self.history is not None:
            self.history_train = self.history["train"]
            self.history_val = self.history["val"]

        else:
            print("Warning : No history provided. You won't be able to plot the training and validation histories.")
            print("Use `add_attributes(history=your_history)` to add one.")

        if self.seg_dict is not None:
            self._generate_dicts_viz_pred()

        else:
            print("Warning : No segmentation provided. You won't be able to plot the segmentation.")
            print("Use `add_attributes(seg_dict=your_seg_dict)` to add one.")

    def __repr__(self):
        attrs = ", ".join(f"{k}={getattr(self, k, None)}" for k in self.EXPECTED_VARIABLES)
        return f"PredAnalyzer({attrs})"

    @classmethod
    def expected_variables(cls):
        """Retourne les clés attendues."""
        return cls.EXPECTED_VARIABLES

    def add_attributes(self, **kwargs):
        """
        Ajoute dynamiquement des attributs à l'instance uniquement si
        ces attributs sont dans expected_variables.

        Args:
            **kwargs: Noms et valeurs des attributs à ajouter.

        Raise:
            ValueError: Si une clé ne figure pas dans expected_variables.
        """
        for key, value in kwargs.items():
            if key not in self.EXPECTED_VARIABLES:
                raise ValueError(
                    f"Cannot add attribute '{key}', " f"it is not in the expected keys: {self.EXPECTED_VARIABLES}"
                )

            elif key == "history":
                self.history_train = value["train"]
                self.history_val = value["val"]

            elif key == "seg_dict":
                self._generate_dicts_viz_pred(value)

            setattr(self, key, value)

    def list_attributes(self):
        """
        Retourne tous les attributs actuels de l'instance.
        """
        return {key: getattr(self, key, None) for key in self.EXPECTED_VARIABLES}

    def extract_stats(self, metric="predicted"):
        """
        Extrait des statistiques sur les prédictions à partir d'un DataFrame de probabilités.

        Args:
            Tout refaire

        Returns:
            pd.DataFrame: DataFrame avec les statistiques pour chaque classe.
        """
        stats = {}
        ct_list = list(self.predictions.columns)

        if metric == "predicted":
            for cell_id, pred in self.predicted_labels.items():
                max_prob = self.predictions.loc[cell_id].max()

                if pred["cell_type"] not in stats:
                    stats[pred["cell_type"]] = {"probs": [], "count": 0}

                stats[pred["cell_type"]]["probs"].append(max_prob)
                stats[pred["cell_type"]]["count"] += 1

            data = []
            for ct, class_stats in stats.items():
                class_probs = class_stats["probs"]
                row = [
                    self.predictions.columns.get_loc(ct),
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

            for _, prob_vector in self.predictions.iterrows():
                for class_id, prob in enumerate(prob_vector):
                    stats[ct_list[class_id]].append(prob)

            data = []
            for ct, class_probs in stats.items():
                row = [
                    self.predictions.columns.get_loc(ct),
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

    def plot_history(self, show=False, savefig=None):
        return plot_history(self.history_train, self.history_val, show=show, savefig=savefig)

    def find_cell_max_cell_type(self, cell_type):
        if cell_type not in self.ct_list:
            raise ValueError(f"Cell type '{cell_type}' not found in the prediction DataFrame.")
        max_cell_id = self.predictions[cell_type].idxmax()
        return max_cell_id

    @require_attributes("spot_dict", "image_dict")
    def plot_mosaic_cells(self, spot_id=None, num_cols=8, display=True):
        """
        Plots a grid of individual cell images for a given spot_id along with their predicted labels if provided.
        If labels_pred is None, no title is added to the cell images.
        """

        return plot_mosaic_cells(
            self.spot_dict,
            self.image_dict,
            spot_id=spot_id,
            predicted_labels=self.predicted_labels,
            num_cols=num_cols,
            display=display,
        )

    @require_attributes("spot_dict", "image_dict", "image_path", "adata", "adata_name")
    def plot_predicted_cell_labels_in_spot(self, spot_id=None, display=True):  # dict_cells,
        """
        Plot a spot's visualization along with all cell images arranged in a grid, showing predicted labels.
        Combines the spot and the mosaic of cells into a single figure.
        """

        if spot_id is None:
            spot_id = random.choice(list(self.spot_dict.keys()))
            print(f"Randomly selected spot_id: {spot_id}")

        elif spot_id not in self.spot_dict:
            raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

        # Générer les deux figures (le spot et la mosaïque)
        plotter = StdVisualizer(self.image_path, self.adata, self.adata_name)
        fig1 = plotter.plot_specific_spot(spot_id=spot_id, display=False)
        fig2 = self.plot_mosaic_cells(spot_id=spot_id, display=False)

        img1 = fig_to_array(fig1)
        img2 = fig_to_array(fig2)

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

    @require_attributes("spot_dict", "proportions", "image_path", "adata", "adata_name")
    def plot_spot_proportions(self, spot_id=None, draw_seg=False):
        if draw_seg:
            if self.seg_dict_w_class is None:
                raise ValueError("You must run `generate_dicts_viz_pred` before to be able to plot segmentation.")

        if spot_id is None:
            spot_id = random.choice(list(self.spot_dict.keys()))
            print(f"Randomly selected spot_id: {spot_id}")

        elif spot_id not in self.spot_dict:
            raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

        # Gridspec layout
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

        # Spot image
        ax0 = fig.add_subplot(gs[:, 0])
        plotter = StdVisualizer(
            self.image_path,
            self.adata,
            self.adata_name,
            [None, self.seg_dict_w_class][draw_seg],
            [None, self.color_dict][draw_seg],
        )

        fig1 = plotter.plot_specific_spot(spot_id=spot_id, display=False)
        img1 = fig_to_array(fig1)
        ax0.imshow(img1)
        ax0.axis("off")

        list_cells = self.spot_dict[spot_id]

        dict_pie_colors = generate_color_dict(self.ct_list, format="classic")

        # mean predicted probabilities
        ax1 = fig.add_subplot(gs[0, 1])
        mean_prob_ct = self.predictions[self.predictions.index.isin(list_cells)].mean(axis=0)
        plot_pie_chart(ax1, mean_prob_ct, dict_colors=dict_pie_colors)
        ax1.set_title("Mean Predicted Probabilities")

        # predicted cell type proportions
        ax2 = fig.add_subplot(gs[0, 2])
        prop_ct = self.predictions[self.predictions.index.isin(list_cells)].idxmax(axis=1).value_counts() / len(
            list_cells
        )
        plot_pie_chart(ax2, prop_ct, dict_colors=dict_pie_colors)
        ax2.set_title("Predicted Cell Type Proportions")

        # true cell type proportions
        ax3 = fig.add_subplot(gs[1, 1])
        true_prop = self.proportions.loc[spot_id]
        plot_pie_chart(ax3, true_prop, dict_colors=dict_pie_colors)
        ax3.set_title("True Cell Type Proportions")

        # legend
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        plot_legend(ax4, dict_pie_colors)
        ax4.set_title("Legend")

        plt.tight_layout()
        plt.show()

    @require_attributes("proportions", "spot_dict")
    def evaluate_spot_predictions(self):
        """
        Compare the true and predicted proportions using multiple metrics.

        Parameters:
        - true_proportions (pd.DataFrame): DataFrame with true proportions per spot.
        - predicted_proportions (pd.DataFrame): DataFrame with predicted proportions per spot.

        Returns:
        - dict: A dictionary with the calculated metrics.
        """

        predicted_proportions = self.get_predicted_proportions()

        # Align dataframes and extracting labels
        true_proportions, predicted_proportions = self.proportions.align(predicted_proportions, join="inner", axis=0)
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

    @require_attributes("image_dict")
    def plot_grid_celltype(self, cell_type, n=20, selection="random", show_probs=True, display=False):
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

        return plot_grid_celltype(
            self.predictions,
            self.image_dict,
            cell_type,
            n=n,
            selection=selection,
            show_probs=show_probs,
            display=display,
        )

    @require_attributes("spot_dict")
    def get_predicted_proportions(self):
        """
        Get the predicted proportions of cell types for each spot in the dataset.
        """

        cell_to_spot = {cell: spot for spot, cells in self.spot_dict.items() for cell in cells}
        spot_series = pd.Series(cell_to_spot, name="spot")

        pred_with_spot = self.predictions.join(spot_series, how="inner")

        proportion_df = pred_with_spot.groupby("spot").mean()

        return proportion_df

    def _get_labels_slide(self):
        predicted_classes = self.predictions.idxmax(axis=1)

        predicted_class_indices = predicted_classes.map(self.predictions.columns.get_loc)

        predicted_labels = {
            cell_id: {"predicted_class": cls_idx, "cell_type": cls}
            for cell_id, cls_idx, cls in zip(self.predictions.index, predicted_class_indices, predicted_classes)
        }

        return predicted_labels

    def _generate_dicts_viz_pred(self, seg_dict):
        """
        Updates the nuclear dictionary with the predicted classes.
        """

        # self.seg_dict_w_class = deepcopy(seg_dict)

        # for i, (key, _) in enumerate(self.seg_dict_w_class["nuc"].items()):
        #     if str(i) in self.predicted_labels:
        #         self.seg_dict_w_class["nuc"][key]["type"] = self.predicted_labels[str(i)]["predicted_class"]

        self.seg_dict_w_class = {
            "nuc": {
                key: {
                    **value,
                    "type": self.predicted_labels[str(i)]["predicted_class"]
                    if str(i) in self.predicted_labels
                    else value.get("type", None),
                }
                for i, (key, value) in enumerate(seg_dict["nuc"].items())
            }
        }
