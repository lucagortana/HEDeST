from __future__ import annotations

import pickle
import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from hedest.analysis.plots import plot_grid_celltype
from hedest.analysis.plots import plot_history
from hedest.analysis.plots import plot_legend
from hedest.analysis.plots import plot_mosaic_cells
from hedest.analysis.plots import plot_pie_chart
from hedest.analysis.plots import plot_predicted_cell_labels_in_spot
from hedest.analysis.postseg import StdVisualizer
from hedest.utils import fig_to_array
from hedest.utils import generate_color_dict
from hedest.utils import require_attributes


class PredAnalyzer:
    """
    A class to analyze predictions made by a cell classifier.
    """

    EXPECTED_VARIABLES = {
        "model_name",
        "hidden_dims",
        "spot_dict",
        "train_spot_dict",
        "proportions",
        "history",
        "preds",
        "image_dict",
        "image_path",
        "adata",
        "adata_name",
        "seg_dict",
        "ground_truth",
    }

    def __init__(
        self, model_state: str = "best", adjusted: bool = False, model_info: Optional[Union[dict, str]] = None, **kwargs
    ):
        """
        Initialize PredAnalyzer with variables from a dictionary or a pickle file containing model informations
        and predictions. All variables can be None, except for 'preds' which must be provided. You can add more
        attributes dynamically using the `add_attributes` method.

        Args:
            model_state (str): Model state to use, either "best" or "final".
            adjusted (bool): Whether to use adjusted predictions.
            model_info (Union[dict, str], optional): Model information provided as:
                - A dictionary with variable data.
                - A path to a pickle file.
            **kwargs: Additional variables to add dynamically.
        """

        self.seg_dict_w_class = None
        self.model_state = model_state
        self.adjusted = adjusted
        self.model_info = {}

        # Load data from pickle if provided
        if model_info:
            if isinstance(model_info, dict):
                self.model_info = model_info
            elif isinstance(model_info, str):
                with open(model_info, "rb") as file:
                    self.model_info = pickle.load(file)
            else:
                raise ValueError("Invalid model_info type. Must be a dictionary or a path to a pickle file.")

        # Update with kwargs
        self.model_info.update(kwargs)

        unexpected_variables = set(self.model_info.keys()) - self.EXPECTED_VARIABLES
        if unexpected_variables:
            raise ValueError(
                f"Unexpected keys: {unexpected_variables}. " f"Expected keys are: {self.EXPECTED_VARIABLES}"
            )

        # Dynamic attribute assignment
        for key in self.EXPECTED_VARIABLES:
            setattr(self, key, self.model_info.get(key, None))

        assert self.preds is not None, "The 'preds' attribute must be provided and cannot be None."
        assert self.spot_dict is not None, "The 'spot_dict' attribute must be provided and cannot be None."

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
        self.predicted_labels = self._get_labels_slide(self.predictions)
        self.predicted_proportions = self.get_predicted_proportions()

        all_spots = set(self.spot_dict.keys())
        all_cells = {cell for cell_list in self.spot_dict.values() for cell in cell_list}
        if self.train_spot_dict is not None:
            self.train_spots = list(self.train_spot_dict.keys())
            self.train_cells = list({cell for cell_list in self.train_spot_dict.values() for cell in cell_list})
            self.no_train_spots = list(all_spots - set(self.train_spots))
            self.no_train_cells = list(all_cells - set(self.train_cells))

        if self.ground_truth is not None:
            print("Loading true labels...")
            self.true_labels = self._get_labels_slide(self.ground_truth)

        if self.history is not None:
            self.history_train = self.history["train"]
            self.history_val = self.history["val"]

        else:
            print("Warning : No history provided. You won't be able to plot the training and validation histories.")
            print("Use `add_attributes(history=your_history)` to add one.")

        if self.seg_dict is not None:
            self._generate_dicts_viz_pred(self.seg_dict)

        else:
            print("Warning : No segmentation provided. You won't be able to plot the segmentation.")
            print("Use `add_attributes(seg_dict=your_seg_dict)` to add one.")

    def __repr__(self) -> str:
        """
        Return a string representation of the PredAnalyzer instance.

        Returns:
            str: A string containing all expected attributes and their values.
        """

        attrs = ", ".join(f"{k}={getattr(self, k, None)}" for k in self.EXPECTED_VARIABLES)
        return f"PredAnalyzer({attrs})"

    @classmethod
    def expected_variables(cls) -> Set[str]:
        """
        Get the set of expected variable keys.

        Returns:
            Set[str]: Expected variable keys.
        """

        return cls.EXPECTED_VARIABLES

    def add_attributes(self, **kwargs) -> None:
        """
        Dynamically add attributes to the instance if they are in EXPECTED_VARIABLES.

        Args:
            **kwargs: Attribute names and values to add.

        Raises:
            ValueError: If any key is not in EXPECTED_VARIABLES.
        """

        for key, value in kwargs.items():
            if key not in self.EXPECTED_VARIABLES:
                raise ValueError(
                    f"Cannot add attribute '{key}', " f"it is not in the expected keys: {self.EXPECTED_VARIABLES}"
                )

            elif key == "ground_truth":
                self.true_labels = self._get_labels_slide(value)

            elif key == "history":
                self.history_train = value["train"]
                self.history_val = value["val"]

            elif key == "seg_dict":
                self._generate_dicts_viz_pred(value)

            setattr(self, key, value)

    def list_attributes(self) -> Dict[str, Any]:
        """
        Return all current attributes of the instance.

        Returns:
            Dict[str, Any]: Dictionary of current attributes.
        """

        return {key: getattr(self, key, None) for key in self.EXPECTED_VARIABLES}

    def extract_stats(self, metric: str = "predicted") -> pd.DataFrame:
        """
        Extract statistics from predictions.

        Args:
            metric (str): Metric to use, either "predicted" or "all".

        Returns:
            pd.DataFrame: DataFrame containing class-wise statistics.

        Raises:
            ValueError: If an invalid metric is specified.
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

    @require_attributes("history_train", "history_val")
    def plot_history(self, show: bool = False, savefig: Optional[str] = None) -> None:
        """
        Plot training and validation history.

        Args:
            show (bool): Whether to display the plot.
            savefig (str, optional): File path to save the plot.

        Returns:
            Optional[plt.Figure]: Train and validation history of the model.
        """

        return plot_history(self.history_train, self.history_val, show=show, savefig=savefig)

    def find_cell_max_cell_type(self, cell_type: str) -> str:
        """
        Find the cell ID with the highest probability for a given cell type.

        Args:
            cell_type (str): Name of a cell type.

        Returns:
            str: Cell ID with the highest probability for the corresponding cell type.

        Raises:
            ValueError: If the cell type is not found in the predictions.
        """

        if cell_type not in self.ct_list:
            raise ValueError(f"Cell type '{cell_type}' not found in the prediction DataFrame.")
        max_cell_id = self.predictions[cell_type].idxmax()
        return max_cell_id

    @require_attributes("spot_dict", "image_dict")
    def plot_mosaic_cells(
        self, spot_id: Optional[str] = None, num_cols: int = 8, display: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot a grid of individual cell images for a given spot ID.

        Args:
            spot_id (str, optional): Spot ID to plot. If None, spot ID will be random.
            num_cols (int): Number of columns in the grid.
            display (bool): Whether to display the plot.

        Returns:
            Optional[plt.Figure]: Image grid with individual cell images of the same spot.
        """

        return plot_mosaic_cells(
            self.spot_dict,
            self.image_dict,
            spot_id=spot_id,
            predicted_labels=self.predicted_labels,
            true_labels=self.true_labels,
            num_cols=num_cols,
            display=display,
        )

    @require_attributes("spot_dict", "image_dict", "image_path", "adata", "adata_name")
    def plot_predicted_cell_labels_in_spot(
        self, spot_id: Optional[str] = None, show_labels: bool = True, display: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot a spot's visualization with all cell images arranged in a grid.

        Args:
            spot_id (str, optional): Spot ID to plot. If None, spot ID will be random.
            show_labels (bool): Whether to show predicted labels.
            display (bool): Whether to display the plot.

        Returns:
            Optional[plt.Figure]: Image spot with cell spots and potentially cell labels.
        """

        return plot_predicted_cell_labels_in_spot(
            spot_dict=self.spot_dict,
            adata=self.adata,
            adata_name=self.adata_name,
            image_path=self.image_path,
            image_dict=self.image_dict,
            predicted_labels=[None, self.predicted_labels][show_labels],
            true_labels=[None, self.true_labels][show_labels],
            spot_id=spot_id,
            display=display,
        )

    @require_attributes("spot_dict", "proportions", "image_path", "adata", "adata_name")
    def plot_spot_proportions(self, spot_id: Optional[str] = None, draw_seg: bool = False) -> None:
        """
        Plot true and predicted cell type proportions for a given spot.

        Args:
            spot_id (str, optional): Spot ID to plot. If None, selects a random spot.
            draw_seg (bool): Whether to draw segmentation overlays.
        """

        if draw_seg:
            if self.seg_dict_w_class is None:
                raise ValueError("You must run `generate_dicts_viz_pred` before to be able to plot segmentation.")

        if spot_id is None:
            spot_id = random.choice(list(self.spot_dict.keys()))
            print(f"Randomly selected spot_id: {spot_id}")

        elif spot_id not in self.spot_dict:
            raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

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

        pie_color_dict = generate_color_dict(self.ct_list, format="classic")

        # mean predicted probabilities
        ax1 = fig.add_subplot(gs[0, 1])
        mean_prob_ct = self.predictions[self.predictions.index.isin(list_cells)].mean(axis=0)
        plot_pie_chart(ax1, mean_prob_ct, color_dict=pie_color_dict)
        ax1.set_title("Mean Predicted Probabilities")

        # predicted cell type proportions
        ax2 = fig.add_subplot(gs[0, 2])
        prop_ct = self.predictions[self.predictions.index.isin(list_cells)].idxmax(axis=1).value_counts() / len(
            list_cells
        )
        plot_pie_chart(ax2, prop_ct, color_dict=pie_color_dict)
        ax2.set_title("Predicted Cell Type Proportions")

        # true cell type proportions
        ax3 = fig.add_subplot(gs[1, 1])
        true_prop = self.proportions.loc[spot_id]
        plot_pie_chart(ax3, true_prop, color_dict=pie_color_dict)
        ax3.set_title("True Cell Type Proportions")

        # legend
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        plot_legend(pie_color_dict, ax4)
        ax4.set_title("Legend")

        plt.tight_layout()
        plt.show()

    @require_attributes("proportions", "spot_dict")
    def evaluate_prop_predictions(self, subset="all") -> Dict[str, float]:
        """
        Evaluate slide-level predictions using various metrics. With this function, you can compute a series of
        metrics to compare, for each cell-type, true vs predicted proportions over the histological slide.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        if subset == "train":
            predicted_proportions = self.predicted_proportions.loc[self.train_spots]
        elif subset == "no_train":
            predicted_proportions = self.predicted_proportions.loc[self.no_train_spots]
        elif subset == "all":
            predicted_proportions = self.predicted_proportions.copy()
        else:
            raise ValueError("Invalid subset. Choose 'train', 'no_train', or 'all'.")

        true_proportions, predicted_proportions = self.proportions.align(predicted_proportions, join="inner", axis=0)

        if predicted_proportions.isna().any().any():
            predicted_proportions = predicted_proportions.dropna()
            true_proportions = true_proportions.loc[predicted_proportions.index]

        metrics = {}

        # Compute per-cell-type metrics
        pearson_list = []
        spearman_list = []
        mse_list = []
        mae_list = []

        for cell_type in true_proportions.columns:
            true_col = true_proportions[cell_type].values
            pred_col = predicted_proportions[cell_type].values

            if np.std(true_col) > 0 and np.std(pred_col) > 0:  # Ensure non-constant values
                pearson_corr = pearsonr(true_col, pred_col)[0]
                spearman_corr = spearmanr(true_col, pred_col)[0]
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan

            mse_value = mean_squared_error(true_col, pred_col)
            mae_value = mean_absolute_error(true_col, pred_col)

            metrics[f"Pearson Correlation {cell_type}"] = pearson_corr
            metrics[f"Spearman Correlation {cell_type}"] = spearman_corr
            metrics[f"MSE {cell_type}"] = mse_value
            metrics[f"MAE {cell_type}"] = mae_value

            pearson_list.append(pearson_corr)
            spearman_list.append(spearman_corr)
            mse_list.append(mse_value)
            mae_list.append(mae_value)

        # Compute global metrics (Averaged approach)
        metrics["Pearson Correlation global"] = np.nanmean(pearson_list)
        metrics["Spearman Correlation global"] = np.nanmean(spearman_list)
        metrics["MSE global"] = np.mean(mse_list)
        metrics["MAE global"] = np.mean(mae_list)

        return metrics

    def evaluate_cell_predictions(self, subset="all", per_class=True) -> Dict[str, float]:
        """
        Evaluate cell-level predictions using various metrics.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        if self.true_labels is None:
            raise ValueError("True labels are not available. Please provide ground_truth.")

        if subset == "train":
            true_labels = {k: v for k, v in self.true_labels.items() if k in self.train_cells}
            predicted_labels = {k: v for k, v in self.predicted_labels.items() if k in self.train_cells}
        elif subset == "no_train":
            true_labels = {k: v for k, v in self.true_labels.items() if k in self.no_train_cells}
            predicted_labels = {k: v for k, v in self.predicted_labels.items() if k in self.no_train_cells}
        elif subset == "all":
            true_labels = self.true_labels.copy()
            predicted_labels = self.predicted_labels.copy()
        else:
            raise ValueError("Invalid subset. Choose 'train', 'no_train', or 'all'.")

        true_labels = pd.Series({k: v["cell_type"] for k, v in true_labels.items()})
        predicted_labels = pd.Series({k: v["cell_type"] for k, v in predicted_labels.items()})

        # Global accuracy
        global_accuracy = accuracy_score(true_labels, predicted_labels)

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        # Weighted metrics
        weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        weighted_precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        weighted_recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

        metrics = {
            "Global Accuracy": global_accuracy,
            "Balanced Accuracy": balanced_acc,
            "Weighted F1 Score": weighted_f1,
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
        }

        if per_class:
            unique_classes = np.unique(true_labels)
            f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
            precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
            recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
            cm = confusion_matrix(true_labels, predicted_labels)

            metrics.update(
                {
                    "F1 Score (Per Class)": dict(zip(unique_classes, f1_per_class)),
                    "Precision (Per Class)": dict(zip(unique_classes, precision_per_class)),
                    "Recall (Per Class)": dict(zip(unique_classes, recall_per_class)),
                    "Confusion Matrix": pd.DataFrame(cm, columns=list(unique_classes), index=list(unique_classes)),
                }
            )

        return metrics

    @require_attributes("image_dict")
    def plot_grid_celltype(
        self, cell_type: str, n: int = 20, selection: str = "random", show_probs: bool = True, display: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot a grid of cell images predicted as a specific cell type.

        Args:
            cell_type (str): Target cell type to display.
            n (int): Number of images to include in the grid.
            selection (str): Selection mode ("max" or "random").
            show_probs (bool): Whether to show probability labels.
            display (bool): Whether to display the plot.

        Returns:
            Optional[plt.Figure]: The generated matplotlib figure.
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
    def get_predicted_proportions(self) -> pd.DataFrame:
        """
        Compute predicted proportions of cell types for each spot.

        Returns:
            pd.DataFrame: DataFrame with predicted proportions per spot.
        """

        spot_proportions = {}

        for spot_id, cell_ids in self.spot_dict.items():
            spot_cells = self.predictions.loc[self.predictions.index.isin(cell_ids)]
            spot_proportions[spot_id] = spot_cells.mean(axis=0)

        predicted_proportions_df = pd.DataFrame.from_dict(spot_proportions, orient="index")
        predicted_proportions_df.index.name = "spot"

        return predicted_proportions_df

    def _get_labels_slide(self, data) -> Dict[str, Dict[str, Any]]:
        """
        Generate a dictionary mapping cell IDs to predicted labels.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of cell IDs to predicted labels.
        """

        predicted_classes = data.idxmax(axis=1)

        predicted_class_indices = predicted_classes.map(data.columns.get_loc)

        predicted_labels = {
            cell_id: {"class": cls_idx, "cell_type": cls}
            for cell_id, cls_idx, cls in zip(data.index, predicted_class_indices, predicted_classes)
        }

        return predicted_labels

    def _generate_dicts_viz_pred(self, seg_dict: Dict[str, Any]) -> None:
        """
        Update the segmentation dictionary with predicted classes.

        Args:
            seg_dict (Dict[str, Any]): Dictionary containing segmentation data.
        """

        self.seg_dict_w_class = {
            "nuc": {
                key: {
                    **value,
                    "type": self.predicted_labels[key]["class"],
                }
                for key, value in seg_dict["nuc"].items()
                if key in self.predicted_labels
            }
        }
