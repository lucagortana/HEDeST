from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from deconvplugin.analysis.pred_analyzer import PredAnalyzer


def compute_statistics(metrics_list: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute mean and confidence intervals for a list of metrics.

    Args:
        metrics_list (List[Dict[str, float]]): List of dictionaries containing metrics from each run.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Tuple containing the mean and confidence intervals for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std()
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    ci_values = {f"{key} ci": 1.96 * se for key, se in se_values.items()}

    return mean_values, ci_values


def evaluate_performance(
    table: pd.DataFrame,
    feature_a: Union[str, List[str]],
    feature_b: Union[str, List[str]],
    metric: str,
    na_fill: Optional[Any] = None,
    **fixed_features: Any,
) -> pd.DataFrame:
    """
    Evaluate performance by calculating mean metrics grouped by two features.

    Args:
        table (pd.DataFrame): DataFrame containing performance metrics.
        feature_a (Union[str, List[str]]): First feature(s) to group by.
        feature_b (Union[str, List[str]]): Second feature(s) to group by.
        metric (str): Metric to evaluate.
        na_fill (Optional[Any]): Value to fill NaN values in the metric column.
        **fixed_features: Fixed feature-value pairs to filter the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame of mean metrics grouped by feature_a and feature_b.
    """

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


def extract_stats(
    sim_folder: str,
    ground_truth_file: str,
) -> None:
    """
    Extracts the statistics from the simulation models.

    Args:
        sim_folder (str): Path to the simulation folder.
        ground_truth_file (str): Path to the ground truth file.
    """

    logger.info(f"Extracting statistics from simulation folder: {sim_folder}")
    logger.info(f"Ground truth file path: {ground_truth_file}")

    ground_truth = pd.read_csv(ground_truth_file, index_col=0)
    ground_truth.index = ground_truth.index.astype(str)

    keys_to_keep_cell = [
        "Global Accuracy",
        "Balanced Accuracy",
        "Weighted F1 Score",
        "Weighted Precision",
        "Weighted Recall",
    ]

    # Group folders by config (excluding seed)
    config_to_seeds = defaultdict(list)
    pattern = re.compile(
        r"model_(?P<model>[^_]+)_alpha_(?P<alpha>[^_]+)_lr_(?P<lr>[^_]+)_weights_(?P<weights>[^_]+)_"
        r"divergence_(?P<divergence>[^_]+)_seed_(?P<seed>\d+)"
    )

    for entry in os.listdir(sim_folder):
        match = pattern.match(entry)
        if match:
            config = (
                match.group("model"),
                match.group("alpha"),
                match.group("lr"),
                match.group("weights"),
                match.group("divergence"),
            )
            config_to_seeds[config].append((entry, match.group("seed")))

    # Prepare result containers
    all_results = {
        key: []
        for key in [
            "cells_best",
            "cells_best_train",
            "cells_best_no_train",
            "cells_best_adj",
            "cells_best_adj_train",
            "cells_best_adj_no_train",
            "slide_best",
            "slide_best_train",
            "slide_best_no_train",
            "slide_best_adj",
            "slide_best_adj_train",
            "slide_best_adj_no_train",
        ]
    }

    for config, runs in config_to_seeds.items():
        model_name, alpha, lr, weights, divergence = config
        metrics_lists = {key: [] for key in all_results}

        for folder_name, seed in runs:
            info_path = os.path.join(sim_folder, folder_name, "info.pickle")

            if not os.path.exists(info_path):
                logger.warning(f"Missing: {info_path}")
                continue

            with open(info_path, "rb") as f:
                model_info = pickle.load(f)

            analyzer_best = PredAnalyzer(
                model_info=model_info, model_state="best", adjusted=False, ground_truth=ground_truth
            )
            analyzer_best_adj = PredAnalyzer(
                model_info=model_info, model_state="best", adjusted=True, ground_truth=ground_truth
            )

            # Cell-level
            metrics_lists["cells_best"].append(
                {k: analyzer_best.evaluate_cell_predictions()[k] for k in keys_to_keep_cell}
            )
            metrics_lists["cells_best_train"].append(
                {k: analyzer_best.evaluate_cell_predictions(subset="train")[k] for k in keys_to_keep_cell}
            )
            metrics_lists["cells_best_no_train"].append(
                {k: analyzer_best.evaluate_cell_predictions(subset="no_train")[k] for k in keys_to_keep_cell}
            )

            metrics_lists["cells_best_adj"].append(
                {k: analyzer_best_adj.evaluate_cell_predictions()[k] for k in keys_to_keep_cell}
            )
            metrics_lists["cells_best_adj_train"].append(
                {k: analyzer_best_adj.evaluate_cell_predictions(subset="train")[k] for k in keys_to_keep_cell}
            )
            metrics_lists["cells_best_adj_no_train"].append(
                {k: analyzer_best_adj.evaluate_cell_predictions(subset="no_train")[k] for k in keys_to_keep_cell}
            )

            # Slide-level
            metrics_lists["slide_best"].append(analyzer_best.evaluate_spot_predictions_global())
            metrics_lists["slide_best_train"].append(analyzer_best.evaluate_spot_predictions_global(subset="train"))
            metrics_lists["slide_best_no_train"].append(
                analyzer_best.evaluate_spot_predictions_global(subset="no_train")
            )

            metrics_lists["slide_best_adj"].append(analyzer_best_adj.evaluate_spot_predictions_global())
            metrics_lists["slide_best_adj_train"].append(
                analyzer_best_adj.evaluate_spot_predictions_global(subset="train")
            )
            metrics_lists["slide_best_adj_no_train"].append(
                analyzer_best_adj.evaluate_spot_predictions_global(subset="no_train")
            )

        for key, metric_list in metrics_lists.items():
            if not metric_list:
                continue
            mean_vals, ci_vals = compute_statistics(metric_list)
            row = {
                "model": model_name,
                "alpha": str(alpha),
                "lr": str(lr),
                "weights": str(weights),
                "divergence": str(divergence),
                **mean_vals,
                **ci_vals,
            }
            all_results[key].append(row)

        logger.info(f"Done: model={model_name}, alpha={alpha}, lr={lr}, weights={weights}, divergence={divergence}")

    # Save final results
    for key, rows in all_results.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(sim_folder, f"summary_metrics_{key}.csv"), index=False)

    logger.info("All stats saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stats extraction across seeds")
    parser.add_argument("sim_folder", type=str, help="Path to simulation output folder")
    parser.add_argument("ground_truth_file", type=str, help="Path to ground truth file")
    args = parser.parse_args()

    extract_stats(args.sim_folder, args.ground_truth_file)
