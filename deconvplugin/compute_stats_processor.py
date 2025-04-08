from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


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


def process_config(args):
    """
    Process a single configuration (all seeds) and return results for all metric keys.
    """
    from deconvplugin.analysis.pred_analyzer import PredAnalyzer

    config, runs, sim_folder, ground_truth = args
    model_name, alpha, lr, weights, divergence = config
    logger.info(f"[START] Model: {model_name}, alpha: {alpha}, lr: {lr}, weights: {weights}, divergence: {divergence}")

    keys_to_keep_cell = [
        "Global Accuracy",
        "Balanced Accuracy",
        "Weighted F1 Score",
        "Weighted Precision",
        "Weighted Recall",
    ]

    metrics_lists = {
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
        for subset in ["", "_train", "_no_train"]:
            key = f"cells_best{subset}"
            metrics_lists[key].append(
                {
                    k: analyzer_best.evaluate_cell_predictions(subset=subset[1:] if subset else "all")[k]
                    for k in keys_to_keep_cell
                }
            )
            key_adj = f"cells_best_adj{subset}"
            metrics_lists[key_adj].append(
                {
                    k: analyzer_best_adj.evaluate_cell_predictions(subset=subset[1:] if subset else "all")[k]
                    for k in keys_to_keep_cell
                }
            )

        # Slide-level
        for subset in ["", "_train", "_no_train"]:
            sub = subset[1:] if subset else "all"
            metrics_lists[f"slide_best{subset}"].append(analyzer_best.evaluate_spot_predictions_global(subset=sub))
            metrics_lists[f"slide_best_adj{subset}"].append(
                analyzer_best_adj.evaluate_spot_predictions_global(subset=sub)
            )

    # Compute mean + CI
    results = []
    for key, metric_list in metrics_lists.items():
        if not metric_list:
            continue
        mean_vals, ci_vals = compute_statistics(metric_list)
        row = {
            "model": model_name,
            "alpha": alpha,
            "lr": lr,
            "weights": weights,
            "divergence": divergence,
            **mean_vals,
            **ci_vals,
        }
        results.append((key, row))
    return config, results


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

    num_workers = min(8, multiprocessing.cpu_count())
    results = defaultdict(list)
    args_list = [(config, runs, sim_folder, ground_truth) for config, runs in config_to_seeds.items()]

    logger.info(f"Starting parallel processing with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_config, args) for args in args_list]

        for future in as_completed(futures):
            try:
                config, res = future.result()
                for key, row in res:
                    results[key].append(row)
            except Exception as e:
                logger.error(f"Failed to process a config: {e}")

    for key, rows in results.items():
        df = pd.DataFrame(rows).sort_values(by=["model", "alpha", "lr", "weights", "divergence"])
        df.to_csv(os.path.join(sim_folder, f"summary_metrics_{key}.csv"), index=False)

    logger.info("All stats saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stats extraction across seeds")
    parser.add_argument("sim_folder", type=str, help="Path to simulation output folder")
    parser.add_argument("ground_truth_file", type=str, help="Path to ground truth file")
    args = parser.parse_args()

    extract_stats(args.sim_folder, args.ground_truth_file)
