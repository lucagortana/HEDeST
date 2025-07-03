from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import pickle
import re
from collections import defaultdict
from typing import List

import pandas as pd
from joblib import delayed
from joblib import Parallel
from loguru import logger
from pyinstrument import Profiler

from benchmark.utils import compute_statistics


def process_config(config, runs, sim_folder, ground_truth, cell_list=None):
    """
    Process a single configuration (all seeds) and return results for all metric keys.
    """
    from hedest.analysis.pred_analyzer import PredAnalyzer

    profiler = Profiler()
    profiler.start()

    model_name, alpha, lr, divergence, beta = config

    text1 = f"[START] Model: {model_name}, alpha: {alpha}, lr: {lr},"
    text2 = f"divergence: {divergence}, beta: {beta}"
    logger.info(f"{text1} {text2}")

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
            model_info_subset = None

        analyzer_best = PredAnalyzer(
            model_info=model_info, model_state="best", adjusted=False, ground_truth=ground_truth
        )
        analyzer_best_adj = PredAnalyzer(
            model_info=model_info, model_state="best", adjusted=True, ground_truth=ground_truth
        )

        if cell_list is not None:
            model_info_subset = copy.deepcopy(model_info)
            preds = model_info_subset.get("preds", {})
            preds["pred_best"] = preds["pred_best"].loc[cell_list]
            preds["pred_best_adjusted"] = preds["pred_best_adjusted"].loc[cell_list]

            model_info_subset["spot_dict"] = {
                k: [c for c in v if c in cell_list] for k, v in model_info_subset["spot_dict"].items()
            }
            model_info_subset["train_spot_dict"] = {
                k: [c for c in v if c in cell_list] for k, v in model_info_subset["train_spot_dict"].items()
            }

            ground_truth_subset = ground_truth.loc[cell_list]

            logger.info(
                f"Filtered ground truth and predictions to {len(ground_truth_subset)} cells based on cell list."
            )

            analyzer_best_subset = PredAnalyzer(
                model_info=model_info_subset, model_state="best", adjusted=False, ground_truth=ground_truth_subset
            )
            analyzer_best_adj_subset = PredAnalyzer(
                model_info=model_info_subset, model_state="best", adjusted=True, ground_truth=ground_truth_subset
            )

        # Cell-level
        for subset in ["", "_train", "_no_train"]:
            sub = subset[1:] if subset else "all"

            # Cell-level
            if cell_list is not None:
                metrics_lists[f"cells_best{subset}"].append(
                    analyzer_best_subset.evaluate_cell_predictions(subset=sub, per_class=False)
                )
                metrics_lists[f"cells_best_adj{subset}"].append(
                    analyzer_best_adj_subset.evaluate_cell_predictions(subset=sub, per_class=False)
                )
            else:
                metrics_lists[f"cells_best{subset}"].append(
                    analyzer_best.evaluate_cell_predictions(subset=sub, per_class=False)
                )
                metrics_lists[f"cells_best_adj{subset}"].append(
                    analyzer_best_adj.evaluate_cell_predictions(subset=sub, per_class=False)
                )

            # Slide-level
            metrics_lists[f"slide_best{subset}"].append(analyzer_best.evaluate_prop_predictions(subset=sub))
            metrics_lists[f"slide_best_adj{subset}"].append(analyzer_best_adj.evaluate_prop_predictions(subset=sub))

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
            "divergence": divergence,
            "beta": beta,
            **mean_vals,
            **ci_vals,
        }
        results.append((key, row))

    profiler.stop()
    profiler.print()

    return config, results, metrics_lists


def extract_stats(
    sim_folder: str,
    ground_truth_file: str,
    cell_list_file: List[str] = None,
) -> None:
    """
    Extracts the statistics from the simulation models.

    Args:
        sim_folder (str): Path to the simulation folder.
        ground_truth_file (str): Path to the ground truth file.
        cell_list_file (List[str], optional): Path to a CSV file containing a list of cell IDs
        to filter the ground truth and predictions.
    """

    logger.info(f"Extracting statistics from simulation folder: {sim_folder}")
    logger.info(f"Ground truth file path: {ground_truth_file}")

    ground_truth = pd.read_csv(ground_truth_file, index_col=0)
    ground_truth.index = ground_truth.index.astype(str)

    cell_list = None
    if cell_list_file:
        cell_list = pd.read_csv(cell_list_file, dtype=str)
        cell_list = pd.Index(cell_list.iloc[:, 0], name="nucleus_id")

    # Group folders by config (excluding seed)
    config_to_seeds = defaultdict(list)
    pattern = re.compile(
        r"model_(?P<model>[^_]+)_alpha_(?P<alpha>[^_]+)_lr_(?P<lr>[^_]+)_"
        r"divergence_(?P<divergence>[^_]+)_beta_(?P<beta>[^_]+)_seed_(?P<seed>\d+)"
    )

    for entry in os.listdir(sim_folder):
        match = pattern.match(entry)
        if match:
            config = (
                match.group("model"),
                match.group("alpha"),
                match.group("lr"),
                match.group("divergence"),
                match.group("beta"),
            )
            config_to_seeds[config].append((entry, match.group("seed")))

    num_workers = min(8, multiprocessing.cpu_count())
    args_list = [(config, runs, sim_folder, ground_truth, cell_list) for config, runs in config_to_seeds.items()]

    logger.info(f"Starting parallel processing with {num_workers} workers...")

    processed = Parallel(n_jobs=num_workers)(delayed(process_config)(*args) for args in args_list)

    summary_by_key = defaultdict(list)
    per_run_by_key = defaultdict(list)

    for config, summary_rows, metrics_lists in processed:
        model, alpha, lr, divergence, beta = config

        for key, row in summary_rows:
            summary_by_key[key].append(row)

        for key, metric_dicts in metrics_lists.items():
            for metric in metric_dicts:
                row = {
                    "model": model,
                    "alpha": alpha,
                    "lr": lr,
                    "divergence": divergence,
                    "beta": beta,
                    **metric,
                }
                per_run_by_key[key].append(row)

    # Write one Excel file per metric key
    for key in summary_by_key:
        summary_df = pd.DataFrame(summary_by_key[key]).sort_values(by=["model", "alpha", "lr", "divergence", "beta"])
        per_run_df = pd.DataFrame(per_run_by_key[key]).sort_values(by=["model", "alpha", "lr", "divergence", "beta"])

        output_path = os.path.join(sim_folder, f"metrics_{key}.xlsx")
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            per_run_df.to_excel(writer, sheet_name="per_run", index=False)

    logger.info("All stats saved to Excel files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stats extraction across seeds")
    parser.add_argument("sim_folder", type=str, help="Path to simulation output folder")
    parser.add_argument("ground_truth_file", type=str, help="Path to ground truth file")
    parser.add_argument("--cell_list_file", type=str, default=None, help="Optional path to cell list file")

    args = parser.parse_args()

    extract_stats(
        sim_folder=args.sim_folder,
        ground_truth_file=args.ground_truth_file,
        cell_list_file=args.cell_list_file,
    )
