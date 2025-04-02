from __future__ import annotations

import argparse
import itertools
import os
import pickle
import subprocess
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


def run_experiment(
    image_dict_path: str,
    spot_prop_df: str,
    spot_dict_file: str,
    spot_dict_global_file: str,
    model_name: str,
    batch_size: int,
    alpha: float,
    lr: float,
    weights: bool,
    divergence: str,
    out_dir: str,
    seed: int,
) -> None:
    """
    Run one experiment (model) with the specified parameters.

    Args:
        image_dict_path (str): Path to the image dictionary file.
        spot_prop_df (str): Path to the spot proportions DataFrame.
        spot_dict_file (str): Path to the spot dictionary file.
        spot_dict_global_file (str): Path to the global spot dictionary file.
        model_name (str): Name of the model to use.
        batch_size (int): Batch size for training.
        alpha (float): Regularization parameter for the model.
        lr (float): Learning rate for training.
        weights (bool): Whether to use weighted loss during training.
        divergence (str): Divergence metric to use.
        seed (int): Random seed for reproducibility.
    """

    config_out_dir = os.path.join(
        out_dir, f"model_{model_name}_alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}"
    )
    os.makedirs(config_out_dir, exist_ok=True)

    args = [
        "python3",
        "-u",
        "deconvplugin/main.py",
        image_dict_path,
        spot_prop_df,
        "--spot-dict-file",
        spot_dict_file,
        "--spot-dict-global-file",
        spot_dict_global_file,
        "--model-name",
        model_name,
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--agg",
        "proba",
        "--divergence",
        divergence,
        "--alpha",
        str(alpha),
        "--epochs",
        "80",
        "--out-dir",
        config_out_dir,
        "--tb-dir",
        "models/TBruns",
        "--rs",
        str(seed),
    ]

    if weights:
        args.append("--weights")

    subprocess.run(args, check=True)


def main_simulation(
    image_dict_path: str,
    spot_prop_df: str,
    spot_dict_file: str,
    spot_dict_global_file: str,
    ground_truth_file: str,
    models: List[str],
    alphas: List[float],
    learning_rates: List[float],
    weights_options: List[bool],
    divergences: List[str],
    seeds: List[int],
    batch_size: int,
    out_dir: str,
) -> None:
    """
    Perform the main simulation pipeline for a given divergence metric.

    Args:
        image_dict_path (str): Path to the image dictionary file.
        spot_prop_df (str): Path to the spot proportions DataFrame.
        spot_dict_file (str): Path to the spot dictionary file.
        spot_dict_global_file (str): Path to the global spot dictionary file.
        ground_truth_file (str): Path to the ground truth file.
        models (List[str]): List of model names.
        alphas (List[float]): List of alpha values.
        learning_rates (List[float]): List of learning rates.
        weights_options (List[bool]): List of weight options.
        divergences (List[str]): List of divergence metrics.
        seeds (List[int]): List of random seed values.
        batch_size (int): Batch size for training.
        out_dir (str): Output directory path.
    """

    logger.info(f"Image dictionary path: {image_dict_path}")
    logger.info(f"Spot proportions DataFrame path: {spot_prop_df}")
    logger.info(f"Spot dictionary file path: {spot_dict_file}")
    logger.info(f"Global spot dictionary file path: {spot_dict_global_file}")
    logger.info(f"Ground truth file path: {ground_truth_file}")
    logger.info(f"Models: {models}")
    logger.info(f"Alpha values: {alphas}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Weights options: {weights_options}")
    logger.info(f"Divergence metrics: {divergences}")
    logger.info(f"Random seeds: {seeds}")
    logger.info(f"Output directory: {out_dir}\n")

    results_spots_best = []
    results_spots_best_train = []
    results_spots_best_no_train = []
    results_spots_best_adj = []
    results_cells_best = []
    results_cells_best_train = []
    results_cells_best_no_train = []
    results_cells_best_adj = []
    results_slide_best = []
    results_slide_best_train = []
    results_slide_best_no_train = []
    results_slide_best_adj = []
    results_spots_final = []
    results_spots_final_adj = []
    results_cells_final = []
    results_cells_final_adj = []
    results_slide_final = []
    results_slide_final_adj = []

    ground_truth = pd.read_csv(ground_truth_file, index_col=0)
    ground_truth.index = ground_truth.index.astype(str)

    keys_to_keep_cell = [
        "Global Accuracy",
        "Balanced Accuracy",
        "Weighted F1 Score",
        "Weighted Precision",
        "Weighted Recall",
    ]

    combinations = list(itertools.product(models, alphas, learning_rates, weights_options, divergences))

    for model_name, alpha, lr, weights, divergence in combinations:

        metrics_spots_best_list = []
        metrics_spots_best_train_list = []
        metrics_spots_best_no_train_list = []
        metrics_spots_best_adj_list = []
        metrics_cells_best_list = []
        metrics_cells_best_train_list = []
        metrics_cells_best_no_train_list = []
        metrics_cells_best_adj_list = []
        metrics_slide_best_list = []
        metrics_slide_best_train_list = []
        metrics_slide_best_no_train_list = []
        metrics_slide_best_adj_list = []
        metrics_spots_final_list = []
        metrics_spots_final_adj_list = []
        metrics_cells_final_list = []
        metrics_cells_final_adj_list = []
        metrics_slide_final_list = []
        metrics_slide_final_adj_list = []

        for seed in seeds:

            # Run the experiment
            run_experiment(
                image_dict_path,
                spot_prop_df,
                spot_dict_file,
                spot_dict_global_file,
                model_name,
                batch_size,
                alpha,
                lr,
                weights,
                divergence,
                out_dir,
                seed,
            )

            info_path = os.path.join(
                out_dir,
                f"model_{model_name}_alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}",
                "info.pickle",
            )

            with open(info_path, "rb") as f:
                model_info = pickle.load(f)

            analyzer_best = PredAnalyzer(
                model_info=model_info, model_state="best", adjusted=False, ground_truth=ground_truth
            )
            analyzer_best_adj = PredAnalyzer(
                model_info=model_info, model_state="best", adjusted=True, ground_truth=ground_truth
            )

            is_final = True
            try:
                analyzer_final = PredAnalyzer(
                    model_info=model_info, model_state="final", adjusted=False, ground_truth=ground_truth
                )
                analyzer_final_adj = PredAnalyzer(
                    model_info=model_info, model_state="final", adjusted=True, ground_truth=ground_truth
                )
            except Exception:
                is_final = False

            # Evaluate the predictions
            metrics_spots_best = analyzer_best.evaluate_spot_predictions()
            metrics_spots_best_train = analyzer_best.evaluate_spot_predictions(subset="train")
            metrics_spots_best_no_train = analyzer_best.evaluate_spot_predictions(subset="no_train")
            metrics_spots_best_adj = analyzer_best_adj.evaluate_spot_predictions()
            metrics_cells_best = dict(
                (key, analyzer_best.evaluate_cell_predictions()[key]) for key in keys_to_keep_cell
            )
            metrics_cells_best_train = dict(
                (key, analyzer_best.evaluate_cell_predictions(subset="train")[key]) for key in keys_to_keep_cell
            )
            metrics_cells_best_no_train = dict(
                (key, analyzer_best.evaluate_cell_predictions(subset="no_train")[key]) for key in keys_to_keep_cell
            )
            metrics_cells_best_adj = dict(
                (key, analyzer_best_adj.evaluate_cell_predictions()[key]) for key in keys_to_keep_cell
            )
            metrics_slide_best = analyzer_best.evaluate_spot_predictions_global()
            metrics_slide_best_train = analyzer_best.evaluate_spot_predictions_global(subset="train")
            metrics_slide_best_no_train = analyzer_best.evaluate_spot_predictions_global(subset="no_train")
            metrics_slide_best_adj = analyzer_best_adj.evaluate_spot_predictions_global()

            if is_final:
                metrics_spots_final = analyzer_final.evaluate_spot_predictions()
                metrics_spots_final_adj = analyzer_final_adj.evaluate_spot_predictions()
                metrics_cells_final = dict(
                    (key, analyzer_final.evaluate_cell_predictions()[key]) for key in keys_to_keep_cell
                )
                metrics_cells_final_adj = dict(
                    (key, analyzer_final_adj.evaluate_cell_predictions()[key]) for key in keys_to_keep_cell
                )
                metrics_slide_final = analyzer_final.evaluate_spot_predictions_global()
                metrics_slide_final_adj = analyzer_final_adj.evaluate_spot_predictions_global()

            metrics_spots_best_list.append(metrics_spots_best)
            metrics_spots_best_train_list.append(metrics_spots_best_train)
            metrics_spots_best_no_train_list.append(metrics_spots_best_no_train)
            metrics_spots_best_adj_list.append(metrics_spots_best_adj)
            metrics_cells_best_list.append(metrics_cells_best)
            metrics_cells_best_train_list.append(metrics_cells_best_train)
            metrics_cells_best_no_train_list.append(metrics_cells_best_no_train)
            metrics_cells_best_adj_list.append(metrics_cells_best_adj)
            metrics_slide_best_list.append(metrics_slide_best)
            metrics_slide_best_train_list.append(metrics_slide_best_train)
            metrics_slide_best_no_train_list.append(metrics_slide_best_no_train)
            metrics_slide_best_adj_list.append(metrics_slide_best_adj)

            if is_final:
                metrics_spots_final_list.append(metrics_spots_final)
                metrics_spots_final_adj_list.append(metrics_spots_final_adj)
                metrics_cells_final_list.append(metrics_cells_final)
                metrics_cells_final_adj_list.append(metrics_cells_final_adj)
                metrics_slide_final_list.append(metrics_slide_final)
                metrics_slide_final_adj_list.append(metrics_slide_final_adj)

        # Calculate the mean metrics across seeds
        mean_metrics_spots_best, ci_metrics_spots_best = compute_statistics(metrics_spots_best_list)
        mean_metrics_spots_best_train, ci_metrics_spots_best_train = compute_statistics(metrics_spots_best_train_list)
        mean_metrics_spots_best_no_train, ci_metrics_spots_best_no_train = compute_statistics(
            metrics_spots_best_no_train_list
        )
        mean_metrics_spots_best_adj, ci_metrics_spots_best_adj = compute_statistics(metrics_spots_best_adj_list)
        mean_metrics_cells_best, ci_metrics_cells_best = compute_statistics(metrics_cells_best_list)
        mean_metrics_cells_best_train, ci_metrics_cells_best_train = compute_statistics(metrics_cells_best_train_list)
        mean_metrics_cells_best_no_train, ci_metrics_cells_best_no_train = compute_statistics(
            metrics_cells_best_no_train_list
        )
        mean_metrics_cells_best_adj, ci_metrics_cells_best_adj = compute_statistics(metrics_cells_best_adj_list)
        mean_metrics_slide_best, ci_metrics_slide_best = compute_statistics(metrics_slide_best_list)
        mean_metrics_slide_best_train, ci_metrics_slide_best_train = compute_statistics(metrics_slide_best_train_list)
        mean_metrics_slide_best_no_train, ci_metrics_slide_best_no_train = compute_statistics(
            metrics_slide_best_no_train_list
        )
        mean_metrics_slide_best_adj, ci_metrics_slide_best_adj = compute_statistics(metrics_slide_best_adj_list)
        mean_metrics_spots_final, ci_metrics_spots_final = compute_statistics(metrics_spots_final_list)
        mean_metrics_spots_final_adj, ci_metrics_spots_final_adj = compute_statistics(metrics_spots_final_adj_list)
        mean_metrics_cells_final, ci_metrics_cells_final = compute_statistics(metrics_cells_final_list)
        mean_metrics_cells_final_adj, ci_metrics_cells_final_adj = compute_statistics(metrics_cells_final_adj_list)
        mean_metrics_slide_final, ci_metrics_slide_final = compute_statistics(metrics_slide_final_list)
        mean_metrics_slide_final_adj, ci_metrics_slide_final_adj = compute_statistics(metrics_slide_final_adj_list)

        result_spots_best = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best,
            **ci_metrics_spots_best,
        }
        result_spots_best_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best_train,
            **ci_metrics_spots_best_train,
        }
        result_spots_best_no_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best_no_train,
            **ci_metrics_spots_best_no_train,
        }
        result_spots_best_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best_adj,
            **ci_metrics_spots_best_adj,
        }
        result_cells_best = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best,
            **ci_metrics_cells_best,
        }
        result_cells_best_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best_train,
            **ci_metrics_cells_best_train,
        }
        result_cells_best_no_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best_no_train,
            **ci_metrics_cells_best_no_train,
        }
        result_cells_best_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best_adj,
            **ci_metrics_cells_best_adj,
        }
        result_slide_best = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_best,
            **ci_metrics_slide_best,
        }
        result_slide_best_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_best_train,
            **ci_metrics_slide_best_train,
        }
        result_slide_best_no_train = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_best_no_train,
            **ci_metrics_slide_best_no_train,
        }
        result_slide_best_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_best_adj,
            **ci_metrics_slide_best_adj,
        }
        result_spots_final = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_final,
            **ci_metrics_spots_final,
        }
        result_spots_final_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_final_adj,
            **ci_metrics_spots_final_adj,
        }
        result_cells_final = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_final,
            **ci_metrics_cells_final,
        }
        result_cells_final_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_final_adj,
            **ci_metrics_cells_final_adj,
        }
        result_slide_final = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_final,
            **ci_metrics_slide_final,
        }
        result_slide_final_adj = {
            "model": model_name,
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_slide_final_adj,
            **ci_metrics_slide_final_adj,
        }

        results_spots_best.append(result_spots_best)
        results_spots_best_train.append(result_spots_best_train)
        results_spots_best_no_train.append(result_spots_best_no_train)
        results_spots_best_adj.append(result_spots_best_adj)
        results_cells_best.append(result_cells_best)
        results_cells_best_train.append(result_cells_best_train)
        results_cells_best_no_train.append(result_cells_best_no_train)
        results_cells_best_adj.append(result_cells_best_adj)
        results_slide_best.append(result_slide_best)
        results_slide_best_train.append(result_slide_best_train)
        results_slide_best_no_train.append(result_slide_best_no_train)
        results_slide_best_adj.append(result_slide_best_adj)
        results_spots_final.append(result_spots_final)
        results_spots_final_adj.append(result_spots_final_adj)
        results_cells_final.append(result_cells_final)
        results_cells_final_adj.append(result_cells_final_adj)
        results_slide_final.append(result_slide_final)
        results_slide_final_adj.append(result_slide_final_adj)

        info1 = f"Completed configuration: model={model_name}, alpha={alpha}, "
        info2 = f"lr={lr}, weights={weights}, divergence={divergence}."
        logger.info(info1 + info2)

    # Save final results
    results_df_spots_best = pd.DataFrame(results_spots_best)
    results_df_spots_best_train = pd.DataFrame(results_spots_best_train)
    results_df_spots_best_no_train = pd.DataFrame(results_spots_best_no_train)
    results_df_spots_best_adj = pd.DataFrame(results_spots_best_adj)
    results_df_cells_best = pd.DataFrame(results_cells_best)
    results_df_cells_best_train = pd.DataFrame(results_cells_best_train)
    results_df_cells_best_no_train = pd.DataFrame(results_cells_best_no_train)
    results_df_cells_best_adj = pd.DataFrame(results_cells_best_adj)
    results_df_slide_best = pd.DataFrame(results_slide_best)
    results_df_slide_best_train = pd.DataFrame(results_slide_best_train)
    results_df_slide_best_no_train = pd.DataFrame(results_slide_best_no_train)
    results_df_slide_best_adj = pd.DataFrame(results_slide_best_adj)
    results_df_spots_final = pd.DataFrame(results_spots_final)
    results_df_spots_final_adj = pd.DataFrame(results_spots_final_adj)
    results_df_cells_final = pd.DataFrame(results_cells_final)
    results_df_cells_final_adj = pd.DataFrame(results_cells_final_adj)
    results_df_slide_final = pd.DataFrame(results_slide_final)
    results_df_slide_final_adj = pd.DataFrame(results_slide_final_adj)

    results_df_spots_best.to_csv(os.path.join(out_dir, "summary_metrics_spots_best.csv"), index=False)
    results_df_spots_best_train.to_csv(os.path.join(out_dir, "summary_metrics_spots_best_train.csv"), index=False)
    results_df_spots_best_no_train.to_csv(os.path.join(out_dir, "summary_metrics_spots_best_no_train.csv"), index=False)
    results_df_spots_best_adj.to_csv(os.path.join(out_dir, "summary_metrics_spots_best_adj.csv"), index=False)
    results_df_cells_best.to_csv(os.path.join(out_dir, "summary_metrics_cells_best.csv"), index=False)
    results_df_cells_best_train.to_csv(os.path.join(out_dir, "summary_metrics_cells_best_train.csv"), index=False)
    results_df_cells_best_no_train.to_csv(os.path.join(out_dir, "summary_metrics_cells_best_no_train.csv"), index=False)
    results_df_cells_best_adj.to_csv(os.path.join(out_dir, "summary_metrics_cells_best_adj.csv"), index=False)
    results_df_slide_best.to_csv(os.path.join(out_dir, "summary_metrics_slide_best.csv"), index=False)
    results_df_slide_best_train.to_csv(os.path.join(out_dir, "summary_metrics_slide_best_train.csv"), index=False)
    results_df_slide_best_no_train.to_csv(os.path.join(out_dir, "summary_metrics_slide_best_no_train.csv"), index=False)
    results_df_slide_best_adj.to_csv(os.path.join(out_dir, "summary_metrics_slide_best_adj.csv"), index=False)
    results_df_spots_final.to_csv(os.path.join(out_dir, "summary_metrics_spots_final.csv"), index=False)
    results_df_spots_final_adj.to_csv(os.path.join(out_dir, "summary_metrics_spots_final_adj.csv"), index=False)
    results_df_cells_final.to_csv(os.path.join(out_dir, "summary_metrics_cells_final.csv"), index=False)
    results_df_cells_final_adj.to_csv(os.path.join(out_dir, "summary_metrics_cells_final_adj.csv"), index=False)
    results_df_slide_final.to_csv(os.path.join(out_dir, "summary_metrics_slide_final.csv"), index=False)
    results_df_slide_final_adj.to_csv(os.path.join(out_dir, "summary_metrics_slide_final_adj.csv"), index=False)

    logger.info("Testing completed. Summary metrics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters")

    # String arguments
    parser.add_argument("image_dict_path", type=str, help="Path to the image dictionary file")
    parser.add_argument("spot_prop_df", type=str, help="Path to the spot proportions DataFrame")
    parser.add_argument("spot_dict_file", type=str, help="Path to the spot dictionary file")
    parser.add_argument("spot_dict_global_file", type=str, help="Path to the global spot dictionary file")
    parser.add_argument("ground_truth_file", type=str, help="Path to the ground truth file")

    # List arguments
    parser.add_argument("--models", nargs="+", type=str, required=True, help="List of model names")
    parser.add_argument("--alphas", nargs="+", type=float, required=True, help="List of alpha values")
    parser.add_argument("--learning_rates", nargs="+", type=float, required=True, help="List of learning rates")
    parser.add_argument(
        "--weights_options",
        nargs="+",
        type=int,
        required=True,
        choices=[0, 1],
        help="List of weight options (0 for False, 1 for True)",
    )
    parser.add_argument("--divergences", nargs="+", type=str, required=True, help="List of divergence metrics")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of random seed values")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    # Output directory
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory path")

    args = parser.parse_args()
    weights_options = [bool(w) for w in args.weights_options]

    main_simulation(
        args.image_dict_path,
        args.spot_prop_df,
        args.spot_dict_file,
        args.spot_dict_global_file,
        args.ground_truth_file,
        args.models,
        args.alphas,
        args.learning_rates,
        weights_options,
        args.divergences,
        args.seeds,
        args.batch_size,
        args.out_dir,
    )
