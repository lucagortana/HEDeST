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
from typing import Union

import pandas as pd
from loguru import logger

from deconvplugin.analysis.pred_analyzer import PredAnalyzer


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the mean for each metric across multiple runs.

    Args:
        metrics_list (List[Dict[str, float]]): List of dictionaries containing metrics from each run.

    Returns:
        Dict[str, float]: Dictionary containing the mean of each metric.
    """

    mean_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    return mean_metrics


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
    model_name: str,
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
        model_name (str): Name of the model to use.
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
        "--model-name",
        model_name,
        "--lr",
        lr,
        "--agg",
        "proba",
        "--divergence",
        divergence,
        "--reduction",
        "mean",
        "--alpha",
        alpha,
        "--epochs",
        "80",
        "--out-dir",
        config_out_dir,
        "--tb-dir",
        "models/TBruns",
        "--rs",
        seed,
    ]

    if weights:
        args.append("--weights")

    subprocess.run(args, check=True)


def main_simulation(
    image_dict_path: str,
    spot_prop_df: str,
    spot_dict_file: str,
    models: List[str],
    alphas: List[float],
    learning_rates: List[float],
    weights_options: List[bool],
    divergences: List[str],
    seeds: List[int],
    out_dir: str,
) -> None:
    """
    Perform the main simulation pipeline for a given divergence metric.

    Args:
        divergence (str): Divergence metric to use in the simulation.
    """

    logger.info("Starting simulation...")
    logger.info("Image dictionary path: ", image_dict_path)
    logger.info("Spot proportions DataFrame path: ", spot_prop_df)
    logger.info("Spot dictionary file path: ", spot_dict_file)
    logger.info("Models: ", models)
    logger.info("Alpha values: ", alphas)
    logger.info("Learning rates: ", learning_rates)
    logger.info("Weights options: ", weights_options)
    logger.info("Divergence metrics: ", divergences)
    logger.info("Random seeds: ", seeds)
    logger.info("Output directory: ", out_dir)

    results_spots_best = []
    results_spots_best_adj = []
    results_cells_best = []
    results_cells_best_adj = []
    results_spots_final = []
    results_spots_final_adj = []
    results_cells_final = []
    results_cells_final_adj = []

    combinations = list(itertools.product(models, alphas, learning_rates, weights_options, divergences))

    for model_name, alpha, lr, weights, divergence in combinations:

        metrics_spots_best_list = []
        metrics_spots_best_adj_list = []
        metrics_cells_best_list = []
        metrics_cells_best_adj_list = []
        metrics_spots_final_list = []
        metrics_spots_final_adj_list = []
        metrics_cells_final_list = []
        metrics_cells_final_adj_list = []

        for seed in seeds:

            # Run the experiment
            run_experiment(
                image_dict_path, spot_prop_df, spot_dict_file, model_name, alpha, lr, weights, divergence, out_dir, seed
            )

            info_path = os.path.join(
                out_dir,
                f"model_{model_name}_alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}",
                "info.pickle",
            )

            with open(info_path, "rb") as f:
                model_info = pickle.load(f)

            analyzer_best = PredAnalyzer(model_info=model_info, model_state="best", adjusted=False)
            analyzer_best_adj = PredAnalyzer(model_info=model_info, model_state="best", adjusted=True)

            is_final = True
            try:
                analyzer_final = PredAnalyzer(model_info=model_info, model_state="final", adjusted=False)
                analyzer_final_adj = PredAnalyzer(model_info=model_info, model_state="final", adjusted=True)
            except Exception:
                is_final = False

            # Evaluate the predictions
            metrics_spots_best = analyzer_best.evaluate_spot_predictions()
            metrics_spots_best_adj = analyzer_best_adj.evaluate_spot_predictions()
            metrics_cells_best = analyzer_best.evaluate_cell_predictions()
            metrics_cells_best_adj = analyzer_best_adj.evaluate_cell_predictions()

            if is_final:
                metrics_spots_final = analyzer_final.evaluate_spot_predictions()
                metrics_spots_final_adj = analyzer_final_adj.evaluate_spot_predictions()
                metrics_cells_final = analyzer_final.evaluate_cell_predictions()
                metrics_cells_final_adj = analyzer_final_adj.evaluate_cell_predictions()

            metrics_spots_best_list.append(metrics_spots_best)
            metrics_spots_best_adj_list.append(metrics_spots_best_adj)
            metrics_cells_best_list.append(metrics_cells_best)
            metrics_cells_best_adj_list.append(metrics_cells_best_adj)

            if is_final:
                metrics_spots_final_list.append(metrics_spots_final)
                metrics_spots_final_adj_list.append(metrics_spots_final_adj)
                metrics_cells_final_list.append(metrics_cells_final)
                metrics_cells_final_adj_list.append(metrics_cells_final_adj)

        # Calculate the mean metrics across seeds
        mean_metrics_spots_best = aggregate_metrics(metrics_spots_best_list)
        mean_metrics_spots_best_adj = aggregate_metrics(metrics_spots_best_adj_list)
        mean_metrics_cells_best = aggregate_metrics(metrics_cells_best_list)
        mean_metrics_cells_best_adj = aggregate_metrics(metrics_cells_best_adj_list)
        mean_metrics_spots_final = aggregate_metrics(metrics_spots_final_list)
        mean_metrics_spots_final_adj = aggregate_metrics(metrics_spots_final_adj_list)
        mean_metrics_cells_final = aggregate_metrics(metrics_cells_final_list)
        mean_metrics_cells_final_adj = aggregate_metrics(metrics_cells_final_adj_list)

        result_spots_best = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best,
        }
        result_spots_best_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_best_adj,
        }
        result_cells_best = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best,
        }
        result_cells_best_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_best_adj,
        }
        result_spots_final = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_final,
        }
        result_spots_final_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_spots_final_adj,
        }
        result_cells_final = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_final,
        }
        result_cells_final_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_cells_final_adj,
        }

        results_spots_best.append(result_spots_best)
        results_spots_best_adj.append(result_spots_best_adj)
        results_cells_best.append(result_cells_best)
        results_cells_best_adj.append(result_cells_best_adj)
        results_spots_final.append(result_spots_final)
        results_spots_final_adj.append(result_spots_final_adj)
        results_cells_final.append(result_cells_final)
        results_cells_final_adj.append(result_cells_final_adj)

        logger.info(
            f"Completed configuration: model={model_name}, alpha={alpha}, ",
            f"lr={lr}, weights={weights}, divergence={divergence}.",
        )

    # Save final results
    results_df_spots_best = pd.DataFrame(results_spots_best)
    results_df_spots_best_adj = pd.DataFrame(results_spots_best_adj)
    results_df_cells_best = pd.DataFrame(results_cells_best)
    results_df_cells_best_adj = pd.DataFrame(results_cells_best_adj)
    results_df_spots_final = pd.DataFrame(results_spots_final)
    results_df_spots_final_adj = pd.DataFrame(results_spots_final_adj)
    results_df_cells_final = pd.DataFrame(results_cells_final)
    results_df_cells_final_adj = pd.DataFrame(results_cells_final_adj)

    results_df_spots_best.to_csv(os.path.join(out_dir, "summary_metrics_spots_best.csv"), index=False)
    results_df_spots_best_adj.to_csv(os.path.join(out_dir, "summary_metrics_spots_best_adj.csv"), index=False)
    results_df_cells_best.to_csv(os.path.join(out_dir, "summary_metrics_cells_best.csv"), index=False)
    results_df_cells_best_adj.to_csv(os.path.join(out_dir, "summary_metrics_cells_best_adj.csv"), index=False)
    results_df_spots_final.to_csv(os.path.join(out_dir, "summary_metrics_spots_final.csv"), index=False)
    results_df_spots_final_adj.to_csv(os.path.join(out_dir, "summary_metrics_spots_final_adj.csv"), index=False)
    results_df_cells_final.to_csv(os.path.join(out_dir, "summary_metrics_cells_final.csv"), index=False)
    results_df_cells_final_adj.to_csv(os.path.join(out_dir, "summary_metrics_cells_final_adj.csv"), index=False)

    logger.info("Testing completed. Summary metrics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters")

    # String arguments
    parser.add_argument("image_dict_path", type=str, help="Path to the image dictionary file")
    parser.add_argument("spot_prop_df", type=str, help="Path to the spot proportions DataFrame")
    parser.add_argument("spot_dict_file", type=str, help="Path to the spot dictionary file")

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

    # Output directory
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory path")

    args = parser.parse_args()
    weights_options = [bool(w) for w in args.weights_options]

    main_simulation(
        args.image_dict_path,
        args.spot_prop_df,
        args.spot_dict_file,
        args.models,
        args.alphas,
        args.learning_rates,
        weights_options,
        args.divergences,
        args.seeds,
        args.out_dir,
    )
