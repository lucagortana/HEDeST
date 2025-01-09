from __future__ import annotations

import argparse
import itertools
import os
import pickle
import subprocess

import pandas as pd
from loguru import logger

from deconvplugin.analysis import PredAnalyzer

# Hyperparameters to test
alphas = [0, 0.0001, 0.001]
learning_rates = [1e-4, 1e-3]
weights_options = [True, False]
seeds = [42, 43]

output_dir = "../out/new"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_experiment(alpha, lr, weights, divergence, seed):

    config_out_dir = os.path.join(
        output_dir, f"alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}"
    )
    os.makedirs(config_out_dir, exist_ok=True)

    proportions_file1 = "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"
    proportions_file2 = "/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv"

    args = [
        "python3",
        "run.py",
        "--adata_name",
        "CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma",
        "--json_path",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json",
        "--image_path",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/image_dict_64.pt",
        "--path_st_adata",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/",
        "--proportions_file",
        proportions_file1 + proportions_file2,
        "--batch_size",
        "1",
        "--lr",
        f"{lr}",
        "--agg",
        "proba",
        "--divergence",
        divergence,
        "--reduction",
        "mean",
        "--alpha",
        f"{alpha}",
        "--epochs",
        "80",
        "--train_size",
        "0.7",
        "--val_size",
        "0.15",
        "--out_dir",
        config_out_dir,
        "--level",
        "0",
        "--size_edge",
        "64",
        "--rs",
        f"{seed}",
    ]

    if weights:
        args.append("--weights")

    subprocess.run(args, check=True)


def aggregate_metrics(metrics_list):
    """Calculate the mean for each metric across multiple runs."""
    mean_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    return mean_metrics


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


def main_test(divergence):
    results_best = []
    results_best_adj = []
    results_final = []
    results_final_adj = []
    combinations = list(itertools.product(alphas, learning_rates, weights_options))

    for alpha, lr, weights in combinations:
        metrics_best_list = []
        metrics_best_adj_list = []
        metrics_final_list = []
        metrics_final_adj_list = []

        for seed in seeds:

            run_experiment(alpha, lr, weights, divergence, seed)

            # Assuming predictions are saved in "predicted_proportions.csv" and "true_proportions.csv"
            info_path = os.path.join(
                output_dir,
                f"alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}",
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

            # Evaluate the metrics
            metrics_best = analyzer_best.evaluate_spot_predictions()
            metrics_best_adj = analyzer_best_adj.evaluate_spot_predictions()

            if is_final:
                metrics_final = analyzer_final.evaluate_spot_predictions()
                metrics_final_adj = analyzer_final_adj.evaluate_spot_predictions()

            metrics_best_list.append(metrics_best)
            metrics_best_adj_list.append(metrics_best_adj)

            if is_final:
                metrics_final_list.append(metrics_final)
                metrics_final_adj_list.append(metrics_final_adj)

        # Calculate the mean metrics across seeds
        mean_metrics_best = aggregate_metrics(metrics_best_list)
        mean_metrics_best_adj = aggregate_metrics(metrics_best_adj_list)
        mean_metrics_final = aggregate_metrics(metrics_final_list)
        mean_metrics_final_adj = aggregate_metrics(metrics_final_adj_list)

        result_best = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_best,
        }
        result_best_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_best_adj,
        }
        result_final = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_final,
        }
        result_final_adj = {
            "alpha": str(alpha),
            "lr": str(lr),
            "weights": str(weights),
            "divergence": str(divergence),
            **mean_metrics_final_adj,
        }

        results_best.append(result_best)
        results_best_adj.append(result_best_adj)
        results_final.append(result_final)
        results_final_adj.append(result_final_adj)

        # Log results
        logger.info(f"Completed configuration: alpha={alpha}, lr={lr}, weights={weights}, divergence={divergence}")

    # Save final results
    results_df_best = pd.DataFrame(results_best)
    results_df_best_adj = pd.DataFrame(results_best_adj)
    results_df_final = pd.DataFrame(results_final)
    results_df_final_adj = pd.DataFrame(results_final_adj)

    results_df_best.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_best.csv"), index=False)
    results_df_best_adj.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_best_adj.csv"), index=False)
    results_df_final.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_final.csv"), index=False)
    results_df_final_adj.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_final_adj.csv"), index=False)
    logger.info("Testing completed. Summary metrics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified divergence")
    parser.add_argument(
        "divergence", type=str, choices=["l1", "l2", "kl", "rot"], help="Divergence type for the experiment"
    )
    args = parser.parse_args()
    main_test(args.divergence)
