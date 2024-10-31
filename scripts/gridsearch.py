from __future__ import annotations

import argparse
import itertools
import logging
import os
import pickle
import subprocess

import pandas as pd
from run import main
from tools.analysis import evaluate_spot_predictions
from tools.analysis import get_predicted_proportions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hyperparameters to test
alphas = [0, 0.2, 0.3, 0.5]
learning_rates = [1e-5, 1e-4]
weights_options = [True, False]
seeds = [42, 43, 44, 45, 46]

output_dir = "../out/new"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_experiment(alpha, lr, weights, divergence, seed):

    config_out_dir = os.path.join(
        output_dir, f"alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}"
    )
    os.makedirs(config_out_dir, exist_ok=True)

    args = [
        "python3",
        "run.py",
        "--adata_name",
        "CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma",
        "--json_path",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json",
        "--image_path",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_64.pt",
        "--path_st_adata",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/",
        "--proportions_file",
        "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv",
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
                info = pickle.load(f)

            spot_dict = info["spot_dict"]
            true_proportions = info["proportions"]
            pred_best = info["pred_best"]
            pred_best_adj = info["pred_best_adjusted"]

            is_final = True
            try:
                pred_final = info["pred_final"]
                pred_final_adj = info["pred_final_adjusted"]
            except Exception:
                is_final = False

            # Load true and predicted proportions
            predicted_best_proportions = get_predicted_proportions(pred_best, spot_dict)
            predicted_best_proportions_adj = get_predicted_proportions(pred_best_adj, spot_dict)

            if is_final:
                predicted_final_proportions = get_predicted_proportions(pred_final, spot_dict)
                predicted_final_proportions_adj = get_predicted_proportions(pred_final_adj, spot_dict)

            # Evaluate the metrics
            metrics_best = evaluate_spot_predictions(true_proportions, predicted_best_proportions)
            metrics_best_adj = evaluate_spot_predictions(true_proportions, predicted_best_proportions_adj)

            if is_final:
                metrics_final = evaluate_spot_predictions(true_proportions, predicted_final_proportions)
                metrics_final_adj = evaluate_spot_predictions(true_proportions, predicted_final_proportions_adj)

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

        result_best = {"alpha": alpha, "lr": lr, "weights": weights, "divergence": divergence, **mean_metrics_best}
        result_best_adj = {
            "alpha": alpha,
            "lr": lr,
            "weights": weights,
            "divergence": divergence,
            **mean_metrics_best_adj,
        }
        result_final = {"alpha": alpha, "lr": lr, "weights": weights, "divergence": divergence, **mean_metrics_final}
        result_final_adj = {
            "alpha": alpha,
            "lr": lr,
            "weights": weights,
            "divergence": divergence,
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
