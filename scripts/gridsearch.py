from __future__ import annotations

import argparse
import itertools
import logging
import os
import pickle

import pandas as pd
from run import main
from tools.analysis import evaluate_spot_predictions
from tools.analysis import get_predicted_proportions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hyperparameters to test
alphas = [0, 0.5]  # [0, 0.2, 0.3, 0.5]
learning_rates = [1e-5]  # [1e-5, 1e-4]
weights_options = [False]  # [True, False]
seeds = [42, 43]  # [42, 43, 44, 45, 46]

output_dir = "../out/new"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_experiment(alpha, lr, weights, divergence, seed):

    config_out_dir = os.path.join(
        output_dir, f"alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_seed_{seed}"
    )
    os.makedirs(config_out_dir, exist_ok=True)

    main(
        adata_name="CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma",
        json_path="/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json",
        image_path="/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_64.pt",
        path_st_adata="/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/",
        proportions_file="/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv",
        batch_size=1,
        lr=lr,
        weights=weights,
        agg="proba",
        divergence=divergence,
        reduction="mean",
        alpha=alpha,
        epochs=80,
        train_size=0.7,
        val_size=0.15,
        out_dir=config_out_dir,
        level=0,
        size_edge=64,
        dict_types=None,
        save_images=None,
        rs=seed,
    )


def aggregate_metrics(metrics_list):
    """Calculate the mean for each metric across multiple runs."""
    mean_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    return mean_metrics


def main_test(divergence):
    results_best = []
    results_final = []
    combinations = list(itertools.product(alphas, learning_rates, weights_options))

    for alpha, lr, weights, divergence in combinations:
        metrics_best_list = []
        metrics_final_list = []

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
            spot_dict, true_proportions, pred_best, pred_final = (
                info["spot_dict"],
                info["proportions"],
                info["pred_best"],
                info["pred_final"],
            )

            # Load true and predicted proportions
            predicted_best_proportions = get_predicted_proportions(pred_best, spot_dict)
            predicted_final_proportions = get_predicted_proportions(pred_final, spot_dict)

            # Evaluate the metrics
            metrics_best = evaluate_spot_predictions(true_proportions, predicted_best_proportions)
            metrics_final = evaluate_spot_predictions(true_proportions, predicted_final_proportions)
            metrics_best_list.append(metrics_best)
            metrics_final_list.append(metrics_final)

        # Calculate the mean metrics across seeds
        mean_metrics_best = aggregate_metrics(metrics_best_list)
        mean_metrics_final = aggregate_metrics(metrics_final_list)
        result_best = {"alpha": alpha, "lr": lr, "weights": weights, "divergence": divergence, **mean_metrics_best}
        result_final = {"alpha": alpha, "lr": lr, "weights": weights, "divergence": divergence, **mean_metrics_final}
        results_best.append(result_best)
        results_final.append(result_final)

        # Log results
        logger.info(f"Completed configuration: alpha={alpha}, lr={lr}, weights={weights}, divergence={divergence}")

    # Save final results
    results_df_best = pd.DataFrame(results_best)
    results_df_final = pd.DataFrame(results_final)
    results_df_best.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_best.csv"), index=False)
    results_df_final.to_csv(os.path.join(output_dir, f"summary_metrics_{divergence}_final.csv"), index=False)
    logger.info("Testing completed. Summary metrics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified divergence")
    parser.add_argument(
        "divergence", type=str, choices=["l1", "l2", "kl", "rot"], help="Divergence type for the experiment"
    )
    args = parser.parse_args()
    main_test(args.divergence)
