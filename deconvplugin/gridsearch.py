from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from typing import List

from loguru import logger


def run_experiment(
    image_dict_path: str,
    spot_prop_df: str,
    spot_dict_file: str,
    spot_dict_adjust_file: str,
    model_name: str,
    batch_size: int,
    alpha: float,
    beta: float,
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
        spot_dict_adjust_file (str): Path to the adjustment spot dictionary file.
        model_name (str): Name of the model to use.
        batch_size (int): Batch size for training.
        alpha (float): Regularization parameter for the model.
        beta (float): Regularization parameter for bayesian adjustment.
        lr (float): Learning rate for training.
        weights (bool): Whether to use weighted loss during training.
        divergence (str): Divergence metric to use.
        seed (int): Random seed for reproducibility.
    """

    config_out_dir = os.path.join(
        out_dir,
        f"model_{model_name}_alpha_{alpha}_lr_{lr}_weights_{weights}_divergence_{divergence}_beta_{beta}_seed_{seed}",
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
        "--spot-dict-adjust-file",
        spot_dict_adjust_file,
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
        "--beta",
        str(beta),
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
    spot_dict_adjust_file: str,
    models: List[str],
    alphas: List[float],
    betas: List[float],
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
        spot_dict_adjust_file (str): Path to the global spot dictionary file.
        models (List[str]): List of model names.
        alphas (List[float]): List of alpha values.
        betas (List[float]): List of beta values.
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
    logger.info(f"Adjustment spot dictionary file path: {spot_dict_adjust_file}")
    logger.info(f"Models: {models}")
    logger.info(f"Alpha values: {alphas}")
    logger.info(f"Beta values: {betas}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Weights options: {weights_options}")
    logger.info(f"Divergence metrics: {divergences}")
    logger.info(f"Random seeds: {seeds}")
    logger.info(f"Output directory: {out_dir}\n")

    combinations = list(itertools.product(models, alphas, learning_rates, weights_options, divergences, betas))

    for model_name, alpha, lr, weights, divergence, beta in combinations:
        for seed in seeds:
            run_experiment(
                image_dict_path,
                spot_prop_df,
                spot_dict_file,
                spot_dict_adjust_file,
                model_name,
                batch_size,
                alpha,
                beta,
                lr,
                weights,
                divergence,
                out_dir,
                seed,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters")

    # String arguments
    parser.add_argument("image_dict_path", type=str, help="Path to the image dictionary file")
    parser.add_argument("spot_prop_df", type=str, help="Path to the spot proportions DataFrame")
    parser.add_argument("spot_dict_file", type=str, help="Path to the spot dictionary file")
    parser.add_argument("spot_dict_adjust_file", type=str, help="Path to the adjustment spot dictionary file")

    # List arguments
    parser.add_argument("--models", nargs="+", type=str, required=True, help="List of model names")
    parser.add_argument("--alphas", nargs="+", type=float, required=True, help="List of alpha values")
    parser.add_argument("--betas", nargs="+", type=float, required=True, help="List of beta values")
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
        args.spot_dict_adjust_file,
        args.models,
        args.alphas,
        args.betas,
        args.learning_rates,
        weights_options,
        args.divergences,
        args.seeds,
        args.batch_size,
        args.out_dir,
    )
