from __future__ import annotations

import os
import pickle
import time
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from torch import optim

from deconvplugin.analysis.pred_analyzer import PredAnalyzer
from deconvplugin.basics import format_time
from deconvplugin.basics import set_seed
from deconvplugin.bayes_adjust import BayesianAdjustment
from deconvplugin.dataset import SpotDataset
from deconvplugin.dataset_utils import split_data
from deconvplugin.model.cell_classifier import CellClassifier
from deconvplugin.predict import predict_slide
from deconvplugin.trainer import ModelTrainer

# def run_pri_deconv()


def run_sec_deconv(
    image_dict: Dict[str, torch.Tensor],
    spot_dict: Dict[str, List[str]],
    spot_prop_df: pd.DataFrame,
    spot_dict_global: Optional[Dict[str, List[str]]] = None,
    model_name: str = "resnet18",
    hidden_dims: List[int] = [512, 256],
    batch_size: int = 1,
    lr: float = 0.001,
    weights: bool = False,
    agg: str = "proba",
    divergence: str = "l1",
    reduction: str = "mean",
    alpha: float = 0.5,
    epochs: int = 25,
    train_size: float = 0.5,
    val_size: float = 0.25,
    out_dir: str = "results",
    tb_dir: str = "runs",
    rs: int = 42,
) -> None:
    """
    Runs secondary deconvolution pipeline for cell classification.

    Args:
        image_dict: Dictionary mapping cell IDs to image tensors.
        spot_dict: Dictionary mapping cell IDs to their spot.
        spot_prop_df: DataFrame containing cell type proportions for each spot.
        spot_dict_global: Dictionary mapping cell IDs to the closest spot.
        model_name: Name of the model to use.
        batch_size: Batch size for data loaders.
        lr: Learning rate for the optimizer.
        weights: Whether to use class weights based on global proportions.
        agg: Aggregation type for predictions ("proba" or "onehot").
        divergence: Type of divergence loss to use ("l1", "l2", "kl", "rot").
        reduction: Reduction method for the loss ("mean" or "sum").
        alpha: Weighting factor for the loss function.
        epochs: Number of training epochs.
        train_size: Proportion of data used for training.
        val_size: Proportion of data used for validation.
        out_dir: Directory to save results.
        tb_dir: Directory for TensorBoard logs.
        rs: Random seed for reproducibility.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")

    train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions = split_data(
        spot_dict, spot_prop_df, train_size=train_size, val_size=val_size, rs=rs
    )

    # Create datasets
    set_seed(rs)
    logger.debug("Creating datasets...")
    train_dataset = SpotDataset(train_spot_dict, train_proportions, image_dict)
    val_dataset = SpotDataset(val_spot_dict, val_proportions, image_dict)
    test_dataset = SpotDataset(test_spot_dict, test_proportions, image_dict)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = spot_prop_df.shape[1]
    ct_list = list(spot_prop_df.columns)

    # weights construction
    if weights:
        global_proportions = spot_prop_df.mean(axis=0).values
        weights = 1.0 / global_proportions
        weights /= weights.sum()
        weights = torch.tensor(weights)
    else:
        weights = None

    # Model initialization
    model = CellClassifier(model_name=model_name, num_classes=num_classes, hidden_dims=hidden_dims, device=device)
    model = model.to(device)
    logger.info(f"-> {num_classes} classes detected.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Model training
    trainer = ModelTrainer(
        model=model,
        ct_list=ct_list,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        weights=weights,
        agg=agg,
        divergence=divergence,
        reduction=reduction,
        alpha=alpha,
        num_epochs=epochs,
        out_dir=out_dir,
        tb_dir=tb_dir,
        rs=rs,
    )

    logger.info("Starting training...")
    TRAIN_START = time.time()
    trainer.train()
    trainer.save_history()
    TRAIN_TIME = format_time(time.time() - TRAIN_START)
    logger.info("Training completed.")

    # Predict on the whole slide
    logger.info("Starting prediction on the whole slide...")
    model4pred_best = CellClassifier(
        model_name=model_name, num_classes=num_classes, hidden_dims=hidden_dims, device=device
    )
    model4pred_best.load_state_dict(torch.load(trainer.best_model_path))
    cell_prob_best = predict_slide(model4pred_best, image_dict, ct_list)

    is_final = True
    try:
        model4pred_final = CellClassifier(
            model_name=model_name, num_classes=num_classes, hidden_dims=hidden_dims, device=device
        )
        model4pred_final.load_state_dict(torch.load(trainer.final_model_path))
        cell_prob_final = predict_slide(model4pred_final, image_dict, ct_list)
    except Exception:
        is_final = False

    # Bayesian adjustment
    is_bayesian = spot_dict_global is not None
    if is_bayesian:
        logger.info("Starting Bayesian adjustment...")
        p_c = spot_prop_df.loc[list(train_spot_dict.keys())].mean(axis=0)
        cell_prob_best_adjusted = BayesianAdjustment(
            cell_prob_best, spot_dict_global, spot_prop_df, p_c, device=device
        ).forward()
        if is_final:
            cell_prob_final_adjusted = BayesianAdjustment(
                cell_prob_final, spot_dict_global, spot_prop_df, p_c, device=device
            ).forward()

    # Save model infos
    model_info = {
        "model_name": model_name,
        "hidden_dims": hidden_dims,
        "spot_dict": spot_dict,
        "proportions": spot_prop_df,
        "history": {"train": trainer.history_train, "val": trainer.history_val},
        "preds": {
            "pred_best": cell_prob_best,
            **({"pred_best_adjusted": cell_prob_best_adjusted} if is_bayesian else {}),
            **(
                {"pred_final": cell_prob_final, "pred_final_adjusted": cell_prob_final_adjusted}
                if is_final and is_bayesian
                else {}
            ),
            **({"pred_final": cell_prob_final} if is_final and not is_bayesian else {}),
        },
    }

    info_dir = os.path.join(out_dir, "info.pickle")
    logger.info(f"Saving objects to {info_dir}...")
    with open(info_dir, "wb") as f:
        pickle.dump(model_info, f)

    # Extract and save statistics
    logger.info("Extracting and saving statistics...")

    stats_best_predicted = PredAnalyzer(model_info=model_info).extract_stats(metric="predicted")
    stats_best_all = PredAnalyzer(model_info=model_info).extract_stats(metric="all")

    if is_bayesian:
        stats_best_adj_predicted = PredAnalyzer(model_info=model_info, adjusted=True).extract_stats(metric="predicted")
        stats_best_adj_all = PredAnalyzer(model_info=model_info, adjusted=True).extract_stats(metric="all")

    if is_final:
        stats_final_predicted = PredAnalyzer(model_info=model_info, model_state="final").extract_stats(
            metric="predicted"
        )
        stats_final_all = PredAnalyzer(model_info=model_info, model_state="final").extract_stats(metric="all")

        if is_bayesian:
            stats_final_adj_predicted = PredAnalyzer(
                model_info=model_info, model_state="final", adjusted=True
            ).extract_stats(metric="predicted")
            stats_final_adj_all = PredAnalyzer(model_info=model_info, model_state="final", adjusted=True).extract_stats(
                metric="all"
            )

    with pd.ExcelWriter(os.path.join(out_dir, "stats.xlsx")) as writer:
        stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
        stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
        if is_bayesian:
            stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
            stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)
        if is_final:
            stats_final_predicted.to_excel(writer, sheet_name="final_predicted", index=False)
            stats_final_all.to_excel(writer, sheet_name="final_all", index=False)
            if is_bayesian:
                stats_final_adj_predicted.to_excel(writer, sheet_name="final_adj_predicted", index=False)
                stats_final_adj_all.to_excel(writer, sheet_name="final_adj_all", index=False)

    logger.info("Secondary deconvolution process completed successfully.")
    logger.info(f"Training time: {TRAIN_TIME}")
