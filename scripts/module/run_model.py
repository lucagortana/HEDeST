from __future__ import annotations

import logging
import os
import pickle
import time

import pandas as pd
import torch
from module.bayes_adjust import BayesianAdjustment
from module.cell_classifier import CellClassifier
from module.load_data import split_data
from module.load_data import SpotDataset
from module.trainer import ModelTrainer
from tools.analysis import extract_stats
from tools.analysis import get_labels_slide
from tools.analysis import predict_slide
from tools.basics import format_time
from tools.basics import set_seed
from torch import optim

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# def run_pri_deconv()


def run_sec_deconv(
    image_dict,
    spot_dict,
    spot_dict_global,
    proportions,
    batch_size=1,
    lr=0.001,
    weights=False,
    agg="proba",
    divergence="l1",
    reduction="mean",
    alpha=0.5,
    epochs=25,
    train_size=0.5,
    val_size=0.25,
    out_dir="results",
    rs=42,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")

    train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions = split_data(
        spot_dict, proportions, train_size=train_size, val_size=val_size, rs=rs
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

    num_classes = proportions.shape[1]
    ct_list = list(proportions.columns)

    # weights construction
    if weights:
        global_proportions = proportions.mean(axis=0).values
        weights = 1.0 / global_proportions
        weights /= weights.sum()
        weights = torch.tensor(weights)
    else:
        weights = None

    size_edge = image_dict["0"].shape[1]
    model = CellClassifier(size_edge=size_edge, num_classes=num_classes, device=device)
    model = model.to(device)
    logger.info(f"-> {num_classes} classes detected.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = ModelTrainer(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        weights=weights,
        agg=agg,
        divergence=divergence,
        reduction=reduction,
        alpha=alpha,
        num_epochs=epochs,
        out_dir=out_dir,
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
    model4pred_best = CellClassifier(size_edge=size_edge, num_classes=num_classes, device=device)
    model4pred_best.load_state_dict(torch.load(trainer.best_model_path))
    pred_best = predict_slide(model4pred_best, image_dict, ct_list)

    is_final = True
    try:
        model4pred_final = CellClassifier(size_edge=size_edge, num_classes=num_classes, device=device)
        model4pred_final.load_state_dict(torch.load(trainer.final_model_path))
        pred_final = predict_slide(model4pred_final, image_dict, ct_list)
    except Exception:
        is_final = False

    # Bayesian adjustment
    logger.info("Starting Bayesian adjustment...")
    p_c = proportions.loc[list(train_spot_dict.keys())].mean(axis=0)
    pred_best_adjusted = BayesianAdjustment(pred_best, spot_dict_global, proportions, p_c, device=device).forward()
    if is_final:
        pred_final_adjusted = BayesianAdjustment(
            pred_final, spot_dict_global, proportions, p_c, device=device
        ).forward()

    # Save model infos
    info = {
        "spot_dict": spot_dict,
        "proportions": proportions,
        "pred_best": pred_best,
        "pred_best_adjusted": pred_best_adjusted,
    }
    if is_final:
        info["pred_final"] = pred_final
        info["pred_final_adjusted"] = pred_final_adjusted
    info_dir = f"{out_dir}/info.pickle"
    logger.info(f"Saving objects to {info_dir}...")
    with open(info_dir, "wb") as f:
        pickle.dump(info, f)

    logger.info("Extracting and saving statistics...")
    pred_best_labels = get_labels_slide(pred_best)
    stats_best_predicted = extract_stats(pred_best, predicted_labels=pred_best_labels, metric="predicted")
    stats_best_all = extract_stats(pred_best, predicted_labels=pred_best_labels, metric="all")

    pred_best_adj_labels = get_labels_slide(pred_best_adjusted)
    stats_best_adj_predicted = extract_stats(
        pred_best_adjusted, predicted_labels=pred_best_adj_labels, metric="predicted"
    )
    stats_best_adj_all = extract_stats(pred_best_adjusted, predicted_labels=pred_best_adj_labels, metric="all")

    if is_final:
        pred_final_labels = get_labels_slide(pred_final)
        stats_final_predicted = extract_stats(pred_final, predicted_labels=pred_final_labels, metric="predicted")
        stats_final_all = extract_stats(pred_final, predicted_labels=pred_final_labels, metric="all")

        pred_final_adj_labels = get_labels_slide(pred_final_adjusted)
        stats_final_adj_predicted = extract_stats(
            pred_final_adjusted, predicted_labels=pred_final_adj_labels, metric="predicted"
        )
        stats_final_adj_all = extract_stats(pred_final_adjusted, predicted_labels=pred_final_adj_labels, metric="all")

    with pd.ExcelWriter(f"{out_dir}/stats.xlsx") as writer:
        stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
        stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
        stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
        stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)
        if is_final:
            stats_final_predicted.to_excel(writer, sheet_name="final_predicted", index=False)
            stats_final_all.to_excel(writer, sheet_name="final_all", index=False)
            stats_final_adj_predicted.to_excel(writer, sheet_name="final_adj_predicted", index=False)
            stats_final_adj_all.to_excel(writer, sheet_name="final_adj_all", index=False)

    logger.info("Secondary deconvolution process completed successfully.")
    logger.info(f"Training time: {TRAIN_TIME}")
