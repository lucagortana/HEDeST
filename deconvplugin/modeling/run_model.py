from __future__ import annotations

import logging
import os
import pickle
import time

import pandas as pd
import torch
from torch import optim

from deconvplugin.analysis import PredAnalyzer
from deconvplugin.basics import format_time
from deconvplugin.basics import set_seed
from deconvplugin.dataset import split_data
from deconvplugin.dataset import SpotDataset
from deconvplugin.modeling.bayes_adjust import BayesianAdjustment
from deconvplugin.modeling.cell_classifier import CellClassifier
from deconvplugin.modeling.predict import predict_slide
from deconvplugin.modeling.trainer import ModelTrainer

# from module.cell_classifier import CellClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# def run_pri_deconv()


def run_sec_deconv(
    image_dict,
    spot_dict,
    spot_dict_global,
    proportions,
    mtype="convnet",
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
    tb_dir="runs",
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
    model = CellClassifier(size_edge=size_edge, num_classes=num_classes, mtype=mtype, device=device)
    model = model.to(device)
    logger.info(f"-> {num_classes} classes detected.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = ModelTrainer(
        model,
        ct_list,
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
    model4pred_best = CellClassifier(size_edge=size_edge, num_classes=num_classes, mtype=mtype, device=device)
    model4pred_best.load_state_dict(torch.load(trainer.best_model_path))
    pred_best = predict_slide(model4pred_best, image_dict, ct_list)

    is_final = True
    try:
        model4pred_final = CellClassifier(size_edge=size_edge, num_classes=num_classes, mtype=mtype, device=device)
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
        "mtype": mtype,
        "spot_dict": spot_dict,
        "proportions": proportions,
        "history": {"train": trainer.history_train, "val": trainer.history_val},
        "preds": {"pred_best": pred_best, "pred_best_adjusted": pred_best_adjusted},
    }
    if is_final:
        info["preds"]["pred_final"] = pred_final
        info["preds"]["pred_final_adjusted"] = pred_final_adjusted
    info_dir = os.path.join(out_dir, "info.pickle")
    logger.info(f"Saving objects to {info_dir}...")
    with open(info_dir, "wb") as f:
        pickle.dump(info, f)

    logger.info("Extracting and saving statistics...")

    stats_best_predicted = PredAnalyzer(model_info=info).extract_stats(metric="predicted")
    stats_best_all = PredAnalyzer(model_info=info).extract_stats(metric="all")

    stats_best_adj_predicted = PredAnalyzer(model_info=info, adjusted=True).extract_stats(metric="predicted")
    stats_best_adj_all = PredAnalyzer(model_info=info, adjusted=True).extract_stats(metric="all")

    if is_final:
        stats_final_predicted = PredAnalyzer(model_info=info, model_state="final").extract_stats(metric="predicted")
        stats_final_all = PredAnalyzer(model_info=info, model_state="final").extract_stats(metric="all")

        stats_final_adj_predicted = PredAnalyzer(model_info=info, model_state="final", adjusted=True).extract_stats(
            metric="predicted"
        )
        stats_final_adj_all = PredAnalyzer(model_info=info, model_state="final", adjusted=True).extract_stats(
            metric="all"
        )

    with pd.ExcelWriter(os.path.join(out_dir, "stats.xlsx")) as writer:
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
