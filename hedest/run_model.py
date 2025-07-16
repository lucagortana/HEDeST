from __future__ import annotations

import os
import pickle
import time
from typing import Dict
from typing import List

import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch import optim

from hedest.dataset import SpotDataset
from hedest.dataset import SpotEmbedDataset
from hedest.dataset_utils import custom_collate
from hedest.dataset_utils import get_transform
from hedest.dataset_utils import split_data
from hedest.model.cell_classifier import CellClassifier
from hedest.predict import predict_slide
from hedest.prob_adjust import PPSAdjustment
from hedest.trainer import ModelTrainer
from hedest.utils import format_time
from hedest.utils import set_seed

# from hedest.analysis.pred_analyzer import PredAnalyzer


def run_hedest(
    image_dict: Dict[str, torch.Tensor],
    spot_prop_df: pd.DataFrame,
    json_path: str,
    adata: AnnData,
    adata_name: str,
    spot_dict: Dict[str, List[str]],
    model_name: str = "resnet18",
    hidden_dims: List[int] = [512, 256],
    batch_size: int = 64,
    lr: float = 0.0001,
    divergence: str = "l2",
    alpha: float = 0.5,
    beta: float = 0.0,
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
        model_name: Name of the model to use.
        batch_size: Batch size for data loaders.
        lr: Learning rate for the optimizer.
        divergence: Type of divergence loss to use ("l1", "l2", "kl", "rot").
        alpha: Weighting factor for the loss function.
        beta: Weighting factor for the Bayesian adjustment.
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
    if model_name == "quick":
        train_dataset = SpotEmbedDataset(train_spot_dict, train_proportions, image_dict)
        val_dataset = SpotEmbedDataset(val_spot_dict, val_proportions, image_dict)
        test_dataset = SpotEmbedDataset(test_spot_dict, test_proportions, image_dict)

    else:
        transform = get_transform(model_name)
        train_dataset = SpotDataset(train_spot_dict, train_proportions, image_dict, transform)
        val_dataset = SpotDataset(val_spot_dict, val_proportions, image_dict, transform)
        test_dataset = SpotDataset(test_spot_dict, test_proportions, image_dict, transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )

    num_classes = spot_prop_df.shape[1]
    ct_list = list(spot_prop_df.columns)

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
        divergence=divergence,
        alpha=alpha,
        # beta=beta,
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

    # Prior Probability Shift adjustment
    logger.info("Starting Bayesian adjustment...")
    p_c = spot_prop_df.loc[list(train_spot_dict.keys())].mean(axis=0)

    cell_prob_best_adjusted = PPSAdjustment(
        cell_prob_best,
        spot_dict,
        spot_prop_df,
        p_c,
        adata=adata,
        adata_name=adata_name,
        json_path=json_path,
        beta=beta,
        device=device,
    ).adjust()
    if is_final:
        cell_prob_final_adjusted = PPSAdjustment(
            cell_prob_final,
            spot_dict,
            spot_prop_df,
            p_c,
            adata=adata,
            adata_name=adata_name,
            json_path=json_path,
            beta=beta,
            device=device,
        ).adjust()

    # Save model infos
    model_info = {
        "model_name": model_name,
        "hidden_dims": hidden_dims,
        "spot_dict": spot_dict,
        "train_spot_dict": train_spot_dict,  # plus tard mettre que les clés
        "proportions": spot_prop_df,
        "history": {"train": trainer.history_train, "val": trainer.history_val},
        "preds": {
            "pred_best": cell_prob_best,
            "pred_best_adjusted": cell_prob_best_adjusted,  # cell_prob_best_adjusted
            **({"pred_final": cell_prob_final, "pred_final_adjusted": cell_prob_final_adjusted} if is_final else {}),
        },
    }

    info_dir = os.path.join(out_dir, "info.pickle")
    logger.info(f"Saving objects to {info_dir}...")
    with open(info_dir, "wb") as f:
        pickle.dump(model_info, f)

    # # Extract and save statistics
    # logger.info("Extracting and saving statistics...")

    # stats_best_predicted = PredAnalyzer(model_info=model_info).extract_stats(metric="predicted")
    # stats_best_all = PredAnalyzer(model_info=model_info).extract_stats(metric="all")

    # stats_best_adj_predicted = PredAnalyzer(model_info=model_info, adjusted=True).extract_stats(metric="predicted")
    # stats_best_adj_all = PredAnalyzer(model_info=model_info, adjusted=True).extract_stats(metric="all")

    # if is_final:
    #     stats_final_predicted = PredAnalyzer(model_info=model_info, model_state="final").extract_stats(
    #         metric="predicted"
    #     )
    #     stats_final_all = PredAnalyzer(model_info=model_info, model_state="final").extract_stats(metric="all")

    #     stats_final_adj_predicted = PredAnalyzer(
    #         model_info=model_info, model_state="final", adjusted=True
    #     ).extract_stats(metric="predicted")
    #     stats_final_adj_all = PredAnalyzer(model_info=model_info, model_state="final", adjusted=True).extract_stats(
    #         metric="all"
    #     )

    # with pd.ExcelWriter(os.path.join(out_dir, "stats.xlsx")) as writer:
    #     stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
    #     stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
    #     stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
    #     stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)
    #     if is_final:
    #         stats_final_predicted.to_excel(writer, sheet_name="final_predicted", index=False)
    #         stats_final_all.to_excel(writer, sheet_name="final_all", index=False)
    #         stats_final_adj_predicted.to_excel(writer, sheet_name="final_adj_predicted", index=False)
    #         stats_final_adj_all.to_excel(writer, sheet_name="final_adj_all", index=False)

    logger.info("Secondary deconvolution process completed successfully.")
    logger.info(f"Training time: {TRAIN_TIME}")
