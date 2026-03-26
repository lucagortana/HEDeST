from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch import optim

from hedest.analysis.pred_analyzer import PredAnalyzer
from hedest.dataset import SpotDataset
from hedest.dataset import SpotEmbedDataset
from hedest.dataset_utils import custom_collate
from hedest.dataset_utils import get_transform
from hedest.dataset_utils import split_data
from hedest.model.cell_classifier import CellClassifier
from hedest.ppsa import PPSAdjustment
from hedest.ppsa import PPSANaive
from hedest.predict import predict_slide
from hedest.trainer import ModelTrainer
from hedest.utils import format_time
from hedest.utils import set_seed
from hedest.run_model import run_hedest
#import torch.multiprocessing as mp


def run_sequential_hedest(    image_dict: Dict[str, torch.Tensor],
    spot_prop_df_list: List[pd.DataFrame],
    spot_dict: Dict[str, List[str]],
    json_path: Optional[str] = None,
    adata: Optional[AnnData] = None,
    adata_name: Optional[str] = None,
    model_name: str = "default",
    hidden_dims: List[int] = [512, 256],
    norm: bool = False,
    dropout: float = 0.0,
    batch_size: int = 64,
    lr: float = 0.0001,
    divergence: str = "l2",
    alpha: float = 0.0,
    beta: float = 0.0,
    epochs: int = 60,
    train_size: float = 0.5,
    val_size: float = 0.25,
    out_dir: str = "results",
    save_geojson: bool = False,
    color_dict_file: Optional[str] = None,
    rs: int = 42,
) -> None:
    """
       Runs sequential HEDeST for cell classification.

       Args:
           image_dict: Dictionary mapping cell IDs to image tensors.
           spot_prop_df_list: List of dataFrame containing cell type proportions for each spot.
           spot_dict: Dictionary mapping cell IDs to their spot.
           json_path: Path to the post-segmentation file.
           adata: AnnData object containing spatial transcriptomics data.
           adata_name: Name of the sample in the AnnData object.
           model_name: Name of the model to use.
           hidden_dims: List of hidden layer dimensions.
           norm: Whether to add a LayerNorm layer.
           dropout: Dropout rate.
           batch_size: Batch size for data loaders.
           lr: Learning rate for the optimizer.
           divergence: Type of divergence loss to use ("l1", "l2", "kl", "rot").
           alpha: Weighting factor for the loss function.
           beta: Weighting factor for the Bayesian adjustment.
           epochs: Number of training epochs.
           train_size: Proportion of data used for training.
           val_size: Proportion of data used for validation.
           out_dir: Directory to save results.
           save_geojson: Whether to export a GeoJSON file for QuPath.
           color_dict_file: Path to a YAML color dict (special format).
           rs: Random seed for reproducibility.
       """

    for cell_annot_level, spot_prop_df in enumerate(spot_prop_df_list):
        run_hedest(image_dict=image_dict, spot_prop_df=spot_prop_df,
                   cell_annot_level=cell_annot_level, spot_dict=spot_dict,
                   json_path=json_path, adata=adata, adata_name=adata_name,
                   model_name=model_name, hidden_dims=hidden_dims, norm=norm,
                   dropout=dropout, batch_size=batch_size, lr=lr,
                   divergence=divergence, alpha=alpha, beta=beta,
                   epochs=epochs, train_size=train_size, val_size=val_size,
                   out_dir=out_dir, save_geojson=save_geojson,
                   color_dict_file=color_dict_file, rs=rs)
