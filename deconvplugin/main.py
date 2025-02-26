from __future__ import annotations

import json
import os
import time
from typing import List
from typing import Optional

import scanpy as sc
import torch
import typer
from loguru import logger

from deconvplugin.analysis.postseg import extract_tiles_hovernet
from deconvplugin.analysis.postseg import map_cells_to_spots
from deconvplugin.basics import format_time
from deconvplugin.dataset_utils import pp_prop
from deconvplugin.run_model import run_sec_deconv

app = typer.Typer()


@app.command()
def main(
    image_path: str = typer.Argument(..., help="Path to the high-quality WSI directory or image dict."),
    spot_prop_file: str = typer.Argument(..., help="Path to the proportions file."),
    json_path: Optional[str] = typer.Option(None, help="Path to the post-segmentation file."),
    path_st_adata: Optional[str] = typer.Option(None, help="Path to the ST anndata object."),
    adata_name: Optional[str] = typer.Option(None, help="Name of the sample."),
    spot_dict_file: Optional[str] = typer.Option(None, help="Path to the spot-to-cell json file."),
    model_name: str = typer.Option("resnet18", help="Type of model. Can be 'resnet18' or 'resnet50'."),
    hidden_dims: List[int] = typer.Option([512, 256], help="Hidden dimensions for the model."),
    batch_size: int = typer.Option(1, help="Batch size for model training."),
    lr: float = typer.Option(0.001, help="Learning rate."),
    weights: bool = typer.Option(False, help="If True, the model uses a weighted loss."),
    agg: str = typer.Option("proba", help="Aggregation of the probability vectors. Can be 'proba' or 'onehot'."),
    divergence: str = typer.Option(
        "l1", help="Metric to use for divergence computation. Can be 'l1', 'l2', 'kl', or 'rot'."
    ),
    reduction: str = typer.Option("mean", help="Aggregation parameter for loss computation. Can be 'mean' or 'sum'."),
    alpha: float = typer.Option(0.5, help="Alpha parameter for loss function."),
    epochs: int = typer.Option(25, help="Number of training epochs."),
    train_size: float = typer.Option(0.5, help="Training set size as a fraction."),
    val_size: float = typer.Option(0.25, help="Validation set size as a fraction."),
    out_dir: str = typer.Option("results", help="Output directory."),
    tb_dir: str = typer.Option("runs", help="Tensorboard directory."),
    level: int = typer.Option(0, help="Image extraction level."),
    edge_size: int = typer.Option(64, help="Edge size of the extracted tiles."),
    dict_types=typer.Option(None, help="Dictionary of cell types to use for organization when saving cell images."),
    save_images: Optional[str] = typer.Option(
        None, help="'jpg' to save images, 'dict' to save dictionary, 'both' to save both."
    ),
    rs: int = typer.Option(42, help="Random seed"),
):

    # Validate inputs
    valid_agg = {"proba", "onehot"}
    valid_divergence = {"l1", "l2", "kl", "rot"}
    valid_model_name = {"resnet18", "resnet50"}
    valid_reduction = {"mean", "sum"}

    if agg not in valid_agg:
        raise ValueError(f"Invalid value for 'agg': {agg}. Must be one of {valid_agg}.")
    if divergence not in valid_divergence:
        raise ValueError(f"Invalid value for 'divergence': {divergence}. Must be one of {valid_divergence}.")
    if model_name not in valid_model_name:
        raise ValueError(f"Invalid value for 'mtype': {model_name}. Must be one of {valid_model_name}.")
    if reduction not in valid_reduction:
        raise ValueError(f"Invalid value for 'reduction': {reduction}. Must be one of {valid_reduction}.")

    MAIN_START = time.time()

    size = (edge_size, edge_size)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"-> Created output directory: {out_dir}")

    if image_path.endswith(".pt"):
        if save_images is not None:
            logger.warning("save_images is ignored when loading an image dictionary.")
        logger.info(f"-> Loading image dictionary from {image_path}")
        image_dict = torch.load(image_path)

    else:
        logger.info(f"-> Extracting images from whole-slide image at {image_path}")
        save_options = {
            None: (None, None),
            "jpg": (os.path.join(out_dir, "extracted_images"), None),
            "dict": (None, os.path.join(out_dir, "image_dict.pt")),
            "both": (os.path.join(out_dir, "extracted_images"), os.path.join(out_dir, "image_dict.pt")),
        }

        if save_images in save_options:
            img_dir, dict_dir = save_options[save_images]
        else:
            raise ValueError("save_images must be one of None, 'jpg', 'dict', or 'both'")
        try:
            image_dict = extract_tiles_hovernet(
                image_path=image_path,
                json_path=json_path,
                level=level,
                size=size,
                dict_types=dict_types,
                save_images=img_dir,
                save_dict=dict_dir,
            )
            logger.info("-> Image extraction completed successfully.")

        except Exception as e:
            raise ValueError(
                "Failed to extract images. Please check the image format and file paths.\n"
                "If it's an image dictionary, it must be in .pt format.\n"
                "If it's a Whole-Slide Image, it must be in one of the following formats:\n"
                ".tif, .tiff, .svs, .dcm, or .ndpi."
            ) from e

    size = (image_dict["0"].shape[1], image_dict["0"].shape[1])

    logger.info(f"Loading spatial transcriptomics data from {path_st_adata}...")

    if path_st_adata is not None:
        adata = sc.read_visium(path_st_adata)

    logger.info(f"Loading proportions from {spot_prop_file}...")
    spot_prop_df = pp_prop(spot_prop_file)

    logger.info("Cell Mapping...")
    if spot_dict_file is not None and os.path.splitext(spot_dict_file)[1] == ".json":
        logger.info(f"Loading spot-to-cell dictionary from {spot_dict_file}...")
        with open(spot_dict_file) as json_file:
            spot_dict = json.load(json_file)
    else:
        logger.info("Mapping cells to the spot in which they are located...")
        spot_dict = map_cells_to_spots(adata, adata_name, json_path, only_in=True)

    logger.info("-> Mapping cells to the closest spot...")

    try:
        spot_dict_global = map_cells_to_spots(adata, adata_name, json_path, only_in=False)
    except Exception:
        logger.warning("Failed to map cells to the closest spot." "Have you provided adata, adata_name, and json_path?")
        spot_dict_global = None

    # Recap variables
    logger.info("=" * 50)
    logger.info("RUNNING SECONDARY DECONVOLUTION")
    logger.info("Parameters:")
    logger.info(f"Image size: {size}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Hidden dims: {hidden_dims}")
    logger.info(f"Batch size (#spots): {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weighted loss: {weights}")
    logger.info(f"Aggregation: {agg}")
    logger.info(f"Divergence: {divergence}")
    logger.info(f"Reduction: {reduction}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Random state: {rs}")
    logger.info("=" * 50)

    # Run secondary deconvolution
    run_sec_deconv(
        image_dict=image_dict,
        spot_dict=spot_dict,
        spot_dict_global=spot_dict_global,
        spot_prop_df=spot_prop_df,
        model_name=model_name,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        lr=lr,
        weights=weights,
        agg=agg,
        divergence=divergence,
        reduction=reduction,
        alpha=alpha,
        epochs=epochs,
        train_size=train_size,
        val_size=val_size,
        out_dir=out_dir,
        tb_dir=tb_dir,
        rs=rs,
    )
    TOTAL_TIME = format_time(time.time() - MAIN_START)
    logger.info(f"Total time: {TOTAL_TIME}")


if __name__ == "__main__":
    app()
