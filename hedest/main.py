from __future__ import annotations

import json
import os
import time
from typing import List
from typing import Optional

import torch
import typer
from loguru import logger

from external.hovernet.extract_cell_images import extract_images_hn
from hedest.analysis.postseg import map_cells_to_spots
from hedest.dataset_utils import pp_prop
from hedest.run_model import run_sec_deconv
from hedest.utils import format_time
from hedest.utils import load_spatial_adata

app = typer.Typer()


def parse_hidden_dims(hidden_dims: str) -> List[int]:
    try:
        return [int(dim) for dim in hidden_dims.split(",")]
    except ValueError:
        typer.echo("'hidden_dims' should be a str chain of integers separated by comas (e.g. '512,258').", err=True)
        raise typer.Abort()


@app.command()
def main(
    image_path: str = typer.Argument(..., help="Path to the high-quality WSI directory or image dict."),
    spot_prop_file: str = typer.Argument(..., help="Path to the proportions file."),
    json_path: str = typer.Argument(..., help="Path to the post-segmentation file."),
    path_st_adata: str = typer.Argument(..., help="Path to the ST anndata object."),
    adata_name: str = typer.Argument(..., help="Name of the sample."),
    spot_dict_file: Optional[str] = typer.Option(None, help="Path to the spot-to-cell json file."),
    model_name: str = typer.Option("resnet18", help="Type of model. Can be 'resnet18' or 'resnet50'."),
    hidden_dims: str = typer.Option("256,128", help="Hidden dimensions for the model (comma-separated)."),
    batch_size: int = typer.Option(64, help="Batch size for model training."),
    lr: float = typer.Option(0.0001, help="Learning rate."),
    divergence: str = typer.Option(
        "l2", help="Metric to use for divergence computation. Can be 'l1', 'l2', 'kl', or 'rot'."
    ),
    alpha: float = typer.Option(0.5, help="Alpha parameter for loss function."),
    beta: float = typer.Option(0.0, help="Beta parameter for bayesian adjustment."),
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

    hidden_dims = parse_hidden_dims(hidden_dims)

    # Validate inputs
    valid_divergence = {"l1", "l2", "kl"}
    valid_model_name = {"convnet", "resnet18", "resnet50", "quick"}

    if divergence not in valid_divergence:
        raise ValueError(f"Invalid value for 'divergence': {divergence}. Must be one of {valid_divergence}.")
    if model_name not in valid_model_name:
        raise ValueError(f"Invalid value for 'model_name': {model_name}. Must be one of {valid_model_name}.")

    MAIN_START = time.time()

    size = (edge_size, edge_size)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"-> Created output directory: {out_dir}")

    # Image data loading
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
            image_dict = extract_images_hn(
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
                "Failed to extract images. Please check the image format.\n"
                "It must be in one of the following formats:\n"
                ".tif, .tiff, .svs, .dcm, or .ndpi.\n"
                "Also, ensure that the json_path is correct and contains "
                "valid segmentation data."
            ) from e

    example_img = image_dict[list(image_dict.keys())[0]]
    try:
        size = (example_img.shape[1], example_img.shape[1])
    except Exception:
        size = example_img.shape[0]

    # Load spot information
    logger.info(f"Loading proportions from {spot_prop_file}...")
    spot_prop_df = pp_prop(spot_prop_file)

    logger.info(f"Loading spatial transcriptomics data from {path_st_adata}...")
    adata = load_spatial_adata(path_st_adata)

    logger.info("Cell Mapping...")
    if spot_dict_file is not None and os.path.splitext(spot_dict_file)[1] == ".json":
        logger.info(f"Loading spot-to-cell dictionary from {spot_dict_file}...")
        with open(spot_dict_file) as json_file:
            spot_dict = json.load(json_file)
    else:
        logger.info("Mapping cells to the spot in which they are located...")
        spot_dict = map_cells_to_spots(adata, adata_name, json_path)

    # Recap variables
    logger.info("=" * 50)
    logger.info("RUNNING SECONDARY DECONVOLUTION")
    logger.info("Parameters:")
    logger.info(f"Image size: {size}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Hidden dims: {hidden_dims}")
    logger.info(f"Batch size (#spots): {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Divergence: {divergence}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Beta: {beta}")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Random state: {rs}")
    logger.info("=" * 50)

    # Run secondary deconvolution
    run_sec_deconv(
        image_dict=image_dict,
        spot_prop_df=spot_prop_df,
        json_path=json_path,
        adata=adata,
        adata_name=adata_name,
        spot_dict=spot_dict,
        model_name=model_name,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        lr=lr,
        divergence=divergence,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        train_size=train_size,
        val_size=val_size,
        out_dir=out_dir,
        tb_dir=tb_dir,
        rs=rs,
    )
    TOTAL_TIME = format_time(time.time() - MAIN_START)
    logger.info(f"Total time: {TOTAL_TIME}\n")


if __name__ == "__main__":
    app()
