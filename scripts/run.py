from __future__ import annotations

import logging
import os
import time

import click
import scanpy as sc
import torch
from module.load_data import pp_prop
from module.run_model import run_sec_deconv
from tools.basics import format_time
from tools.hovernet_tools import extract_tiles_hovernet
from tools.hovernet_tools import map_cells_to_spots

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--adata_name", type=str, required=True, help="Name of the sample.")
@click.option("--json_path", type=str, required=True, help="Path to the post-segmentation file.")
@click.option("--image_path", type=str, required=True, help="Path to the high-quality WSI directory or image dict.")
@click.option("--path_st_adata", type=str, required=True, help="Path to the ST anndata object.")
@click.option("--proportions_file", type=str, required=True, help="Path to the proportions file.")
@click.option("--batch_size", type=int, default=1, help="Batch size for model training.")
@click.option("--lr", type=float, default=0.001, help="Learning rate.")
@click.option("--weights", is_flag=True, default=False, help="If True, the model uses a weighted loss.")
@click.option(
    "--agg",
    type=click.Choice(["proba", "onehot"], case_sensitive=False),
    default="proba",
    help="Aggregation of the probability vectors. Can be 'proba' or 'onehot'.",
)
@click.option(
    "--divergence",
    type=click.Choice(["l1", "l2", "kl", "rot"], case_sensitive=False),
    default="l1",
    help="Metric to use for divergence computation. Can be 'l1', 'l2', 'kl' or 'rot'.",
)
@click.option(
    "--reduction",
    type=click.Choice(["mean", "sum"], case_sensitive=False),
    default="mean",
    help="Aggregation parameter for loss computation. Can be 'mean' or 'sum'.",
)
@click.option("--alpha", type=float, default=0.5, help="Alpha parameter for loss function.")
@click.option("--epochs", type=int, default=25, help="Number of training epochs.")
@click.option("--train_size", type=float, default=0.5, help="Training set size as a fraction.")
@click.option("--val_size", type=float, default=0.25, help="Validation set size as a fraction.")
@click.option("--out_dir", type=str, default="results", help="Output directory.")
@click.option("--tb_dir", type=str, default="runs", help="Tensorboard directory.")
@click.option("--level", type=int, default=0, help="Image extraction level.")
@click.option("--size_edge", type=int, default=64, help="Edge size of the extracted tiles.")
@click.option("--dict_types", type=str, default=None, help="Dictionary of cell types to use for extraction.")
@click.option(
    "--save_images",
    type=click.Choice([None, "jpg", "dict", "both"], case_sensitive=False),
    default=None,
    help="'jpg' to save images, 'dict' to save dictionary, 'both' to save both.",
)
@click.option("--rs", type=int, default=42, help="Random seed")
def main(
    adata_name,
    json_path,
    image_path,
    path_st_adata,
    proportions_file,
    batch_size,
    lr,
    weights,
    agg,
    divergence,
    reduction,
    alpha,
    epochs,
    train_size,
    val_size,
    out_dir,
    tb_dir,
    level,
    size_edge,
    dict_types,
    save_images,
    rs,
):
    MAIN_START = time.time()

    size = (size_edge, size_edge)

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
            "jpg": (out_dir + "/extracted_images/", None),
            "dict": (None, out_dir + "/images_dict.pt"),
            "both": (out_dir + "/extracted_images/", out_dir + "/images_dict.pt"),
        }

        if save_images in save_options:
            img_dir, dict_dir = save_options[save_images]
        else:
            raise ValueError("save_images must be one of None, 'jpg', 'dict', or 'both'")
        try:
            image_dict = extract_tiles_hovernet(
                image_path,
                json_path,
                level=level,
                size=size,
                dict_types=dict_types,
                save_images=img_dir,
                save_dict=dict_dir,
            )
            logger.info("-> Image extraction completed successfully.")

        except Exception as e:
            logger.exception("Failed to extract images. Please check the image format and file paths.")
            raise ValueError(
                "If it's an image dictionary, it must be in .pt format.\n"
                "If it's a Whole-Slide Image, it must be in one of the following formats:\n"
                ".tif, .tiff, .svs, .dcm, or .ndpi."
            ) from e

    size = (image_dict["0"].shape[1], image_dict["0"].shape[1])

    logger.info(f"Loading spatial transcriptomics data from {path_st_adata}...")
    adata = sc.read_visium(path_st_adata)
    logger.info(f"Loading proportions from {proportions_file}...")
    proportions = pp_prop(proportions_file)
    logger.info("Cell Mapping...")
    logger.info("-> Mapping cells to the spot in which they are located...")
    spot_dict = map_cells_to_spots(adata, adata_name, json_path, only_in=True)
    logger.info("-> Mapping cells to the closest spot...")
    spot_dict_global = map_cells_to_spots(adata, adata_name, json_path, only_in=False)

    logger.info("=" * 50)
    logger.info("RUNNING SECONDARY DECONVOLUTION")
    logger.info("Parameters:")
    logger.info(f"Image size: {size}")
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

    run_sec_deconv(
        image_dict,
        spot_dict,
        spot_dict_global,
        proportions,
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
    main()
