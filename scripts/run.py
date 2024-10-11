from __future__ import annotations

import os

import click
import scanpy as sc
import torch
from module.load_data import pp_prop
from module.run_model import run_sec_deconv
from tools.hovernet_tools import extract_tiles_hovernet
from tools.hovernet_tools import map_cells_to_spots

adata_name = "CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"
data_path = f"../data/{adata_name}/"
json_path = data_path + "seg_json/pannuke_fast_mask_lvl3.json"
image_path = data_path + "CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_tissue_image.tif"
path_ST_adata = data_path + "ST/"


@click.command()
@click.option("--adata_name", type=str, required=True, help="Name of the sample")
@click.option("--json_path", type=str, required=True, help="Path to the post-segmentation file")
@click.option("--image_path", type=str, default=None, help="Path to the high-quality WSI directory")
@click.option("--path_st_adata", type=str, required=True, help="Path to the ST anndata object")
@click.option("--proportions_file", type=str, required=True, help="Path to the proportions file")
@click.option("--batch_size", type=int, default=8, help="Batch size for model training")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--agg_loss", type=str, default="mean", help="Aggregation loss function type")
@click.option("--alpha", type=float, default=0.5, help="Alpha parameter for loss function")
@click.option("--epochs", type=int, default=25, help="Number of training epochs")
@click.option("--train_size", type=float, default=0.5, help="Training set size as a fraction")
@click.option("--val_size", type=float, default=0.25, help="Validation set size as a fraction")
@click.option("--out_dir", type=str, default="models", help="Output directory")
@click.option("--rs", type=int, default=42, help="Random seed")
@click.option("--image_dict_path", type=str, default=None, help="Optional path to pre-saved image_dict file")
@click.option("--level", type=int, default=0, help="Image extraction level")
@click.option("--size", type=(int, int), default=(64, 64), help="Size of the extracted tiles")
@click.option("--dict_types", type=str, default=None, help="Dictionary of cell types to use for extraction")
@click.option("--save_images", type=str, default=None, help="Directory to save extracted images")
def main(
    adata_name,
    json_path,
    image_path,
    path_st_adata,
    proportions_file,
    batch_size,
    lr,
    agg_loss,
    alpha,
    epochs,
    train_size,
    val_size,
    out_dir,
    rs,
    image_dict_path,
    level,
    size,
    dict_types,
    save_images,
):

    if image_dict_path is not None and os.path.exists(image_dict_path):
        print(f"Loading image_dict from {image_dict_path}")
        image_dict = torch.load(image_dict_path)
    elif image_path is not None:
        print("Image_dict not provided or path does not exist, extracting using HoverNet segmentation results.")
        image_dict = extract_tiles_hovernet(
            image_path, json_path, level=level, size=size, dict_types=dict_types, save_images=save_images
        )
    else:
        raise ValueError("Either --image_dict_path or --image_path must be provided.")

    adata = sc.read_visium(path_st_adata)
    proportions = pp_prop(proportions_file)
    spot_dict = map_cells_to_spots(adata, adata_name, json_path)

    run_sec_deconv(
        image_dict,
        spot_dict,
        proportions,
        batch_size=batch_size,
        lr=lr,
        agg_loss=agg_loss,
        alpha=alpha,
        epochs=epochs,
        train_size=train_size,
        val_size=val_size,
        out_dir=out_dir,
        rs=rs,
    )


if __name__ == "__main__":
    main()
