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
@click.option("--image_path", type=str, required=True, help="Path to the high-quality WSI directory or image dict")
@click.option("--path_st_adata", type=str, required=True, help="Path to the ST anndata object")
@click.option("--proportions_file", type=str, required=True, help="Path to the proportions file")
@click.option("--batch_size", type=int, default=8, help="Batch size for model training")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--agg_loss", type=str, default="mean", help="Aggregation loss function type")
@click.option("--alpha", type=float, default=0.5, help="Alpha parameter for loss function")
@click.option("--epochs", type=int, default=25, help="Number of training epochs")
@click.option("--train_size", type=float, default=0.5, help="Training set size as a fraction")
@click.option("--val_size", type=float, default=0.25, help="Validation set size as a fraction")
@click.option("--out_dir", type=str, default="results", help="Output directory")
@click.option("--rs", type=int, default=42, help="Random seed")
@click.option("--level", type=int, default=0, help="Image extraction level")
@click.option("--size_edge", type=int, default=64, help="Edge size of the extracted tiles")
@click.option("--dict_types", type=str, default=None, help="Dictionary of cell types to use for extraction")
@click.option(
    "--save_images", type=str, default=None, help="'jpg' to save images, 'dict' to save dictionary, 'both' to save both"
)
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
    level,
    size_edge,
    dict_types,
    save_images,
):

    size = (size_edge, size_edge)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if image_path.endswith(".pt"):
        if save_images is not None:
            print("Warning: save_images is ignored when loading an image dictionary.")
        image_dict = torch.load(image_path)

    else:
        save_options = {
            None: (None, None),
            "jpg": (out_dir + "/extracted_images", None),
            "dict": (None, out_dir + "/image_dict.pt"),
            "both": (out_dir + "/extracted_images", out_dir + "/image_dict.pt"),
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
        except Exception as e:
            raise ValueError(
                "If it's an image dictionary, it must be in .pt format.\n"
                "If it's a Whole-Slide Image, it must be in one of the following formats:\n"
                ".tif, .tiff, .svs, .dcm, or .ndpi."
            ) from e

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
