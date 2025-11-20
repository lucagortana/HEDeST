from __future__ import annotations

import argparse
import os
from typing import Optional

from loguru import logger

from external.hovernet.extract_cell_images import extract_images_hn
from external.mocov3.run_infer import infer_embed
from external.mocov3.utils import image_dict_to_h5
from external.mocov3.utils import run_ssl


def main(
    image_path: str,
    save_path: str,
    tag: str,
    json_path: Optional[str] = None,
    batch_size_infer: int = 2048,
    num_workers: int = 4,
) -> None:
    """
    Main function to run SSL on a dataset.

    Args:
        image_path: Path to the image dict or WSI.
        save_path: Folder to save the results.
        tag: Tag for the run.
        json_path: Path to the json file for segmentation.
        batch_size_infer: Batch size for inference.
        num_workers: Number of workers for data loading.
    """

    # Check your image path
    if image_path.endswith(".pt"):
        logger.info(f"Your image path is a dictionary: {image_path}")
    else:
        logger.info(f"-> Extracting images from whole-slide image at {image_path}")

        if json_path is not None:
            try:
                _ = extract_images_hn(
                    image_path=image_path,
                    json_path=json_path,
                    save_images=None,
                    save_dict=os.path.join(save_path, "image_dict.pt"),
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
        else:
            raise ValueError(
                "Please provide a segmentation file in json_path for image extraction "
                "or ensure the image path is a .pt file."
            )

        image_path = os.path.join(save_path, "image_dict.pt")

    # Save h5 file
    h5_folder = os.path.join(save_path, "cell_images")
    if not os.path.exists(h5_folder):
        os.makedirs(h5_folder)

    sample_id = "slide1"
    h5_path = os.path.join(h5_folder, f"{sample_id}.h5")
    image_dict_to_h5(image_path, h5_path)

    # Run SSL with Moco-v3
    run_ssl(save_path, None, [sample_id], tag, 1, 4)

    # Run inference to get embeddings
    logger.info("-> Running inference to get embeddings")
    infer_embed(image_path, save_path, tag, model_name="resnet50", batch_size=batch_size_infer, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SSL on a dataset")
    parser.add_argument("--image_path", type=str, help="Path to the image dict or WSI")
    parser.add_argument("--save_path", type=str, help="Folder to save the results")
    parser.add_argument("--tag", type=str, help="Tag for the run")
    parser.add_argument("--json_path", type=str, default=None, help="Path to the json file for segmentation")
    parser.add_argument("--batch_size_infer", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args.image_path, args.save_path, args.tag, args.json_path, args.batch_size_infer, args.num_workers)
