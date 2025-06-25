from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger
from torchvision.transforms.v2 import CenterCrop
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.v2 import ToDtype
from torchvision.transforms.v2 import ToImage


def is_port_available(port: int):
    """Check if a port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def find_available_port(start: int = 1024, end: int = 65535):
    """Find an available port in the given range."""
    ports = np.arange(start, end)
    np.random.shuffle(ports)
    for port in ports:
        if is_port_available(port):
            print(f"Using port : {port}")
            return port
    raise RuntimeError("No available port found in the range.")


def image_dict_to_h5(image_dict_path, output_path):
    """
    Convert an image_dict to an HDF5 file.

    Args:
        image_dict (dict): A dictionary where keys are image IDs and values are torch tensors.
        output_path (str): Path to save the HDF5 file.
    """
    image_dict = torch.load(image_dict_path)
    images = [img.numpy() for img in image_dict.values()]
    images = np.stack(images)
    images = np.transpose(images, (0, 2, 3, 1))
    images = images.astype(np.uint8)
    print(images.shape)

    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset("img", data=images, dtype="uint8")

    print(f"Saved HDF5 file to {output_path}")


def load_eval_transform(stats_path: Path) -> Compose:
    with open(stats_path, "r") as f:
        norm_dict = json.load(f)
    mean = norm_dict["mean"]
    std = norm_dict["std"]
    logger.info("Loaded mean/std normalisation.")
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=mean, std=std),
            CenterCrop(size=(48, 48)),
        ]
    )


def run_moco_script(
    tag: str,
    list_slides: list[str],
    path_dataset: str,
    n_gpus: int,
    n_cpus_per_gpu: int,
):
    # Define the base path to the script
    script_path = os.path.abspath(os.path.join("external", "mocov3", "main_moco.py"))
    model_output_path = os.path.abspath(os.path.join("models", "ssl"))
    os.makedirs(model_output_path, exist_ok=True)

    default_port = find_available_port()

    # Construct the command as a list of arguments
    command = [
        "python3",
        script_path,
        "-b",
        "2048",
        "--epochs",
        "150",
        "--workers",
        str(n_gpus * n_cpus_per_gpu),
        "--dist-url",
        f"tcp://localhost:{default_port}",
        "--world-size",
        "1",
        "--multiprocessing-distributed",
        "--rank",
        "0",
        "--tag",
        tag,
        "--n_cell_max",
        "1_000_000",
        "--list_slides",
        *list_slides,
        "--data_path",
        path_dataset,
        "--output_path",
        model_output_path,
    ]

    # Run the command
    try:
        _ = subprocess.run(command, check=True, text=True)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")


def run_ssl(
    path_dataset: str,
    organ: str | None,
    ids_to_query: list[str] | None,
    tag: str,
    n_gpus: int,
    n_cpus_per_gpu: int,
) -> None:
    assert (organ is not None) ^ (
        ids_to_query is not None
    ), f"Only one should not be none, got: organ={organ} and ids_to_query={ids_to_query}"
    if ids_to_query:
        logger.info(f"Working with preselected {ids_to_query} slides...")
    else:
        raise ValueError("You must provide a list of slide IDs.")
    run_moco_script(
        tag=tag,
        list_slides=ids_to_query,
        path_dataset=path_dataset,
        n_gpus=n_gpus,
        n_cpus_per_gpu=n_cpus_per_gpu,
    )
