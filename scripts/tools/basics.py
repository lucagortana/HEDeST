from __future__ import annotations

import io
import random

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from module.cell_classifier import CellClassifier
from PIL import Image


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators in Python libraries.

    Parameters:
        seed (int): The seed value to set for random number generators.

    Example:
        >>> set_seed(42)
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, size_edge: int, num_classes: int) -> CellClassifier:
    """
    Loads a trained model from a file.

    Parameters:
    - model_path (str): The path to the model file.
    - size_edge (int): The size of the image edge.
    - num_classes (int): The number of classes.

    Returns:
    - model (CellClassifier): The trained model.

    Example:
    >>> model = load_model("results/model.pth")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device found to load the model : ", device)

    model = CellClassifier(size_edge=size_edge, num_classes=num_classes, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    if device == torch.device("cuda"):
        model = model.to(device)

    return model


def fig_to_array(fig):
    """
    Convert a Matplotlib figure to a NumPy array (image array) that can be displayed using imshow.
    """
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def check_json_classification(dict):
    first_key = next(iter(dict["nuc"]))
    if dict["nuc"][first_key]["type"] is None:
        return False
    return True
