from __future__ import annotations

import random

import numpy as np
import torch
from module.cell_classifier import CellClassifier


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


def load_model(model_path: str, num_classes: int) -> CellClassifier:
    """
    Loads a trained model from a file.

    Parameters:
    - model_path (str): The path to the model file.

    Returns:
    - model (CellClassifier): The trained model.

    Example:
    >>> model = load_model("models/model.pth")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device found to load the model : ", device)

    model = CellClassifier(num_classes=num_classes, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    if device == torch.device("cuda"):
        model = model.to(device)

    return model
