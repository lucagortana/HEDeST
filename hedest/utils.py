from __future__ import annotations

import io
import os
import random
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from hedest.model.cell_classifier import CellClassifier


def set_seed(seed: int) -> None:
    """
    Sets the seed for random number generators in Python libraries.

    Args:
        seed: The seed value to set for random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, model_name: str, num_classes: int, hidden_dims: List[int]) -> CellClassifier:
    """
    Loads a trained model from a file.

    Args:
        model_path: Path to the model file.
        model_name: Name of the model architecture.
        num_classes: Number of classes in the model.
        hidden_dims: List of hidden layer dimensions.

    Returns:
        The loaded model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device found to load the model : ", device)

    model = CellClassifier(model_name=model_name, num_classes=num_classes, hidden_dims=hidden_dims, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    if device == torch.device("cuda"):
        model = model.to(device)

    return model


def load_spatial_adata(path: str):
    """
    Loads spatial transcriptomics data with Scanpy.

    Args:
        path: Path to an `.h5ad` file **or** a Visium directory.

    Returns:
        adata: The loaded AnnData object.
    """

    try:
        if os.path.isfile(path):
            return sc.read_h5ad(path)
        else:
            raise FileNotFoundError(f"'{path}' is not a valid file path.")
    except Exception as h5ad_error:
        try:
            return sc.read_visium(path)
        except Exception as visium_error:
            raise RuntimeError("Failed to load data with either sc.read_h5ad or sc.read_visium") from (
                h5ad_error or visium_error
            )


def count_cell_types(seg_dict: Dict[str, Any], ct_list: List[str]) -> pd.DataFrame:
    """
    Counts cell types in the segmentation dictionary.

    Args:
        seg_dict: Dictionary containing segmentation data.
        ct_list: List of cell type names.

    Returns:
        DataFrame containing counts of each cell type.
    """

    cell_type_counts = {}
    nuc = seg_dict["nuc"]
    for cell_id in nuc.keys():
        label = nuc[cell_id]["type"]
        cell_type = ct_list[int(label)]
        if cell_type not in cell_type_counts.keys():
            cell_type_counts[cell_type] = 1
        else:
            cell_type_counts[cell_type] += 1
    df = pd.DataFrame([cell_type_counts])

    return df


def fig_to_array(fig: Figure) -> np.ndarray:
    """
    Converts a Matplotlib figure to a NumPy array.

    Args:
        fig: A Matplotlib figure.

    Returns:
        An image array.
    """

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def check_json_classification(data: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> bool:
    """
    Checks if all cells in the JSON classification data have a non-None type.

    Args:
        data: A nested dictionary containing classification data.

    Returns:
        True if all cells have types, False otherwise.
    """

    first_key = next(iter(data["nuc"]))
    return data["nuc"][first_key]["type"] is not None


def seg_colors_compatible(
    seg_dict: Dict[str, Dict[str, Dict[str, Union[str, int]]]], color_dict: Dict[str, Tuple[str, Tuple[int, int, int]]]
) -> bool:
    """
    Checks if segmentation labels are compatible with color dictionary.

    Args:
        seg_dict: Segmentation data dictionary.
        color_dict: Color dictionary.

    Returns:
        True if all segmentation labels have corresponding colors, False otherwise.
    """

    seg_labels = set(str(cell["type"]) for cell_data in seg_dict["nuc"].values() for cell in [cell_data])
    color_labels = set(color_dict.keys())

    return (seg_labels - color_labels) == set()


def format_time(seconds: int) -> str:
    """
    Formats time duration in HH:MM:SS or MM:SS format.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted time string.
    """

    formatted_time = str(timedelta(seconds=int(seconds)))
    if seconds < 3600:
        formatted_time = formatted_time[2:]
    return formatted_time


def revert_dict(data: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Reverts a dictionary with lists as values to a dictionary mapping values to keys.

    Args:
        data: Input dictionary.

    Returns:
        Reverted dictionary.
    """

    return {val: key for key, values in data.items() for val in values}


def remove_empty_keys(data: Dict[str, List]) -> Dict[str, List]:
    """
    Removes keys with empty lists from a dictionary.

    Args:
        data: Input dictionary.

    Returns:
        Dictionary without empty keys.
    """

    empty_keys = []
    for key, value in data.items():
        if value == []:
            empty_keys.append(key)

    for element in empty_keys:
        del data[element]

    return data


def generate_color_dict(
    labels: List[str], palette: str = "tab20", format: str = "classic", n_max: int = 40
) -> Dict[str, Union[Tuple, Tuple[str, List[int]]]]:
    """
    Generates a dictionary of colors for labels.

    Args:
        labels: List of class labels.
        palette: Matplotlib color palette.
        format: Output format - "classic" or "special".
        n_max: Maximum number of unique colors.

    Returns:
        Color dictionary.
    """

    if len(labels) > n_max:
        print("Warning: The number of classes is greater than the maximum number of colors available in the palette.")
        print("The colors will be repeated.")

    num_classes = len(labels)
    cmap = plt.get_cmap(palette)

    if format == "classic":
        return {labels[i]: cmap(i % n_max) for i in range(num_classes)}

    elif format == "special":
        color_dict = {}
        for i, class_name in enumerate(labels):
            color = cmap(i % n_max)
            color = [int(255 * c) for c in color]
            color_dict[str(i)] = [class_name, color]

        return color_dict

    else:
        raise ValueError("Format must be either 'classic' or 'special'.")


def require_attributes(*required_attributes: str) -> Callable:
    """
    Decorator to ensure required attributes of a class are not None.

    Args:
        *required_attributes: Names of required attributes.

    Returns:
        Callable: Decorated function.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            missing_attrs = [attr for attr in required_attributes if getattr(self, attr, None) is None]
            if missing_attrs:
                raise ValueError(
                    f"Your object contains NoneType attribute(s): {', '.join(missing_attrs)}. "
                    "Please add them with the add_attributes function."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
