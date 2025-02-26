from __future__ import annotations

import io
import random
from datetime import timedelta
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from deconvplugin.model.cell_classifier import CellClassifier


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


def load_model(model_path: str, model_name: str, num_classes: int, hidden_dims: List[int]) -> CellClassifier:
    """
    Load a trained model from a file.

    Parameters:
        model_path (str): Path to the model file.
        edge_size (int): Size of the image edge.
        num_classes (int): Number of classes in the model.
        mtype (str): Model type.

    Returns:
        CellClassifier: The loaded model.

    Example:
        >>> model = load_model("model.pth", 128, 10, "resnet")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device found to load the model : ", device)

    model = CellClassifier(model_name=model_name, num_classes=num_classes, hidden_dims=hidden_dims, device=device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    if device == torch.device("cuda"):
        model = model.to(device)

    return model


def fig_to_array(fig: Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a NumPy array.

    Parameters:
        fig (Figure): A Matplotlib figure.

    Returns:
        np.ndarray: An image array.
    """

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def check_json_classification(data: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> bool:
    """
    Check if all cells in the JSON classification data have a non-None type.

    Parameters:
        data (Dict): A nested dictionary containing classification data.

    Returns:
        bool: True if all cells have types, False otherwise.
    """

    first_key = next(iter(data["nuc"]))
    return data["nuc"][first_key]["type"] is not None


def seg_colors_compatible(
    seg_dict: Dict[str, Dict[str, Dict[str, Union[str, int]]]], color_dict: Dict[str, Tuple[str, Tuple[int, int, int]]]
) -> bool:
    """
    Check if segmentation labels are compatible with color dictionary.

    Parameters:
        seg_dict (Dict): Segmentation data dictionary.
        color_dict (Dict): Color dictionary.

    Returns:
        bool: True if all segmentation labels have corresponding colors, False otherwise.
    """

    seg_labels = set(str(cell["type"]) for cell_data in seg_dict["nuc"].values() for cell in [cell_data])
    color_labels = set(color_dict.keys())

    return (seg_labels - color_labels) == set()


def format_time(seconds: int) -> str:
    """
    Format time duration in HH:MM:SS or MM:SS format.

    Parameters:
        seconds (int): Time duration in seconds.

    Returns:
        str: Formatted time string.
    """

    formatted_time = str(timedelta(seconds=int(seconds)))
    if seconds < 3600:
        formatted_time = formatted_time[2:]
    return formatted_time


def revert_dict(data: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Revert a dictionary with lists as values to a dictionary mapping values to keys.

    Parameters:
        data (Dict): Input dictionary.

    Returns:
        Dict: Reverted dictionary.
    """

    return {val: key for key, values in data.items() for val in values}


def remove_empty_keys(data: Dict[str, List]) -> Dict[str, List]:
    """
    Remove keys with empty lists from a dictionary.

    Parameters:
        data (Dict): Input dictionary.

    Returns:
        Dict: Dictionary without empty keys.
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
    Generate a dictionary of colors for labels.

    Parameters:
        labels (List): List of class labels.
        palette (str): Matplotlib color palette.
        format (str): Output format - "classic" or "special".
        n_max (int): Maximum number of unique colors.

    Returns:
        Dict: Color dictionary.
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

    Parameters:
        *required_attributes (str): Names of required attributes.

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
