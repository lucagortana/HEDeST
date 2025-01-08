from __future__ import annotations

import io
import random
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from deconvplugin.modeling.cell_classifier import CellClassifier


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


def load_model(model_path: str, size_edge: int, num_classes: int, mtype: str) -> CellClassifier:
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

    model = CellClassifier(size_edge=size_edge, num_classes=num_classes, mtype=mtype, device=device)

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


def seg_colors_compatible(seg_dict, color_dict):
    seg_labels = set(str(cell["type"]) for cell_data in seg_dict["nuc"].values() for cell in [cell_data])
    color_labels = set(color_dict.keys())

    return (seg_labels - color_labels) == set()


def format_time(seconds):
    """Formats time in HH:MM:SS or MM:SS depending on the duration."""
    formatted_time = str(timedelta(seconds=int(seconds)))
    if seconds < 3600:
        formatted_time = formatted_time[2:]
    return formatted_time


def revert_dict(dict):
    return {val: key for key, values in dict.items() for val in values}


def remove_empty_keys(dict):
    empty_keys = []
    for key, value in dict.items():
        if value == []:
            empty_keys.append(key)

    for element in empty_keys:
        del dict[element]

    return dict


def generate_color_dict(list, palette="tab20", format="pie", n_max=40):
    """
    Generate a dictionary of colors for each class in the list.
    Format can be either "classic" or "special".
    'classic' will generate a dictionary with the class name as key and the color as value.
    'special' will generate a dictionary with the class index as key and the doulbet [class_name, color] as value.
    """

    if len(list) > n_max:
        print("Warning: The number of classes is greater than the maximum number of colors available in the palette.")
        print("The colors will be repeated.")

    num_classes = len(list)
    cmap = plt.get_cmap(palette)

    if format == "classic":
        return {list[i]: cmap(i % n_max) for i in range(num_classes)}

    elif format == "special":
        color_dict = {}
        for i, class_name in enumerate(list):
            color = cmap(i % n_max)
            color = [int(255 * c) for c in color]
            color_dict[str(i)] = [class_name, color]

        return color_dict

    else:
        raise ValueError("Format must be either 'classic' or 'special'.")


def require_attributes(*required_attributes):
    """
    Décorateur pour vérifier que certains attributs de l'instance ne sont pas None.

    Args:
        *required_attributes (str): Liste des noms d'attributs à vérifier.

    Raises:
        ValueError: Si un ou plusieurs attributs sont None.
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
