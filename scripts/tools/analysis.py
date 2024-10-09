from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tools import slide_viz
from tqdm import tqdm

# from matplotlib.gridspec import GridSpec


def predict_slide(model, image_dict, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device)

    model.eval()
    model = model.to(device)
    predictions = {}

    dataloader = torch.utils.data.DataLoader(list(image_dict.items()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on cells", unit="batch"):
            cell_ids, images = batch
            images = images.to(device).float() / 255.0

            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)

            # Store the predicted class and the full probability vector for each cell
            for cell_id, pred_class, prob_vector in zip(cell_ids, predicted_classes, outputs):
                predictions[cell_id] = {
                    "predicted_class": pred_class.item(),
                    "probabilities": prob_vector.cpu().tolist(),  # Convert tensor to a list for easier handling
                }

    return predictions


def extract_stats(predictions, ct_list, metric="predicted"):
    stats = {}

    if metric == "predicted":
        for _, prediction in predictions.items():
            predicted_class = prediction["predicted_class"]
            max_prob = max(prediction["probabilities"])

            if predicted_class not in stats:
                stats[predicted_class] = {"probs": [], "count": 0}

            stats[predicted_class]["probs"].append(max_prob)
            stats[predicted_class]["count"] += 1

        data = []
        for class_id, class_stats in stats.items():
            class_probs = class_stats["probs"]
            row = [
                class_id,
                ct_list[class_id],
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs),
                class_stats["count"],
            ]
            data.append(row)

        columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob", "Cell Count"]
        df_stats = pd.DataFrame(data, columns=columns)

    elif metric == "all":
        num_classes = len(predictions[next(iter(predictions))]["probabilities"])

        for class_id in range(num_classes):
            stats[class_id] = {"probs": [], "count": 0}

        for _, prediction in predictions.items():
            prob_vector = prediction["probabilities"]
            for class_id, prob in enumerate(prob_vector):
                stats[class_id]["probs"].append(prob)
                stats[class_id]["count"] += 1

        data = []
        for class_id, class_stats in stats.items():
            class_probs = class_stats["probs"]
            row = [
                class_id,
                ct_list[class_id],
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs),
            ]
            data.append(row)

        columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob"]
        df_stats = pd.DataFrame(data, columns=columns)

    else:
        raise ValueError("Invalid metric. Choose 'predicted' or 'all'.")

    df_stats = df_stats.sort_values(by="Class", ascending=True).reset_index(drop=True)
    df_stats = df_stats.set_index("Class")

    return df_stats


def generate_dicts_viz_pred(nuc_dict, class_dict, ct_list):
    """
    Updates the nuclear dictionary with the predicted classes.
    """
    for i, (key, _) in enumerate(nuc_dict["nuc"].items()):
        if str(i) in class_dict:
            nuc_dict["nuc"][key]["type"] = class_dict[str(i)]["predicted_class"]

    return nuc_dict, generate_color_dict(ct_list)


def generate_color_dict(list, palette="hsv"):
    """
    Generate a dictionary of colors for each class in the list.
    """
    color_dict = {}
    num_classes = len(list)

    cmap = plt.get_cmap(palette, num_classes)

    for i, class_name in enumerate(list):
        color = cmap(i)[:3]
        color = [int(255 * c) for c in color]
        color_dict[str(i)] = [class_name, color]

    return color_dict


def plot_history(history_train, history_val, show=False, savefig=None) -> None:
    """
    Plot the training and validation loss history.

    Parameters:
    - history_train (list): A list of training loss values.
    - history_val (list): A list of validation loss values.
    - criterion (str): The criterion used for calculating the loss.
    - show (bool, optional): Whether to display the plot. Defaults to False.
    - savefig (str, optional): The filename to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history_train) + 1), history_train, color="blue")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history_val) + 1), history_val, color="blue")
    plt.title("Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_predicted_cell_labels_in_spot(spot_id, model, image_path, adata, adata_name, dict_cells, cell_images):
    """
    Plot a spot's visualization along with all cell images arranged in a grid, showing predicted and real labels.

    Args:
    - spot_id: ID of the spot to visualize.
    - model: Trained model to predict cell types.
    - image_path: Path to the whole-slide image.
    - adata: Anndata object with cell metadata.
    - adata_name: Name of the adata dataset.
    - dict_cells: Dictionary containing cell information (including real labels).
    - dict_types_colors: Dictionary mapping cell types to colors.
    - cell_images: Dictionary or list of cell images for the spot.
    """

    if isinstance(dict_cells, str) and os.path.isfile(dict_cells):
        with open(dict_cells) as json_file:
            data = json.load(json_file)
    elif isinstance(dict_cells, dict) or dict_cells is None:
        data = dict_cells
    else:
        raise ValueError("dict_cells must be a path to a JSON file or a dictionary.")

    # Set up the plot layout
    fig = plt.figure(figsize=(18, 10))
    # gs = GridSpec(1, 2, width_ratios=[1, 2])  # Two columns: 1 for cells, 2 for spot visualization

    # Step 2: Visualize the spot on the right
    # ax1 = fig.add_subplot(gs[1])
    slide_viz.plot_specific_spot(image_path=image_path, adata=adata, adata_name=adata_name, spot_id=spot_id)

    # Step 3: Prepare the grid for cell images on the left
    # ax2 = fig.add_subplot(gs[0])

    # Get the real labels for cells in this spot
    spot_cells = data["nuc"]  # Assuming 'nuc' contains cell info

    # Step 4: Generate predictions for each cell in the spot
    predicted_labels = []
    for cell_id, cell_data in spot_cells.items():
        image = cell_images[cell_id]
        image_tensor = _preprocess_image(image).unsqueeze(0).to(model.device)  # Preprocess and convert to tensor
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output).item()
            predicted_labels.append(predicted_class)

    # Step 5: Create a grid of cell images with labels
    num_cells = len(spot_cells)
    grid_size = int(np.ceil(np.sqrt(num_cells)))  # Square grid
    for i, (cell_id, cell_data) in enumerate(spot_cells.items()):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)

        image = cell_images[cell_id]
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        real_label = cell_data["type"]  # Real label from HoverNet
        predicted_label = predicted_labels[i]

        # Check if real and predicted labels differ
        if real_label == predicted_label:
            label_text = f"Pred: {predicted_label}"
            ax.set_title(label_text, color="black", fontsize=10)
        else:
            label_text = f"Pred: {predicted_label} ({real_label})"
            ax.set_title(label_text, color="red", fontsize=10)

    plt.tight_layout()
    plt.show()


def _preprocess_image(image):
    """
    Preprocess an image for model prediction (resize, normalize, etc.).
    Modify this function based on your model's input requirements.
    """
    # Example: convert image to tensor, normalize, etc.
    # image = cv2.resize(image, (224, 224))  # Example: resizing to 224x224
    image_tensor = torch.from_numpy(image).float() / 255.0  # Normalize to [0,1]
    image_tensor = image_tensor.permute(2, 0, 1)  # Convert HWC to CHW format
    return image_tensor
