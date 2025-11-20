from __future__ import annotations

from typing import Dict
from typing import List

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from hedest.config import TqdmToLogger
from hedest.dataset import EmbedDataset
from hedest.dataset import ImageDataset
from hedest.dataset_utils import get_transform
from hedest.model.cell_classifier import CellClassifier

tqdm_out = TqdmToLogger(logger, level="INFO")


def predict_slide(
    model: CellClassifier,
    image_dict: Dict[str, torch.Tensor],
    ct_list: List[str],
    batch_size: int = 1024,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predicts the cell type probabilities for all cells in a slide.

    Args:
        model (nn.Module): The trained model to use for predictions.
        image_dict (Dict[str, torch.Tensor]): A dictionary where keys are cell IDs and values are image tensors.
        ct_list (List[str]): List of cell type names.
        batch_size (int): Batch size for prediction.
        verbose (bool): Whether to display progress.

    Returns:
        pd.DataFrame: A DataFrame where rows correspond to cell IDs and columns to cell type probabilities.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info("Device used : ", device)

    model.eval()
    model = model.to(device)
    cell_prob = []

    if model.model_name == "default":
        dataset = EmbedDataset(image_dict)
    else:
        transform = get_transform(model.model_name)
        dataset = ImageDataset(image_dict, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, cell_ids in tqdm(
            dataloader, desc="Predicting on cells", unit="batch", file=tqdm_out, disable=(not verbose)
        ):
            images = images.to(device)
            outputs = model(images)

            for cell_id, prob_vector in zip(cell_ids, outputs):
                cell_prob.append(
                    {
                        "cell_id": cell_id,
                        **{ct_list[i]: prob for i, prob in enumerate(prob_vector.cpu().tolist())},
                    }
                )

    cell_prob_df = pd.DataFrame(cell_prob)
    cell_prob_df.set_index("cell_id", inplace=True)

    return cell_prob_df
