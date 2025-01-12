from __future__ import annotations

from typing import Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SpotDataset(Dataset):
    """
    A PyTorch Dataset class for managing cell images and their corresponding spot-level proportions.

    Args:
        spot_dict (dict[str, list[str]]): Dictionary containing {spot_id: list of cell IDs}.
        spot_prop_df (pd.DataFrame): DataFrame where each row corresponds to a spot and columns represent
                                     cell type proportions.
        image_dict (dict[str, torch.Tensor]): Dictionary containing {cell_id: image tensor}.
    """

    def __init__(
        self, spot_dict: dict[str, list[str]], spot_prop_df: pd.DataFrame, image_dict: dict[str, torch.Tensor]
    ) -> None:

        self.spot_dict = spot_dict
        self.spot_prop_df = spot_prop_df
        self.image_dict = image_dict
        self.spot_ids = list(spot_dict.keys())

    def __len__(self) -> int:
        """
        Returns the total number of spots in the dataset.

        Returns:
            int: The number of spots.
        """

        return len(self.spot_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the image tensors and cell type proportions for a given spot.

        Args:
            idx (int): Index of the spot in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Tensor of stacked image tensors for the cells in the spot.
                - Tensor of cell type proportions for the spot.
        """

        spot_id = self.spot_ids[idx]
        cell_ids = self.spot_dict[spot_id]
        images = torch.stack([self.image_dict[cell_id].float() / 255.0 for cell_id in cell_ids])
        proportions = torch.tensor(self.spot_prop_df.loc[spot_id].values, dtype=torch.float32)

        return images, proportions


def split_data(
    spot_dict: dict[str, list[str]],
    spot_prop_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    rs: int = 42,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """
    Splits data into training, validation, and testing sets.

    Args:
        spot_dict (dict[str, list[str]]): Dictionary containing {spot_id: list of cell IDs}.
        spot_prop_df (pd.DataFrame): DataFrame where each row corresponds to a spot and columns
                                     represent cell type proportions.
        train_size (float): Proportion of the dataset to use for training. Defaults to 0.7.
        val_size (float): Proportion of the dataset to use for validation. Defaults to 0.15.
        rs (int): Random state for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - Training set dictionary and corresponding proportions (dict, pd.DataFrame).
            - Validation set dictionary and corresponding proportions (dict, pd.DataFrame).
            - Testing set dictionary and corresponding proportions (dict, pd.DataFrame).
    """

    assert train_size + val_size <= 1, "Train size + validation size must not exceed 1."

    spot_ids = list(spot_dict.keys())

    train_ids, temp_ids = train_test_split(spot_ids, train_size=train_size, random_state=rs)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size / (1 - train_size), random_state=rs)

    train_spot_dict = {spot: spot_dict[spot] for spot in train_ids}
    val_spot_dict = {spot: spot_dict[spot] for spot in val_ids}
    test_spot_dict = {spot: spot_dict[spot] for spot in test_ids}

    train_proportions = spot_prop_df.loc[train_ids]
    val_proportions = spot_prop_df.loc[val_ids]
    test_proportions = spot_prop_df.loc[test_ids]

    return train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions


def pp_prop(spot_prop: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Preprocesses spot proportions by normalizing each row to sum to 1.

    Args:
        spot_prop (Union[pd.DataFrame, str]): A DataFrame where each row corresponds to a spot and columns represent
                                        cell type proportions. If a string is provided, it is treated as a file path,
                                        and the DataFrame is read from the file.

    Returns:
        pd.DataFrame: A normalized DataFrame where each row sums to 1.
    """

    if isinstance(spot_prop, str):
        spot_prop = pd.read_csv(spot_prop, index_col=0)

    row_sums = spot_prop.sum(axis=1)
    spot_prop = spot_prop.div(row_sums, axis=0)

    return spot_prop


# def get_visium_infos(adata, adata_name):

#     centers = adata.obsm["spatial"].astype("int64")
#     diameter = adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"]

#     mpp = 55 / diameter
#     mag = get_mag(mpp)

#     return mag, mpp, centers, diameter


# def get_xenium_infos():
#     mpp = 0.2125
#     # from https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
#     mag = get_mag(mpp)
#     return mag, mpp

# def get_mag(mpp):
#     """Returns the magnification of the image based on the mpp.

#     from HEST
#     """

#     if mpp <= 0.1:
#         mag = 60
#     elif 0.1 < mpp and mpp <= 0.25:
#         mag = 40
#     elif 0.25 < mpp and mpp <= 0.5:
#         mag = 40
#     elif 0.5 < mpp and mpp <= 1:
#         mag = 20
#     elif 1 < mpp and mpp <= 4:
#         mag = 10
#     elif 4 < mpp:
#         mag = 5  # not sure about that one

#     return mag
