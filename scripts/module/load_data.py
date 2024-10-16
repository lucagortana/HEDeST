from __future__ import annotations

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SpotDataset(Dataset):
    def __init__(self, spot_dict, proportion_df, image_dict):
        """
        Args:
            spot_dict (dict): Dictionary containing {spot_id: list of cell ids}.
            proportion_df (pd.DataFrame): DataFrame where each row corresponds to a spot and columns represent cell
                                          type proportions.
            image_dict (dict): Dictionary containing {cell_id: image tensor}.
        """
        self.spot_dict = spot_dict
        self.proportion_df = proportion_df
        self.image_dict = image_dict
        self.spot_ids = list(spot_dict.keys())

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]
        cell_ids = self.spot_dict[spot_id]
        images = torch.stack(
            [self.image_dict[cell_id].float() / 255.0 for cell_id in cell_ids]
        )  # change to float and normalize
        proportions = torch.tensor(self.proportion_df.loc[spot_id].values, dtype=torch.float32)

        return images, proportions


def split_data(spot_dict, proportions, train_size=0.7, val_size=0.15, rs=42):

    assert train_size + val_size <= 1, "Train size + validation size must not exceed 1."

    spot_ids = list(spot_dict.keys())

    # Split into train, validation, and test sets
    train_ids, temp_ids = train_test_split(spot_ids, train_size=train_size, random_state=rs)
    val_ids, test_ids = train_test_split(temp_ids, test_size=val_size / (1 - train_size), random_state=rs)

    # Create dictionaries for each set
    train_spot_dict = {spot: spot_dict[spot] for spot in train_ids}
    val_spot_dict = {spot: spot_dict[spot] for spot in val_ids}
    test_spot_dict = {spot: spot_dict[spot] for spot in test_ids}

    # Subset proportions
    train_proportions = proportions.loc[train_ids]
    val_proportions = proportions.loc[val_ids]
    test_proportions = proportions.loc[test_ids]

    return train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions


def pp_prop(proportions):
    """
    Preprocess proportions by normalizing each row to sum to 1.

    Args:
        proportions (pd.DataFrame or str): DataFrame where each row corresponds to a spot and columns represent cell
                                            type proportions. If str, read DataFrame from file.

    Returns:
        pd.DataFrame: Normalized proportions.
    """

    if isinstance(proportions, str):
        proportions = pd.read_csv(proportions, index_col=0)

    row_sums = proportions.sum(axis=1)
    proportions = proportions.div(row_sums, axis=0)

    return proportions
