from __future__ import annotations

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# from torchvision.transforms import InterpolationMode


class ImageDataset(Dataset):
    """Dataset for loading images from a pre-saved image_dict.pt"""

    def __init__(self, image_dict):
        self.image_dict = image_dict
        self.cell_ids = list(image_dict.keys())
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        # self.transform = transforms.Compose([
        #     transforms.Resize(size=[232], interpolation=InterpolationMode.BILINEAR),  # Resize to 232
        #     transforms.CenterCrop(size=[224]),  # Center crop to 224
        #     transforms.Lambda(lambda x: x.float() / 255.0),  # Rescale to [0.0, 1.0]
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        # ])

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id]

        image = image.float() / 255.0
        image = self.transform(image)

        return image, cell_id


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
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

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
        images = torch.stack([self.transform(self.image_dict[cell_id].float() / 255.0) for cell_id in cell_ids])
        proportions = torch.tensor(self.spot_prop_df.loc[spot_id].values, dtype=torch.float32)

        return images, proportions
