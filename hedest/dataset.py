from __future__ import annotations

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Dataset for loading images from a pre-saved image_dict.pt"""

    def __init__(self, image_dict: dict[str, torch.Tensor], transform: transforms.Compose) -> None:
        self.image_dict = image_dict
        self.cell_ids = list(image_dict.keys())

        self.transform = transform

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id].float() / 255.0

        if self.transform is not None:
            image = self.transform(image)

        return image, cell_id


class SpotDataset(Dataset):
    def __init__(self, spot_dict, spot_prop_df, image_dict, transform):
        self.spot_dict = spot_dict
        self.spot_prop_df = spot_prop_df
        self.image_dict = image_dict
        self.spot_ids = list(spot_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]
        cell_ids = self.spot_dict[spot_id]

        if self.transform is not None:
            images = torch.stack([self.transform(self.image_dict[cell_id].float() / 255.0) for cell_id in cell_ids])
        else:
            images = torch.stack([self.image_dict[cell_id].float() / 255.0 for cell_id in cell_ids])
        proportions = torch.tensor(self.spot_prop_df.loc[spot_id].values, dtype=torch.float32)

        bag_indices = torch.full((len(cell_ids),), idx, dtype=torch.long)  # Track spot index per image

        return {"images": images, "proportions": proportions, "bag_indices": bag_indices}


class EmbedDataset(Dataset):
    def __init__(self, image_dict: dict[str, torch.Tensor]) -> None:
        self.image_dict = image_dict
        self.cell_ids = list(image_dict.keys())

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id]

        return image, cell_id


class SpotEmbedDataset(Dataset):
    def __init__(self, spot_dict, spot_prop_df, image_dict):
        self.spot_dict = spot_dict
        self.spot_prop_df = spot_prop_df
        self.image_dict = image_dict
        self.spot_ids = list(spot_dict.keys())

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]
        cell_ids = self.spot_dict[spot_id]
        images = torch.stack([self.image_dict[cell_id] for cell_id in cell_ids])
        proportions = torch.tensor(self.spot_prop_df.loc[spot_id].values, dtype=torch.float32)

        bag_indices = torch.full((len(cell_ids),), idx, dtype=torch.long)  # Track spot index per image

        return {"images": images, "proportions": proportions, "bag_indices": bag_indices}
