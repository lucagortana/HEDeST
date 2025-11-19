from __future__ import annotations

from torch.utils.data import Dataset


class ImageDictDataset(Dataset):
    """Dataset for loading images from a pre-saved image_dict.pt"""

    def __init__(self, image_dict, transform):
        self.image_dict = image_dict
        self.cell_ids = list(image_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id]

        image = image.float() / 255.0
        image = self.transform(image)

        return image, cell_id
