from __future__ import annotations

import os
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from external.mocov3.image_encoder import InstanceEmbedder
from external.mocov3.utils import load_eval_transform


class ImageDictDataset(Dataset):
    """Dataset for loading images from a pre-saved image_dict.pt"""

    def __init__(self, image_dict, transform):
        self.image_dict = image_dict
        self.transform = transform
        self.cell_ids = list(image_dict.keys())  # Store all cell IDs

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id]  # Tensor (3, 64, 64)

        image = image.float() / 255.0
        image = self.transform(image)

        return image, cell_id


def infer_embed(
    path_image: str,
    save_folder: str,
    tag: str,
    model_name: str = "resnet50",
    batch_size: int = 2048,
    num_workers: int = 4,
) -> None:
    """
    Runs inference to extract embeddings for all images in path_image using a pre-trained MoCo model.

    Args:
        path_image: Path to the .pt file containing the image dictionary.
        save_folder: Folder to save the embeddings.
        tag: Tag identifying the pre-trained model to use.
        model_name: Model architecture name. Defaults to "resnet50".
        batch_size: Batch size for inference. Defaults to 2048.
        num_workers: Number of workers for data loading. Defaults to 4.
    """

    weight_dir = "models/ssl"
    weights_path = os.path.join(weight_dir, tag, "moco_model_best.pth.tar")

    stats_path = os.path.join(Path(weights_path).parent, "moco_model_best_mean_std.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = InstanceEmbedder(model_name, weights_path).to(device)
    encoder.eval()
    encoder.to(device)

    image_dict = torch.load(path_image)
    cell_dataset = ImageDictDataset(image_dict, load_eval_transform(stats_path))
    cell_dataloader = torch.utils.data.DataLoader(
        cell_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    encoder.eval()

    embeddings_dict = {}

    for images, ids in tqdm(cell_dataloader, total=len(cell_dataloader)):
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float32):
            images = images.to(device)
            embeddings = encoder(images)
            embeddings = embeddings.view(embeddings.shape[0], -1)

            for i, cell_id in enumerate(ids):
                embeddings_dict[cell_id] = embeddings[i].cpu()

    embedding_path = os.path.join(save_folder, f"moco_embed_{tag}.pt")
    torch.save(embeddings_dict, embedding_path)
