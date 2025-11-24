from __future__ import annotations

import logging

import timm
import torch
from dataset import ImageDictDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_image_dict(data_path: str) -> dict:
    """
    Loads the image_dict from a .pt file.

    Args:
        data_path: Path to the .pt file containing the image dictionary.

    Returns:
        A dictionary mapping cell IDs to image tensors.
    """

    logger.info("Loading image dictionary from %s", data_path)
    return torch.load(data_path)


def save_embedding_dict(emb_dict: dict, out_path: str) -> None:
    """
    Saves the extracted embeddings to a .pt file.

    Args:
        emb_dict: A dictionary mapping cell IDs to embedding tensors.
        out_path: Path to save the .pt file.
    """

    logger.info("Saving extracted embeddings to %s", out_path)
    torch.save(emb_dict, out_path)


def get_embedding_model(model_name: str) -> tuple[torch.nn.Module, transforms.Compose]:
    """
    Loads a pretrained model from timm and removes the classification head.

    Args:
        model_name: Name of the model to load.

    Returns:
        A tuple containing the model and the corresponding image transform.
    """

    if "resnet" in model_name:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif model_name == "h-optimus-0":
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
            ]
        )
    logger.info("Loaded model %s", model_name)
    model.eval()
    return model, transform


def extract_embeddings(model_name: str, image_dict_path: str, batch_size: int = 32) -> dict:
    """
    Extracts embeddings from cell images stored in image_dict.pt.

    Args:
        model_name: Name of the model to use for embedding extraction.
        image_dict_path: Path to the .pt file containing the image dictionary.
        batch_size: Batch size for processing images.

    Returns:
        A dictionary mapping cell IDs to embedding tensors.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)
    model, transform = get_embedding_model(model_name)
    model = model.to(device).half()
    image_dict = load_image_dict(image_dict_path)

    dataset = ImageDictDataset(image_dict, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_dict = {}

    with torch.no_grad():
        for images, cell_ids in tqdm(dataloader):
            images = images.to(device).half()

            with torch.cuda.amp.autocast():
                embeddings = model(images)

            embeddings = embeddings.view(embeddings.shape[0], -1)

            for i, cell_id in enumerate(cell_ids):
                embeddings_dict[cell_id] = embeddings[i].cpu()

    return embeddings
