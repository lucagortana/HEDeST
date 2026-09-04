# -*- coding: utf-8 -*-
# Cache and manage CellViT pretrained models and classifier
#
# PanoSpace-adapted version
# @ PanoSpace Team
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

from .config import CACHE_DIR
from .download import check_and_download
from .download import PrintLogger


def cache_cellvit_sam(version: str = "H", logger: Optional[logging.Logger] = None) -> Path:
    """
    Download and cache CellViT-SAM model (B, L, or H).

    Args:
        version (str): One of ["B", "L", "H"], indicating SAM backbone version.
        logger (Optional[logging.Logger]): Optional logger.

    Returns:
        Path: Local path to the cached SAM model.
    """
    assert version in ["B", "L", "H"], f"Unknown SAM version: {version}"
    logger = logger or PrintLogger()
    model_name = f"CellViT-SAM-{version}-x40-AMP.pth"
    url = f"https://zenodo.org/records/15094831/files/{model_name}"
    check_and_download(
        directory_path=CACHE_DIR,
        file_name=model_name,
        download_link=url,
        logger=logger,
    )
    return Path(CACHE_DIR) / model_name


def cache_cellvit_256(logger: Optional[logging.Logger] = None) -> Path:
    """
    Download and cache CellViT-256 model.

    Args:
        logger (Optional[logging.Logger]): Optional logger.

    Returns:
        Path: Local path to the cached CellViT-256 model.
    """
    logger = logger or PrintLogger()
    model_name = "CellViT-256-x40-AMP.pth"
    url = f"https://zenodo.org/records/15094831/files/{model_name}"
    check_and_download(
        directory_path=CACHE_DIR,
        file_name=model_name,
        download_link=url,
        logger=logger,
    )
    return Path(CACHE_DIR) / model_name


def cache_classifier(logger: Optional[logging.Logger] = None) -> Path:
    """
    Download and cache the CellViT classifier weights (zip) and extract.

    Args:
        logger (Optional[logging.Logger]): Optional logger.

    Returns:
        Path: Path to the extracted classifier directory.
    """
    logger = logger or PrintLogger()
    classifier_dir = Path(CACHE_DIR) / "classifier"
    classifier_zip = Path(CACHE_DIR) / "classifier.zip"

    if not classifier_dir.exists():
        check_and_download(
            directory_path=CACHE_DIR,
            file_name="classifier.zip",
            download_link="https://zenodo.org/records/15094831/files/classifier.zip",
            logger=logger,
        )
        with zipfile.ZipFile(classifier_zip, "r") as zip_ref:
            zip_ref.extractall(CACHE_DIR)
        os.remove(classifier_zip)
    else:
        if classifier_zip.exists():
            os.remove(classifier_zip)

    return classifier_dir
