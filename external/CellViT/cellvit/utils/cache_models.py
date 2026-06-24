# -*- coding: utf-8 -*-
# Cache CellViT models and classifier
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

from cellvit.config.config import CACHE_DIR
from cellvit.utils.download import check_and_download
from cellvit.utils.logger import PrintLogger


def cache_cellvit_sam_h(logger: Optional[logging.Logger] = None) -> Path:
    """Download and cache the CellViT-SAM-H-x40-AMP.pth model file, if not already cached.

    Args:
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Returns:
        Path: Path to the cached model file.
    """
    logger = logger or PrintLogger()
    check_and_download(
        directory_path=CACHE_DIR,
        file_name="CellViT-SAM-H-x40-AMP.pth",
        download_link="https://zenodo.org/records/15094831/files/CellViT-SAM-H-x40-AMP.pth",
        logger=logger,
    )
    return Path(CACHE_DIR) / "CellViT-SAM-H-x40-AMP.pth"


def cache_cellvit_256(logger: Optional[logging.Logger] = None) -> Path:
    """Download and cache the CellViT-256-x40-AMP.pth model file, if not already cached.

    Args:
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Returns:
        Path: Path to the cached model file.
    """
    logger = logger or PrintLogger()
    check_and_download(
        directory_path=CACHE_DIR,
        file_name="CellViT-256-x40-AMP.pth",
        download_link="https://zenodo.org/records/15094831/files/CellViT-256-x40-AMP.pth",
        logger=logger,
    )
    return Path(CACHE_DIR) / "CellViT-256-x40-AMP.pth"


def cache_classifier(logger: Optional[logging.Logger] = None) -> Path:
    """Download and cache the classifier.zip file, if not already cached.

    Args:
        logger (Optional[logging.Logger], optional): Logger. Defaults to None.

    Returns:
        Path: Path to the cached classifier directory.
    """
    logger = logger or PrintLogger()
    classifier_dir = Path(CACHE_DIR) / "classifier"
    zip_dir = Path(CACHE_DIR) / "classifier.zip"
    if not classifier_dir.exists():
        check_and_download(
            directory_path=CACHE_DIR,
            file_name="classifier.zip",
            download_link="https://zenodo.org/records/15094831/files/classifier.zip",
            logger=logger,
        )
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(CACHE_DIR)
        os.remove(zip_dir)
    else:
        if zip_dir.exists():
            os.remove(zip_dir)
    return classifier_dir


def main():
    """Main function to cache models and classifier."""
    logger = PrintLogger()
    cache_cellvit_sam_h(logger)
    cache_cellvit_256(logger)
    cache_classifier(logger)


if __name__ == "__main__":
    main()
