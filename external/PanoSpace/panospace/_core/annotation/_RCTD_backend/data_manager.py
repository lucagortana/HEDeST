"""
Data downloader for RCTD backend.

Downloads large data files from GitHub Releases on first use.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tarfile
from pathlib import Path

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# GitHub Release information
GITHUB_REPO = "hehuifeng/PanoSpace-core"
DATA_VERSION = "v0.1.0"
DATA_ARCHIVE = "rctd_data_v0.1.0.tar.gz"

# Expected file sizes and MD5 checksums
FILE_INFO = {
    "Q_mat_1_1.txt.gz": {"size": 64 * 1024 * 1024, "md5": None},  # 64MB
    "Q_mat_1_2.txt.gz": {"size": 70 * 1024 * 1024, "md5": None},  # 70MB
    "Q_mat_2_1.txt.gz": {"size": 63 * 1024 * 1024, "md5": None},  # 63MB
    "Q_mat_2_2.txt.gz": {"size": 69 * 1024 * 1024, "md5": None},  # 69MB
    "X_vals.txt": {"size": 43 * 1024, "md5": None},  # 43KB
    "ligand_receptors.txt": {"size": 12 * 1024, "md5": None},  # 12KB
}

# Base URL for GitHub Releases
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{DATA_VERSION}"
ARCHIVE_URL = f"{BASE_URL}/{DATA_ARCHIVE}"


def get_data_dir():
    """Get the directory where data files should be stored."""
    # Use package directory
    return Path(os.path.dirname(os.path.realpath(__file__))) / "extdata"


def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url, filepath, expected_size=None):
    """
    Download a file with progress bar.

    Parameters
    ----------
    url : str
        URL to download from
    filepath : Path
        Where to save the file
    expected_size : int, optional
        Expected file size in bytes
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "The 'requests' library is required to download RCTD data files. "
            "Please install it with: pip install requests"
        )

    logger.info(f"Downloading {filepath.name} from {url}")

    # Stream download with progress
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    total_size = expected_size or int(response.headers.get("content-length", 0))
    downloaded = 0
    import time

    start_time = time.time()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Log progress every 10MB
                if downloaded % (10 * 1024 * 1024) == 0 or downloaded == total_size:
                    if total_size > 0:
                        progress = 100 * downloaded / total_size
                        elapsed = time.time() - start_time
                        speed = (downloaded / 1024 / 1024) / elapsed if elapsed > 0 else 0
                        remaining = (total_size - downloaded) / speed / 1024 / 1024 if speed > 0 else 0
                        logger.info(
                            f"  Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f}MB) "
                            f"Speed: {speed:.2f}MB/s ETA: {remaining:.1f}s"
                        )

    logger.info(f"Downloaded {filepath.name} successfully")


def verify_file(filepath, expected_size=None):
    """
    Verify a downloaded file.

    Parameters
    ----------
    filepath : Path
        Path to the file
    expected_size : int, optional
        Expected file size in bytes

    Returns
    -------
    bool
        True if file is valid
    """
    if not filepath.exists():
        return False

    actual_size = filepath.stat().st_size

    # Check size if provided
    if expected_size is not None:
        # Allow 1% tolerance
        size_diff = abs(actual_size - expected_size) / expected_size
        if size_diff > 0.01:
            logger.warning(
                f"File {filepath.name} has unexpected size: "
                f"{actual_size / 1024 / 1024:.1f}MB vs expected {expected_size / 1024 / 1024:.1f}MB"
            )
            return False

    return True


def ensure_data_files():
    """
    Ensure all required data files are present.
    Downloads and extracts archive from GitHub Releases if needed.

    Returns
    -------
    Path
        Path to the data directory
    """
    data_dir = get_data_dir()

    # Check if all files exist
    missing_files = []
    for filename in FILE_INFO.keys():
        filepath = data_dir / filename
        if not verify_file(filepath, FILE_INFO[filename]["size"]):
            missing_files.append(filename)

    if not missing_files:
        logger.info("All RCTD data files are present and verified.")
        return data_dir

    # Download and extract archive
    logger.info(f"Found {len(missing_files)} missing RCTD data files. Downloading archive...")
    logger.info(f"Download URL: {ARCHIVE_URL}")
    logger.warning("Note: Download may take 20-30 minutes depending on network speed.")
    logger.warning("If download is too slow, press Ctrl+C to cancel and download manually:")
    logger.warning(f"  URL: {ARCHIVE_URL}")
    logger.warning(f"  Extract to: {data_dir}")

    # Download archive to temp location
    temp_archive = Path("/tmp") / DATA_ARCHIVE

    try:
        download_file(ARCHIVE_URL, temp_archive, 265 * 1024 * 1024)  # 265MB

        # Extract archive
        logger.info(f"Extracting {DATA_ARCHIVE}...")
        data_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(temp_archive, "r:gz") as tar:
            tar.extractall(data_dir)

        # Clean up archive
        temp_archive.unlink()

        # Verify all files
        for filename in FILE_INFO.keys():
            filepath = data_dir / filename
            if not verify_file(filepath, FILE_INFO[filename]["size"]):
                raise ValueError(f"Extracted file {filename} failed verification")

        logger.info("All RCTD data files downloaded and extracted successfully.")
        return data_dir

    except Exception as e:
        logger.error(f"Failed to download/extract data archive: {e}")
        logger.error("\n" + "=" * 80)
        logger.error("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.error("=" * 80)
        logger.error(f"1. Download the data archive manually from:")
        logger.error(f"   {ARCHIVE_URL}")
        logger.error(f"\n2. Extract the archive to the data directory:")
        logger.error(f"   mkdir -p {data_dir}")
        logger.error(f"   tar -xzf {DATA_ARCHIVE} -C {data_dir}")
        logger.error(f"\n3. Verify the following files are present:")
        for filename in FILE_INFO.keys():
            logger.error(f"   - {filename}")
        logger.error("=" * 80)
        # Clean up partial download
        if temp_archive.exists():
            temp_archive.unlink()
        raise


def get_data_file_path(filename):
    """
    Get the path to a data file, downloading if necessary.

    Parameters
    ----------
    filename : str
        Name of the data file

    Returns
    -------
    Path
        Path to the data file
    """
    if filename not in FILE_INFO:
        raise ValueError(f"Unknown data file: {filename}")

    data_dir = ensure_data_files()
    return data_dir / filename
