"""Device management utilities for PanoSpace."""
from __future__ import annotations

import logging
from typing import Literal
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import torch, but make it optional
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


def get_device(prefer_gpu: bool = True) -> Literal["cuda", "cpu"]:
    """
    Automatically detect and return the best available device.

    Parameters
    ----------
    prefer_gpu : bool, default True
        Whether to prefer GPU if available. If False, will use CPU even if GPU is available.

    Returns
    -------
    str
        Device name ("cuda" or "cpu").
    """
    if not _TORCH_AVAILABLE:
        logger.info("PyTorch not available, using CPU")
        return "cpu"

    if not prefer_gpu:
        logger.info("GPU disabled by preference, using CPU")
        return "cpu"

    if not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU")
        return "cpu"

    try:
        # Test if CUDA device actually works
        device = torch.device("cuda")
        test_tensor = torch.tensor([1.0], device=device)
        _ = test_tensor.cpu()  # Test transfer back to CPU
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return "cuda"
    except Exception as e:
        logger.warning(f"CUDA device test failed ({e}), falling back to CPU")
        return "cpu"


def check_memory_requirements(device: str, min_memory_gb: Optional[float] = None) -> bool:
    """
    Check if the device has sufficient memory.

    Parameters
    ----------
    device : str
        Device name ("cuda" or "cpu").
    min_memory_gb : float, optional
        Minimum required memory in GB. If None, performs basic check only.

    Returns
    -------
    bool
        True if device has sufficient memory.
    """
    if device == "cpu":
        return True

    if device == "cuda" and _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if min_memory_gb is None:
                # Basic check - at least 2GB available
                return total_memory >= 2.0
            return total_memory >= min_memory_gb
        except Exception as e:
            logger.warning(f"Could not check CUDA memory: {e}")
            return False

    return False
