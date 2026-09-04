"""panospace.tl.detect
====================
User-facing *nuclei/cell detection* wrapper.  It delegates the heavy lifting to
backend implementations in :pymod:`panospace._core.detection` (CellViT,
StarDist, ...).

"""
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

from PIL import Image

Image.MAX_IMAGE_PIXELS = 10000000000

logger = logging.getLogger(__name__)

from .._utils.device_utils import get_device


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def detect_cells(
    img: Optional[Image.Image] = None,
    *,
    seg_dict: Optional[Union[str, Dict[str, Any]]] = None,
    use_morphology: bool = True,
    model: Literal["cellvit", "stardist"] = "cellvit",
    model_name: str = "HIPT",
    device: Optional[Literal["cuda", "cpu"]] = None,
    tile_size: Optional[int] = None,
    overlap: int = 64,
    prefer_gpu: bool = True,
    return_seg_dict: bool = False,
    mpp: Optional[float] = None,
    mag: Optional[float] = None,
    **kwargs: Any,
):
    """Detect nuclei / cells on the high-resolution image, or reuse a segmentation.

    Parameters
    ----------
    img
        High-resolution tissue image as a :class:`PIL.Image.Image`.  Large
        whole-slide images are automatically tiled before inference, so the
        input can be the full slide or an already cropped region.  May be
        ``None`` when ``seg_dict`` is supplied.
    seg_dict
        An **already computed** segmentation, either as a HoVer-Net-format
        dict (``{"mag":..., "mpp":..., "nuc": {cell_id: {...}}}``) or as a path
        to such a JSON file.  When given, no model is run: ``seg_adata`` is
        built straight from it and its ``.obs_names`` are the segmentation cell
        ids, so downstream predictions map back one-to-one.
    use_morphology
        Whether to expose each nucleus' PanNuke class as ``.obs['img_type']``,
        which enables the morphology branch of the annotator (weight ``alpha``).
    model
        Detection backend to use - currently ``"cellvit"`` (default).  Other
        identifiers (e.g. ``"stardist"``) are reserved for future
        implementations.
    model_name
        Name of the model to use within the selected backend. Default is "HIPT".
        Supported models: "HIPT", "SAM" (CellViT variants).
    device
        Device to use for inference ("cuda", "cpu", or None for auto-detection).
        If None, will automatically detect the best available device.
    tile_size
        Size of tiles for processing large images.  **Must equal the CellViT
        model native input size** (256 for both ``"HIPT"`` and ``"SAM"``).  Any
        other value is unsafe: the model resizes each tile to its fixed input
        size internally, creating a coordinate-space mismatch that silently
        produces periodic dead-zone under-detection.  Mismatched values are
        **overridden to 256 with a RuntimeWarning**; set
        ``PANOSPACE_STRICT_TILESIZE=1`` to raise ``ValueError`` instead.
        If ``None``, defaults to 256.
    overlap
        Overlap between tiles in pixels to avoid edge artifacts. Default is 64.
        Recommended to be at least 32 and less than tile_size/4.
    prefer_gpu
        Whether to prefer GPU when auto-detecting device. Ignored if device is explicitly set.
    return_seg_dict
        Also return the segmentation in HoVer-Net format (third tuple element).
        When ``seg_dict`` was supplied it is echoed back unchanged; when the
        model was run, its detections are converted to that format.
    mpp, mag
        Microns-per-pixel and magnification, only recorded in the returned
        HoVer-Net dict when a fresh segmentation is computed.
    **kwargs
        Additional keyword arguments passed to the backend.

    Returns
    -------
    tuple
        ``(seg_adata, contours)``, or ``(seg_adata, contours, seg_dict)`` when
        ``return_seg_dict=True``.

    Examples
    --------
    >>> from PIL import Image
    >>> import panospace as ps
    >>> # Load image from file
    >>> img = Image.open("tissue.tif")
    >>> # Basic usage - tile_size=256 is required for CellViT models
    >>> seg_adata, contours = ps.detect_cells(img, tile_size=256)
    >>> # Force CPU inference
    >>> seg_adata, contours = ps.detect_cells(img, device="cpu", tile_size=256)
    >>> # With smaller overlap for faster processing
    >>> seg_adata, contours = ps.detect_cells(img, tile_size=256, overlap=32)

    Notes
    -----
    CellViT models are trained on 256x256 patches and their backbone positional
    embeddings are structurally locked to that input size.  tile_size **must** be
    256; any other value triggers a coordinate-space mismatch that silently
    under-detects cells in periodic dead zones.  The parameter is preserved for
    future models with different native input sizes.
    """
    # ------------------------------------------------------------------
    # 0a) Reuse an existing segmentation, when one is given
    # ------------------------------------------------------------------
    if seg_dict is not None:
        from ..bench import load_seg_dict, seg_adata_from_hovernet

        if isinstance(seg_dict, str):
            seg_dict = load_seg_dict(seg_dict)
        elif "nuc" not in seg_dict:
            seg_dict = {"mag": None, "mpp": None, "nuc": seg_dict}

        logger.info("Reusing the supplied segmentation (%d cells); skipping detection.", len(seg_dict["nuc"]))
        seg_adata = seg_adata_from_hovernet(seg_dict, use_morphology=use_morphology)
        contours = [cell["contour"] for cell in seg_dict["nuc"].values()]
        if return_seg_dict:
            return seg_adata, contours, seg_dict
        return seg_adata, contours

    if img is None:
        raise ValueError("detect_cells() needs either `img` or `seg_dict`.")

    # ------------------------------------------------------------------
    # 0b) Device detection and parameter setup
    # ------------------------------------------------------------------
    if device is None:
        device = get_device(prefer_gpu=prefer_gpu)

    # Validate device
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Device must be 'cuda' or 'cpu', got {device!r}")

    # Set tile_size (explicit user specification is recommended)
    if tile_size is None:
        tile_size = 256  # Default fallback
        logger.info("No tile_size specified, using default: tile_size=256 " "(CellViT native input size).")

    logger.info("Detecting cells using backend '%s' with device='%s', tile_size=%d", model, device, tile_size)

    # ------------------------------------------------------------------
    # 1) Dispatch to backend implementation (lazy import)
    # ------------------------------------------------------------------
    if model == "cellvit":
        from .._core.detection.cellvit import detect_cells_core as _backend
    # elif model == "stardist":
    #     from panospace._core.detection.stardist import detect_cells as _backend
    else:
        raise ValueError(f"Unknown detection model: {model!r}")

    # Call backend with configurable parameters
    seg_adata, contours, cell_dict_wsi = _backend(
        img, model_name=model_name, device=device, tile_size=tile_size, overlap=overlap, **kwargs
    )

    # Give detected cells stable string ids, matching the HoVer-Net convention
    # ("0", "1", ...), so the two segmentation sources are interchangeable.
    import pandas as _pd

    seg_adata.obs_names = _pd.Index([str(i) for i in range(seg_adata.n_obs)], name="cell_id")
    if not use_morphology and "img_type" in seg_adata.obs:
        del seg_adata.obs["img_type"]

    logger.info("Detected %d cells", len(seg_adata))

    if return_seg_dict:
        from ..bench import hovernet_dict_from_cellvit

        return seg_adata, contours, hovernet_dict_from_cellvit(cell_dict_wsi, mpp=mpp, mag=mag)
    return seg_adata, contours


__all__ = ["detect_cells"]
