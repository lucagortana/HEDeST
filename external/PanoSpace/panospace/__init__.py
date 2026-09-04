from __future__ import annotations

import logging
from importlib import metadata
from typing import TYPE_CHECKING

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
try:
    __version__: str = metadata.version("panospace")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
# Only configure logging if the user hasn't already set it up
# This allows users to customize logging while providing sensible defaults
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
# -----------------------------------------------------------------------------


__all__ = [
    "detect_cells",
    "deconv_celltype",
    "superres_celltype",
    "celltype_annotator",
    # Utility functions for system information
    "list_available_backends",
    "get_backend_error",
]

# ``panospace.bench`` is a real submodule (HEDeST-benchmark I/O adapters) and is
# imported the normal way -- it must NOT go through __getattr__ below, which
# would recurse through importlib's fromlist handling.


if TYPE_CHECKING:
    from .tl.detect import detect_cells
    from .tl.annotate import deconv_celltype, superres_celltype, celltype_annotator
    from . import bench


def __getattr__(name: str):
    if name == "detect_cells":
        from .tl.detect import detect_cells

        return detect_cells
    if name in {"deconv_celltype", "superres_celltype", "celltype_annotator"}:
        from .tl.annotate import deconv_celltype, superres_celltype, celltype_annotator

        return {
            "deconv_celltype": deconv_celltype,
            "superres_celltype": superres_celltype,
            "celltype_annotator": celltype_annotator,
        }[name]
    if name == "list_available_backends":
        from .tl import list_available_backends

        return list_available_backends
    if name == "get_backend_error":
        from .tl import get_backend_error

        return get_backend_error
    raise AttributeError(f"module {__name__!r} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
