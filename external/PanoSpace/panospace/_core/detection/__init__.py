"""panospace._core.detection
===========================
Back-end implementations for nuclei / cell detection.  Each concrete detector
should live in its own module - e.g. :pymod:`cellvit`, :pymod:`stardist` - and
*register* itself via :pyfunc:`panospace._core.register` under the "detection"
category::

    >>> from panospace._core import register
    >>> def detect_cells_core(sdata: "SpatialData", **kwargs):
    ...     ...
    >>> register("detection", "cellvit", detect_cells_core)

This ``__init__`` lazily imports known back-end modules so that merely importing
:pymod:`panospace.tl` does *not* pull heavy DL frameworks unless the user
explicitly requests that detector.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Attempt to import built-in back-ends. They will *register* themselves during
# import; failures are tolerated and only logged at DEBUG level so that the rest
# of PanoSpace remains usable without optional heavy dependencies (e.g. PyTorch
# for CellViT).
# -----------------------------------------------------------------------------

_BUILTIN: Dict[str, str] = {
    "cellvit": ".cellvit",  # Vision-Transformer-based nuclei segmentation
    "stardist": ".stardist",  # StarDist CNN nuclei detector
}

for _name, _rel_mod in _BUILTIN.items():
    try:
        importlib.import_module(__name__ + _rel_mod)
    except ModuleNotFoundError as e:
        logger.debug("Detection backend '%s' could not be imported: %s", _name, e)

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

BackendFunc = Callable[..., Any]
# The first argument must be a SpatialData object; additional backend-specific
# parameters are passed via *args / **kwargs and typed as ``Any`` here to avoid
# over-constraining plugin authors.
