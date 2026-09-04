from __future__ import annotations

from collections.abc import Mapping


_BACKENDS: Mapping[str, str] = {
    "RCTD": "panospace._core.annotation.RCTD:annotate_cells_core",
    "cell2location": "panospace._core.annotation.cell2location:annotate_cells_core",
    "spatialDWLS": "panospace._core.annotation.spatialDWLS:annotate_cells_core",
    "endecon": "panospace._core.annotation.endecon:endecon_core",
    "superres_core": "panospace._core.annotation.superres:superres_core",
    "annotator_core": "panospace._core.annotation.annotator:annotator_core",
}

# Optional: Track backend availability without failing imports
_AVAILABLE_BACKENDS: dict[str, bool] = {}
_BACKEND_ERRORS: dict[str, str] = {}


def _check_backend_availability(backend: str, path: str) -> bool:
    """Check if a backend is available without raising errors at import time.

    Parameters
    ----------
    backend : str
        Backend name.
    path : str
        Module path in format "module:function".

    Returns
    -------
    bool
        True if backend is available, False otherwise.
    """
    try:
        module_path, func_name = path.split(":")
        mod = __import__(module_path, fromlist=[func_name])
        getattr(mod, func_name)
        return True
    except (ImportError, AttributeError) as e:
        _BACKEND_ERRORS[backend] = str(e)
        return False


# Check availability of all backends
for backend, path in _BACKENDS.items():
    _AVAILABLE_BACKENDS[backend] = _check_backend_availability(backend, path)


def _import_backend(name: str):
    """Dynamically import a backend function by name.

    Parameters
    ----------
    name : str
        Backend name, must be one of the keys in `_BACKENDS`.

    Returns
    -------
    callable
        The backend function object.

    Raises
    ------
    ValueError
        If the backend name is not registered.
    ImportError
        If the backend is not available due to missing dependencies.
    """
    if name not in _BACKENDS:
        raise ValueError(f"Unknown annotation backend '{name}'. Available: {list(_BACKENDS)}")

    if not _AVAILABLE_BACKENDS.get(name, False):
        error_msg = _BACKEND_ERRORS.get(name, "Unknown error")
        raise ImportError(
            f"Backend '{name}' is not available. This may be due to missing optional dependencies. "
            f"Error: {error_msg}\n"
            f"To install missing dependencies, try: pip install panospace[annotation]"
        )

    # Import the backend function
    module_path, func_name = _BACKENDS[name].split(":")
    mod = __import__(module_path, fromlist=[func_name])
    return getattr(mod, func_name)


def list_available_backends() -> dict[str, bool]:
    """
    List all registered backends and their availability status.

    Returns
    -------
    dict
        Dictionary mapping backend names to availability (True/False).
    """
    return _AVAILABLE_BACKENDS.copy()


def get_backend_error(backend: str) -> str:
    """
    Get the error message for a failed backend import.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    str
        Error message, or empty string if backend is available.
    """
    return _BACKEND_ERRORS.get(backend, "")
