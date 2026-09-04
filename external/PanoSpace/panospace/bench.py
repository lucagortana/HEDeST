"""panospace.bench
=================
I/O adapters that plug PanoSpace into the HEDeST benchmark.

This module is **not** part of upstream PanoSpace.  It exists so that the
pipeline consumes / produces exactly the file formats used elsewhere in the
HEDeST repository:

======================  =====================================================
HEDeST / STHELAR file   role here
======================  =====================================================
``he.tiff``             histology image (full resolution, RGB)
``pseudovisium.h5ad``   spot-level ST AnnData (``.X`` counts, ``.obsm['spatial']``)
``hovernet.json``       nuclei segmentation, ``{"mag", "mpp", "nuc": {id: {...}}}``
``proportions.csv``     spot x cell-type proportions (index name ``spot_id``)
prediction CSV          cell x cell-type table (index name ``cell_id``),
                        same shape as HEDeST's ``pred_best_adjusted``
======================  =====================================================

Coordinate convention
---------------------
Everything is in **full-resolution image pixels**, ``(x, y)`` = ``(column,
row)``.  HoVer-Net's ``centroid`` / ``contour`` already use that order, and so
does ``adata.obsm['spatial']`` for the STHELAR pseudo-Visium objects.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PanNuke morphology classes, in the integer coding shared by HoVer-Net and
# CellViT.  ``annotator_utils.CellTypeAnnotator`` re-declares the same table;
# it is repeated here so the segmentation I/O is self-contained.
PANNUKE_TYPES: Dict[int, str] = {
    0: "nolabel",
    1: "neopla",
    2: "inflam",
    3: "connec",
    4: "necros",
    5: "no-neo",
}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def load_seg_dict(path: str) -> Dict[str, Any]:
    """Read a HoVer-Net-format segmentation JSON.

    Accepts both the full HoVer-Net document ``{"mag":..., "mpp":..., "nuc":
    {...}}`` and a bare ``{cell_id: {...}}`` mapping, and always returns the
    full form.
    """
    with open(path, "r") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")

    if "nuc" not in raw:
        # bare {cell_id: {...}} mapping
        raw = {"mag": None, "mpp": None, "nuc": raw}

    nuc = raw["nuc"]
    if not isinstance(nuc, dict) or len(nuc) == 0:
        raise ValueError(f"{path}: 'nuc' is empty or not a mapping")

    logger.info("Loaded segmentation from %s (%d cells)", path, len(nuc))
    return raw


def seg_adata_from_hovernet(seg_dict: Dict[str, Any], use_morphology: bool = True):
    """Build the ``seg_adata`` PanoSpace expects from a HoVer-Net dict.

    Parameters
    ----------
    seg_dict
        As returned by :func:`load_seg_dict`.
    use_morphology
        When True (default) the PanNuke ``type`` of each nucleus is written to
        ``.obs['img_type']``, which switches ``CellTypeAnnotator`` into its
        ``'mor'`` mode (morphology prior blended in with weight ``alpha``).
        Set to False to run the SR-propagation branch alone.

    Returns
    -------
    AnnData
        ``n_cells x 1`` dummy matrix, ``.obs_names`` = the segmentation cell
        ids (so predictions can be mapped straight back), ``.obsm['spatial']``
        = centroids, optionally ``.obs['img_type']``.
    """
    import anndata

    nuc = seg_dict["nuc"]
    cell_ids = list(nuc.keys())

    centroids = np.empty((len(cell_ids), 2), dtype=np.float64)
    img_types = np.empty(len(cell_ids), dtype=np.int64)
    for i, cid in enumerate(cell_ids):
        info = nuc[cid]
        centroids[i] = info["centroid"]
        img_types[i] = int(info.get("type", 0))

    seg_adata = anndata.AnnData(X=np.ones((len(cell_ids), 1), dtype=np.float32))
    seg_adata.obs_names = pd.Index([str(c) for c in cell_ids], name="cell_id")
    seg_adata.obs["contour_id"] = np.arange(len(cell_ids))
    if use_morphology:
        seg_adata.obs["img_type"] = img_types
    seg_adata.obsm["spatial"] = centroids

    logger.info(
        "seg_adata: %d cells, morphology=%s, x=[%.0f, %.0f] y=[%.0f, %.0f]",
        seg_adata.n_obs,
        use_morphology,
        centroids[:, 0].min(),
        centroids[:, 0].max(),
        centroids[:, 1].min(),
        centroids[:, 1].max(),
    )
    return seg_adata


def hovernet_dict_from_cellvit(
    cell_dict_wsi: Sequence[Dict[str, Any]],
    mpp: Optional[float] = None,
    mag: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert PanoSpace/CellViT detections into a HoVer-Net-shaped dict.

    The upstream CellViT backend returns one dict per cell with global
    ``centroid`` / ``contour`` in ``(x, y)`` and a PanNuke integer ``type``.
    HoVer-Net JSON additionally carries ``bbox`` in ``[[y0, x0], [y1, x1]]``
    and a ``type_prob`` / ``type_name``; ``bbox`` is recomputed from the
    contour so the two formats agree.
    """
    nuc: Dict[str, Any] = {}
    for i, cell in enumerate(cell_dict_wsi):
        contour = np.asarray(cell["contour"], dtype=np.int64)
        x0, y0 = contour.min(axis=0)
        x1, y1 = contour.max(axis=0)
        ctype = int(cell.get("type", 0))
        nuc[str(i)] = {
            "bbox": [[int(y0), int(x0)], [int(y1), int(x1)]],
            "centroid": [float(cell["centroid"][0]), float(cell["centroid"][1])],
            "contour": contour.tolist(),
            "type_prob": float(cell.get("type_prob", 1.0)),
            "type": ctype,
            "type_name": PANNUKE_TYPES.get(ctype, str(ctype)),
        }
    return {"mag": mag, "mpp": mpp, "nuc": nuc}


def write_seg_dict(seg_dict: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(seg_dict, fh)
    logger.info("Wrote segmentation (%d cells) -> %s", len(seg_dict["nuc"]), path)


# ---------------------------------------------------------------------------
# Spatial transcriptomics
# ---------------------------------------------------------------------------
def load_st_adata(path: str):
    """Read the spot-level ST AnnData and check it carries spatial coordinates."""
    import anndata

    adata = anndata.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise KeyError(f"{path}: .obsm['spatial'] is required (spot pixel coordinates)")
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if adata.var_names.has_duplicates:
        logger.warning(
            "%s: duplicated gene names. Upstream PanoSpace does not de-duplicate, "
            "so they are left as-is; call .var_names_make_unique() yourself if a "
            "deconvolution backend complains.",
            path,
        )
    logger.info("Loaded ST data from %s: %d spots x %d genes", path, adata.n_obs, adata.n_vars)
    return adata


def infer_spot_radius(
    adata,
    radius: Optional[float] = None,
    mpp: Optional[float] = None,
    spot_diameter_um: Optional[float] = None,
) -> int:
    """Resolve the spot radius, in full-resolution pixels.

    Resolution order:

    1. ``radius`` when given explicitly;
    2. ``spot_diameter_um / (2 * mpp)`` when both are given;
    3. ``adata.uns['spatial'][key]['scalefactors']['spot_diameter_fullres'] / 2``;
    4. ``adata.uns['pseudovisium']['spot_diameter_um']`` together with
       ``adata.uns['pseudovisium']['mpp']`` (STHELAR pseudo-Visium objects).

    Returned as an **integer**, which is what upstream PanoSpace expects: its
    quick-start has the user type ``deconv_adata.uns['radius'] = 100`` by hand,
    and the value is used unrounded as the ``range`` step of the sub-spot grid
    (a float raises ``TypeError`` there).  This helper only removes the manual
    step -- it reads the same number off the ST object instead of asking for it.
    """
    if radius is not None:
        out = float(radius)
        src = "--spot-radius"
    elif mpp is not None and spot_diameter_um is not None:
        out = float(spot_diameter_um) / (2.0 * float(mpp))
        src = "spot_diameter_um / (2*mpp)"
    else:
        out, src = None, None
        spatial = adata.uns.get("spatial", {})
        for key, entry in (spatial or {}).items():
            sf = (entry or {}).get("scalefactors", {})
            if "spot_diameter_fullres" in sf:
                out = float(sf["spot_diameter_fullres"]) / 2.0
                src = f"uns['spatial']['{key}']['scalefactors']['spot_diameter_fullres']/2"
                break
        if out is None:
            pv = adata.uns.get("pseudovisium", {}) or {}
            if "spot_diameter_um" in pv and "mpp" in pv:
                out = float(pv["spot_diameter_um"]) / (2.0 * float(pv["mpp"]))
                src = "uns['pseudovisium'] spot_diameter_um / (2*mpp)"
        if out is None:
            raise ValueError(
                "Could not determine the spot radius. Pass --spot-radius (pixels), or "
                "--mpp together with --spot-diameter-um."
            )

    out_i = int(round(float(out)))
    if out_i < 1:
        raise ValueError(f"Spot radius resolved to {out_i} px, which is not usable.")
    logger.info("Spot radius: %d px (from %s; exact value %.3f)", out_i, src, float(out))
    return out_i


# ---------------------------------------------------------------------------
# Cell-type proportions
# ---------------------------------------------------------------------------
def load_proportions(path: str) -> pd.DataFrame:
    """Read a ``proportions.csv`` (first column = spot id, others = cell types)."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicated spot ids")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"{path}: non-numeric proportion columns {non_numeric}")
    logger.info("Loaded proportions from %s: %d spots x %d cell types", path, *df.shape)
    return df


def normalize_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to non-negative and rescale each row to sum to 1."""
    mat = np.clip(df.to_numpy(dtype=float), 0.0, None)
    row = mat.sum(axis=1, keepdims=True)
    row[row <= 0] = 1.0
    return pd.DataFrame(mat / row, index=df.index, columns=df.columns)


def attach_proportions(adata, prop_df: pd.DataFrame):
    """Build a ``deconv_adata`` from user-supplied proportions.

    Mirrors what ``ps.deconv_celltype`` produces: the spots are restricted to
    those present in both objects, one ``.obs`` column is added per cell type,
    ``.uns['celltype']`` lists the cell types and ``.uns['X_deconv_ensemble']``
    holds the proportion matrix.
    """
    spots = [s for s in adata.obs_names.astype(str) if s in prop_df.index]
    if len(spots) == 0:
        raise ValueError(
            "No spot id is shared between the ST object and the proportions file. "
            f"ST example ids: {list(adata.obs_names[:3])}; "
            f"proportions example ids: {list(prop_df.index[:3])}"
        )
    if len(spots) < adata.n_obs:
        logger.warning(
            "%d/%d ST spots have no proportions and are dropped",
            adata.n_obs - len(spots),
            adata.n_obs,
        )

    celltypes = [str(c) for c in prop_df.columns]
    sub = adata[[s for s in spots]].copy()
    sub.obs_names = sub.obs_names.astype(str)

    aligned = prop_df.loc[spots]
    aligned.columns = celltypes
    for ct in celltypes:
        sub.obs[ct] = aligned[ct].to_numpy(dtype=float)

    sub.uns["celltype"] = celltypes
    sub.uns["X_deconv_ensemble"] = aligned
    sub.uns["deconv_source"] = "user-supplied proportions"
    logger.info("deconv_adata from file: %d spots x %d cell types", sub.n_obs, len(celltypes))
    return sub


def proportions_from_deconv(deconv_adata) -> pd.DataFrame:
    """Extract the spot x cell-type table from a ``deconv_celltype`` output."""
    celltypes = [str(c) for c in deconv_adata.uns["celltype"]]
    df = pd.DataFrame(
        deconv_adata.obs[celltypes].to_numpy(dtype=float),
        index=deconv_adata.obs_names.astype(str),
        columns=celltypes,
    )
    return df


def write_proportions(df: pd.DataFrame, path: str, normalize: bool = True) -> None:
    """Write a ``proportions.csv`` in the STHELAR layout (index name ``spot_id``)."""
    out = normalize_proportions(df) if normalize else df.copy()
    out.index = out.index.astype(str)
    out.index.name = "spot_id"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path)
    logger.info("Wrote proportions (%d spots x %d cell types) -> %s", *out.shape, path)


# ---------------------------------------------------------------------------
# Single-cell predictions
# ---------------------------------------------------------------------------
def predictions_to_frame(seg_adata_pred, cell_types: Sequence[str]) -> pd.DataFrame:
    """Shape the annotator output like HEDeST's ``pred_best_adjusted``.

    HEDeST returns a ``cell_id`` x ``cell type`` DataFrame of probabilities and
    the benchmark takes ``df.idxmax(axis=1)``.  PanoSpace assigns exactly one
    type per cell (MILP), so the same frame is produced with one-hot rows —
    ``idxmax`` then recovers the assigned label.
    """
    cell_types = [str(c) for c in cell_types]
    missing = [c for c in cell_types if c not in seg_adata_pred.obs.columns]
    if missing:
        raise KeyError(f"Annotator output is missing one-hot columns: {missing}")

    df = pd.DataFrame(
        seg_adata_pred.obs[cell_types].to_numpy(dtype=float),
        index=seg_adata_pred.obs_names.astype(str),
        columns=cell_types,
    )
    df.index.name = "cell_id"
    return df


def write_predictions(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path)
    logger.info("Wrote predictions (%d cells x %d cell types) -> %s", *df.shape, path)


# ---------------------------------------------------------------------------
# Single-cell reference
# ---------------------------------------------------------------------------
def load_sc_reference(
    path: str,
    celltype_key: str,
    layer: Optional[str] = None,
    min_cells_per_type: int = 0,
    max_cells_per_type: Optional[int] = None,
    drop_types: Optional[Sequence[str]] = None,
    drop_na_labels: bool = False,
    seed: int = 42,
):
    """Load and lightly filter the single-cell reference used for deconvolution.

    **Nothing is filtered by default**: upstream PanoSpace passes the reference
    straight to the backends, so this helper does too.  Every option below is
    opt-in and exists only for practical reasons (RCTD densifies the whole
    reference, so a 175k-cell atlas needs ``max_cells_per_type``).
    """
    import anndata
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if celltype_key not in adata.obs:
        raise KeyError(f"{path}: '{celltype_key}' not found in .obs " f"(available: {list(adata.obs.columns)[:20]})")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"{path}: layer '{layer}' not found (have {list(adata.layers)})")
        adata = anndata.AnnData(X=adata.layers[layer].copy(), obs=adata.obs.copy(), var=adata.var.copy())

    if drop_na_labels:
        labels = adata.obs[celltype_key].astype(str)
        bad = {"nan", "NaN", "None", "NA", "unknown", "Unknown", ""}
        n0 = adata.n_obs
        adata = adata[~labels.isin(bad).to_numpy()].copy()
        if adata.n_obs < n0:
            logger.info("Dropped %d cells with an unusable cell-type label", n0 - adata.n_obs)
    adata.obs[celltype_key] = adata.obs[celltype_key].astype(str)

    if drop_types:
        adata = adata[~adata.obs[celltype_key].isin(list(drop_types)).to_numpy()].copy()

    if min_cells_per_type > 0:
        counts = adata.obs[celltype_key].value_counts()
        keep = counts[counts >= min_cells_per_type].index
        adata = adata[adata.obs[celltype_key].isin(keep).to_numpy()].copy()

    if max_cells_per_type:
        rng = np.random.default_rng(seed)
        chosen: List[np.ndarray] = []
        for _, sub in adata.obs.groupby(celltype_key, observed=True):
            idx = sub.index.to_numpy()
            if len(idx) > max_cells_per_type:
                idx = rng.choice(idx, max_cells_per_type, replace=False)
            chosen.append(idx)
        adata = adata[np.concatenate(chosen)].copy()

    adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key])
    if adata.var_names.has_duplicates:
        logger.warning("%s: duplicated gene names, left as-is (upstream behaviour).", path)
    logger.info(
        "Loaded reference from %s: %d cells x %d genes, %d cell types",
        path,
        adata.n_obs,
        adata.n_vars,
        adata.obs[celltype_key].nunique(),
    )
    return adata


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def align_st_to_reference(adata_vis, sc_adata) -> Tuple[Any, Any]:
    """Restrict both objects to their shared genes (deconvolution needs this)."""
    shared = np.intersect1d(
        np.asarray(adata_vis.var_names, dtype=str),
        np.asarray(sc_adata.var_names, dtype=str),
    )
    if shared.size == 0:
        raise ValueError(
            "ST and reference share no gene name. Check that both use the same " "identifiers (symbols vs Ensembl ids)."
        )
    logger.info("Genes: ST %d, reference %d, shared %d", adata_vis.n_vars, sc_adata.n_vars, shared.size)
    return adata_vis[:, shared].copy(), sc_adata[:, shared].copy()
