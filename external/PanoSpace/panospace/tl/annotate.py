"""panospace.tl.annotate
========================
High-level interface for **cell-type annotation** and deconvolution at
single-cell resolution in spatial transcriptomics.

This module provides three main functionalities:
    1. Perform cell-type deconvolution of spatial data using multiple
       backends (e.g., RCTD, cell2location, spatialDWLS).
    2. Integrate backend results into an ensemble consensus using
       the EnDecon algorithm.
    3. Refine and project cell-type assignments to super-resolved
       or segmented cell units.

Backends
--------
Supported gene-expression-based deconvolution backends:
* ``"RCTD"``          - Robust Cell Type Decomposition.
* ``"cell2location"`` - Probabilistic cell-type mapping model.
* ``"spatialDWLS"``   - Deconvolution via dampened weighted least squares.
* ``"endecon"``       - Ensemble integration of multiple backends.


"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)

from . import _import_backend


# -----------------------------------------------------------------------------
# Cell-type deconvolution
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _hash_df(df: pd.DataFrame) -> str:
    """Generate stable hash of DataFrame for cache validation.

    Slow on large DataFrames but faster than recomputing.
    """
    h = hashlib.sha256()
    h.update(("|".join(map(str, df.index))).encode("utf-8"))
    h.update(("|".join(map(str, df.columns))).encode("utf-8"))
    # Use pandas hash for stable value hashing
    hv = pd.util.hash_pandas_object(df, index=False).values
    h.update(hv.tobytes())
    return h.hexdigest()


def _save_df(df: pd.DataFrame, base_path: Path) -> Path:
    """Save DataFrame to disk, preferring parquet with csv.gz fallback.

    Returns actual file path used.
    """
    _ensure_dir(base_path.parent)

    # Try parquet first (faster/smaller)
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=True)
        return parquet_path
    except Exception:
        # Fallback to csv.gz
        csv_path = base_path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=True, compression="gzip")
        return csv_path


def _load_df(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv.gz")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    raise FileNotFoundError(f"No cache file found for {base_path} (.parquet or .csv.gz).")


def _save_meta(meta: Dict[str, Any], base_path: Path) -> Path:
    meta_path = base_path.with_suffix(".meta.json")
    _safe_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
    return meta_path


def _load_meta(base_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = base_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _check_result_df_schema(
    df: pd.DataFrame,
    celltypes: List[str],
    *,
    method: str,
    require_nonnegative: bool = True,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{method}: backend output must be a pandas DataFrame, got {type(df)}")

    missing_cols = [c for c in celltypes if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{method}: output missing celltype columns (examples): {missing_cols[:8]}")

    if df.index.has_duplicates:
        raise ValueError(f"{method}: output index has duplicates (spot IDs must be unique).")

    # Validate only celltype columns (ignore extra columns from backends)
    mat = df.loc[:, celltypes].to_numpy()
    if not np.isfinite(mat).all():
        raise ValueError(f"{method}: output contains NaN/Inf in celltype columns.")

    if require_nonnegative and (mat < 0).any():
        raise ValueError(f"{method}: output contains negative values (unexpected for abundance/fraction).")


def deconv_celltype(
    adata_vis: AnnData,
    sc_adata: AnnData,
    celltype_key: str,
    methods: List[Literal["RCTD", "cell2location", "spatialDWLS"]] = ["RCTD", "cell2location", "spatialDWLS"],
    *,
    cache_dir: str | Path = "deconv_cache",
    project_name: str = "default",
    resume: bool = True,
    continue_on_error: bool = True,
    require_nonnegative: bool = True,
    min_methods_for_ensemble: int = 2,
) -> AnnData:
    """Perform cell-type deconvolution with checkpoint/resume support.

    Each backend writes results to cache_dir/project_name/ upon completion.
    Supports resuming after interruption or failures. Individual backend
    failures don't block execution (controlled by continue_on_error).

    Cache files:
    - deconv_<method>.parquet (or .csv.gz)
    - deconv_<method>.meta.json
    """

    if celltype_key not in sc_adata.obs:
        raise KeyError(f"Reference AnnData must contain '{celltype_key}' in .obs")

    cache_dir = Path(cache_dir) / project_name
    _ensure_dir(cache_dir)

    celltypes = sorted(sc_adata.obs[celltype_key].unique().tolist())
    adata_vis.uns["celltype"] = celltypes

    success_results: Dict[str, pd.DataFrame] = {}
    failed: Dict[str, str] = {}

    # Lightweight meta for cache validation: celltypes hash, basic shape info
    celltypes_sig = hashlib.sha256(("|".join(celltypes)).encode("utf-8")).hexdigest()
    sc_sig = hashlib.sha256(str(sc_adata.shape).encode("utf-8")).hexdigest()
    vis_sig = hashlib.sha256(str(adata_vis.shape).encode("utf-8")).hexdigest()

    for method in methods:
        base_path = cache_dir / f"deconv_{method}"

        # 1) Try loading from cache
        if resume:
            try:
                meta = _load_meta(base_path)
                if meta is not None:
                    ok = (
                        meta.get("celltypes_sig") == celltypes_sig
                        and meta.get("sc_shape") == list(sc_adata.shape)
                        and meta.get("vis_shape") == list(adata_vis.shape)
                    )
                    if not ok:
                        # Cache doesn't match inputs, skip to avoid wrong results
                        pass
                    else:
                        df = _load_df(base_path)
                        _check_result_df_schema(df, celltypes, method=method, require_nonnegative=require_nonnegative)
                        adata_vis.uns[f"X_deconv_{method}"] = df
                        success_results[method] = df
                        logger.info("Loaded cached deconvolution for '%s' from %s", method, base_path)
                        continue
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Found cache for '%s' but failed to load/validate, will recompute. Err=%s", method, e)

        # 2) Run backend and cache results
        logger.info("Running cell-type deconvolution with backend '%s'", method)

        try:
            t0 = time.time()
            backend_fn = _import_backend(method)
            result_df: pd.DataFrame = backend_fn(
                sc_adata=sc_adata,
                adata_vis=adata_vis,
                celltype_key=celltype_key,
            )
            _check_result_df_schema(result_df, celltypes, method=method, require_nonnegative=require_nonnegative)

            # Store in uns for intra-process access
            adata_vis.uns[f"X_deconv_{method}"] = result_df
            success_results[method] = result_df

            # Write checkpoint to disk
            file_path = _save_df(result_df, base_path)
            meta = {
                "method": method,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "celltype_key": celltype_key,
                "celltypes": celltypes,
                "celltypes_sig": celltypes_sig,
                "sc_shape": list(sc_adata.shape),
                "vis_shape": list(adata_vis.shape),
                "sc_sig": sc_sig,
                "vis_sig": vis_sig,
                "df_hash": _hash_df(result_df),
                "file": str(file_path.name),
                "elapsed_sec": round(time.time() - t0, 3),
            }
            _save_meta(meta, base_path)

            logger.info("Saved checkpoint for '%s' to %s (%.2fs)", method, file_path, meta["elapsed_sec"])

        except Exception as e:
            failed[method] = "".join(traceback.format_exception_only(type(e), e)).strip()
            logger.exception("Backend '%s' failed.", method)
            if not continue_on_error:
                raise

    if len(success_results) == 0:
        raise RuntimeError(f"All backends failed. Errors: {failed}")

    # ---- Align spots and perform ensemble (or downgrade) ----
    used_methods = list(success_results.keys())
    results_list = [success_results[m] for m in used_methods]

    # Find common spots across backends
    common_index = sorted(set.intersection(*(set(df.index) for df in results_list)))
    if len(common_index) == 0:
        raise RuntimeError("No common spots among successful backends (check spot IDs / filtering).")

    aligned_results = [df.loc[common_index, celltypes].to_numpy(dtype=float) for df in results_list]

    # If insufficient methods, downgrade to single-method result
    if len(aligned_results) < min_methods_for_ensemble:
        logger.warning(
            "Only %d backend(s) succeeded (<%d). Skip EnDecon; use '%s' as final result.",
            len(aligned_results),
            min_methods_for_ensemble,
            used_methods[0],
        )
        H_final = pd.DataFrame(aligned_results[0], index=common_index, columns=celltypes)
        ensemble_used = [used_methods[0]]
    else:
        logger.info("Running EnDecon ensemble integration with methods: %s", used_methods)
        EnDecon = _import_backend("endecon")
        ensemble_result = EnDecon(aligned_results)
        H_final = pd.DataFrame(ensemble_result["H_norm"], index=common_index, columns=celltypes)
        ensemble_used = used_methods

    # ---- Construct output AnnData ----
    deconv_adata = adata_vis.copy()[common_index].copy()
    deconv_adata.obs = deconv_adata.obs.join(H_final, how="left")

    # Store provenance metadata
    deconv_adata.uns["deconv_success_methods"] = used_methods
    deconv_adata.uns["deconv_failed_methods"] = failed
    deconv_adata.uns["deconv_ensemble_used"] = ensemble_used
    deconv_adata.uns["deconv_cache_dir"] = str(cache_dir)
    deconv_adata.uns["X_deconv_ensemble"] = H_final

    return deconv_adata


# -----------------------------------------------------------------------------
# Super-resolution refinement
# -----------------------------------------------------------------------------
def superres_celltype(
    deconv_adata: AnnData,
    img_dir: str,
    neighb: int = 3,
    radius: int = 129,
    epoch: int = 50,
    class_weights=None,
    learning_rate: float = 1e-4,
    local_path: str = "~/.panospace_cache/dinov2-base",
    pretrained_model_name: str = "facebook/dinov2-base",
    cache_dir: str = "~/.panospace_cache",
    batch_size: int = 32,
    num_workers: int = 0,
    accelerator: Literal["cpu", "gpu"] = "gpu",
    precision: Literal["16-mixed", "32", "64"] = "16-mixed",
    devices: Union[int, str] = "auto",
    seed: int = 42,
    patience: Union[int, None] = None,
    val_frac: float = 0.15,
    mask_mode: Literal["largest", "spots", "none"] = "largest",
    mask_downscale: Union[int, str] = 1,
    mask_min_spots: int = 1,
) -> AnnData:
    """
    Refine deconvolution results at higher spatial resolution using DINOv2.

    Parameters
    ----------
    deconv_adata : AnnData
        AnnData object with initial deconvolution results.
    img_dir : str
        Path to tissue image corresponding to ``deconv_adata``.
    neighb : int, optional
        Number of neighboring spots/cells considered (default: 3).
    radius : int, optional
        Radius for image patch extraction (default: 129).
    epoch : int, optional
        Maximum number of training epochs (default: 50); early stopping usually
        halts training well before this.
    batch_size : int, optional
        Batch size for MLP training (default: 32). DINOv2 features are
        pre-extracted once, so the ViT input size no longer bounds this.
    num_workers : int, optional
        DataLoader workers (default: 0). Pre-extracted features live in memory
        as tensors, so extra workers usually add IPC overhead for no gain.
    accelerator : {"cpu", "gpu"}, optional
        Compute device (default: "gpu").
    precision : {"16-mixed", "32", "64"}, optional
        Mixed precision mode forwarded to ``pl.Trainer`` (default: "16-mixed").
        Pass ``"32"`` for full fp32 if you hit numerical issues.
    devices : int or "auto", optional
        Number of devices for ``pl.Trainer`` (default: "auto").
    seed : int, optional
        Random seed for ``pl.seed_everything`` and the train/val split
        (default: 42).
    patience : int or None, optional
        Early-stopping patience on ``val_loss`` (default: ``None`` = disabled).
        The default reflects that this pipeline predicts on the same section
        it trains on — there's no held-out generalization target, so disabling
        early stopping fits the training spots as tightly as possible. Pass a
        positive int (e.g. 8) to enable early stopping with a ``val_frac``
        hold-out.
    val_frac : float, optional
        Fraction of spots held out for validation when ``patience`` is set
        (default: 0.15). Ignored when ``patience is None``.
    mask_mode : {"largest", "spots", "none"}, optional
        Which Canny contours define the tissue, and therefore which sub-spots
        are kept. ``"largest"`` (default, upstream behaviour) keeps the single
        biggest contour, which silently discards every other piece of a
        multi-fragment section. ``"spots"`` keeps every fragment carrying at
        least one measured spot -- real tissue in, debris out. ``"none"`` keeps
        the whole image.
    mask_min_spots : int, optional
        With ``mask_mode="spots"``, the minimum number of measured spots a
        fragment must carry to be kept (default: 1). Raising it to a handful
        drops specks of debris that happen to catch a single spot centre, while
        leaving genuine tissue pieces untouched.
    mask_downscale : int or "auto", optional
        Run the contour detection on a 1/N proxy of the slide (default
        ``"auto"``: whatever keeps the proxy under ~16 MP). The tissue outline
        is far coarser than the sub-spot pitch, so this only saves time and
        memory. Pass ``1`` for the upstream full-resolution behaviour.

    Returns
    -------
    AnnData
        ``adata_vis`` with super-resolved deconvolution outputs.
    """
    logger.info("Running super-resolution deconvolution...")
    superres_fn = _import_backend("superres_core")

    sr_adata = superres_fn(
        deconv_adata=deconv_adata,
        img_dir=img_dir,
        neighb=neighb,
        radius=radius,
        class_weights=class_weights,
        learning_rate=learning_rate,
        local_path=local_path,
        pretrained_model_name=pretrained_model_name,
        cache_dir=cache_dir,
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
        precision=precision,
        devices=devices,
        seed=seed,
        patience=patience,
        val_frac=val_frac,
        mask_mode=mask_mode,
        mask_downscale=mask_downscale,
        mask_min_spots=mask_min_spots,
    )
    return sr_adata


# -----------------------------------------------------------------------------
# Cell-type annotation of segments
# -----------------------------------------------------------------------------
def celltype_annotator(
    decov_adata: AnnData,
    sr_deconv_adata: AnnData,
    seg_adata: AnnData,
    priori_type_affinities=None,
    alpha: float = 0.3,
    ot_mode: str = "emd",
    sinkhorn_reg: float = 0.01,
    _global_quota=True,
    _spot_quota=True,
    solver: str = "auto",
    dedup_overlapping_spots: bool = False,
) -> AnnData:
    """
    Assign cell types to segmented cells using deconvolution outputs.

    Combines spot-level deconvolution, super-resolved refinement,
    and segmentation boundaries to obtain consistent cell-type
    assignments via optimal transport and MILP optimization.

    Parameters
    ----------
    decov_adata : AnnData
        Spot-level deconvolution results.
    sr_deconv_adata : AnnData
        Super-resolved deconvolution results.
    seg_adata : AnnData
        Segmentation results (cell masks/coordinates).
    priori_type_affinities : dict, optional
        Prior affinities between cell types.
    alpha : float, optional
        Fusion weight between spatial propagation and morphology prior (default: 0.3).
    ot_mode : {"sinkhorn", "emd"}, optional
        Optimal transport variant for aligning cell types to morphology (default: "emd").
    sinkhorn_reg : float, optional
        Sinkhorn entropy regularization (default: 0.01).
    solver : {"auto", "flow", "gurobi", "scip"}, optional
        Backend for the final 0/1 assignment (default: ``"auto"``). ``"flow"``
        is an OR-Tools min-cost flow that solves exactly the same problem as the
        MILP -- the constraint matrix is totally unimodular -- but stays
        tractable on whole slides. ``"auto"`` picks flow > gurobi > scip.
    dedup_overlapping_spots : bool, optional
        When spot footprints overlap, keep only the nearest spot for each
        segment so the spot-level equality constraints stay consistent
        (default: True; a no-op for Visium-like geometries).

    Returns
    -------
    tuple of (AnnData, CellTypeAnnotator)
        ``seg_adata_pred`` carries ``.obs['pred_cell_type']`` plus one 0/1
        column per cell type; the second element is the fitted annotator, kept
        for inspection of the intermediate quotas / affiliations.
    """
    logger.info("Running cell-type annotation for segmented cells...")
    annotator = _import_backend("annotator_core")

    seg_adata_pred, _seg_adata_pred = annotator(
        spot_adata=decov_adata,
        sr_spot_adata=sr_deconv_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,
        alpha=alpha,
        ot_mode=ot_mode,
        sinkhorn_reg=sinkhorn_reg,
        _global_quota=_global_quota,
        _spot_quota=_spot_quota,
        solver=solver,
        dedup_overlapping_spots=dedup_overlapping_spots,
    )
    return seg_adata_pred, _seg_adata_pred


__all__ = ["deconv_celltype", "superres_celltype", "celltype_annotator"]
