from __future__ import annotations

import logging
from typing import Dict

from anndata import AnnData

from ._annotator_backend.annotator_utils import CellTypeAnnotator

logger = logging.getLogger(__name__)


def annotator_core(
    spot_adata: AnnData,
    sr_spot_adata: AnnData,
    seg_adata: AnnData,
    priori_type_affinities: Dict[str, float] | None = None,
    alpha: float = 0.3,
    ot_mode: str = "emd",  # "sinkhorn" or "emd"
    sinkhorn_reg: float = 0.01,  # Sinkhorn regularization
    _global_quota=True,
    _spot_quota=True,
    solver: str = "auto",
    dedup_overlapping_spots: bool = False,
) -> AnnData:
    """Annotate segmented cells with cell types using spot-level and super-resolved deconvolution results.

    Parameters
    ----------
    spot_adata
        AnnData object containing spot-level deconvolution results.
    sr_spot_adata
        AnnData object containing super-resolved spot-level deconvolution results.
    seg_adata
        AnnData object containing segmented cell data.
    priori_type_affinities
        Optional dictionary specifying prior affinities for cell types.
    alpha
        Fusion weight between spatial propagation and morphology prior.
    ot_mode
        Optimal transport mode to use - either "sinkhorn" or "emd".
    sinkhorn_reg
        Regularization parameter for Sinkhorn algorithm (if used).
    solver
        Assignment solver: ``"auto"`` (default), ``"flow"``, ``"gurobi"`` or ``"scip"``.
    dedup_overlapping_spots
        Keep a single (nearest) spot per segment when spot footprints overlap.

    Returns
    -------
    AnnData
        The annotated segmentation results.

    Notes
    -----
    The final assignment solves a 0/1 problem with per-segment, per-spot and
    global quota constraints. Three interchangeable exact solvers are wired in:
      - ``flow``   : OR-Tools min-cost flow (default, scales to whole slides)
      - ``gurobi`` : MILP (commercial licence)
      - ``scip``   : MILP (open-source, very slow above a few thousand cells)
    """

    _seg_adata_pred = CellTypeAnnotator(
        spot_adata=spot_adata,
        sr_spot_adata=sr_spot_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,
        alpha=alpha,
        ot_mode=ot_mode,  # "sinkhorn" or "emd"
        sinkhorn_reg=sinkhorn_reg,  # Sinkhorn regularization
        _global_quota=_global_quota,
        _spot_quota=_spot_quota,
        solver=solver,
        dedup_overlapping_spots=dedup_overlapping_spots,
    )
    _seg_adata_pred.filter_and_build_affiliations()
    _seg_adata_pred.compute_counts_and_integerize()
    if _seg_adata_pred.mode == "mor":
        _seg_adata_pred.build_type_transfer(factor=2.0)
    seg_adata_pred = _seg_adata_pred.infer_cell_types()

    return seg_adata_pred, _seg_adata_pred
