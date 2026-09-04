from __future__ import annotations

import logging
import os

import anndata as ad
import numpy as np
import pandas as pd

from ._RCTD_backend.RCTD_utils import *


logger = logging.getLogger(__name__)


def annotate_cells_core(
    sc_adata: ad.AnnData,
    adata_vis: ad.AnnData,
    celltype_key: str,
    UMI_min: int = 100,
    UMI_min_sigma: int = 100,
):

    coords = pd.DataFrame(adata_vis.obsm["spatial"], index=adata_vis.obs_names, columns=["x", "y"])
    counts = adata_vis.to_df().T
    nUMI = pd.DataFrame(np.array(adata_vis.X.sum(-1)), index=adata_vis.obs.index)

    puck = SpatialRNA(coords, counts, nUMI)

    Q_mat_all, X_vals_loc = LoadLikelihoodTable()

    counts = sc_adata.to_df().T
    cell_types = pd.DataFrame(sc_adata.obs[celltype_key])
    nUMI = pd.DataFrame(sc_adata.to_df().T.sum(0))
    reference = Reference(counts, cell_types, nUMI, loggings=logger)
    # Upstream value is 22. Kept as the default so the method is unchanged;
    # set PANOSPACE_RCTD_CORES to stay inside a smaller Slurm allocation
    # (worker count does not affect the RCTD result, only the wall time).
    max_cores = int(os.environ.get("PANOSPACE_RCTD_CORES", 0)) or 22
    myRCTD = create_RCTD(
        puck, reference, max_cores=max_cores, UMI_min=UMI_min, UMI_min_sigma=UMI_min_sigma, loggings=logger
    )
    myRCTD = run_RCTD(myRCTD, Q_mat_all, X_vals_loc, loggings=logger)
    df = myRCTD["results"]
    df_clipped = df.clip(lower=0)

    row_sums = df_clipped.sum(axis=1).replace(0, np.nan)

    df_normalized = df_clipped.div(row_sums, axis=0).fillna(0)
    return df_normalized
