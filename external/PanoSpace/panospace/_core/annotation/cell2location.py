from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def annotate_cells_core(
    sc_adata: ad.AnnData,
    adata_vis: ad.AnnData,
    celltype_key: str,
    cell_count_cutoff: int = 5,
    cell_percentage_cutoff2: float = 0.03,
    nonz_mean_cutoff: float = 1.12,
):
    adata_snrna_raw = sc_adata.copy()
    try:
        adata_snrna_raw.X = adata_snrna_raw.raw.X.copy()
    except:
        try:
            adata_snrna_raw.X = adata_snrna_raw.X.astype(int)
        except:
            adata_snrna_raw.X = adata_snrna_raw.X.toarray().astype(int)

    try:
        adata_vis.X = adata_vis.X.astype(int)
    except:
        adata_vis.X = adata_vis.X.toarray().astype(int)

    adata_snrna_raw.X = csr_matrix(adata_snrna_raw.X)
    adata_vis.X = csr_matrix(adata_vis.X)
    adata_snrna_raw = adata_snrna_raw[
        ~adata_snrna_raw.obs[celltype_key].isin(
            np.array(
                adata_snrna_raw.obs[celltype_key]
                .value_counts()[adata_snrna_raw.obs[celltype_key].value_counts() <= 1]
                .index
            )
        )
    ]

    sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
    sc.pp.filter_cells(adata_snrna_raw, min_genes=1)

    adata_snrna_raw.obs[celltype_key] = pd.Categorical(adata_snrna_raw.obs[celltype_key])
    adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isna(), :]

    from ._cell2location_backend.cell2location_utils import filter_genes

    selected = filter_genes(
        adata_snrna_raw,
        cell_count_cutoff=cell_count_cutoff,
        cell_percentage_cutoff2=cell_percentage_cutoff2,
        nonz_mean_cutoff=nonz_mean_cutoff,
    )

    # filter the object
    adata_snrna_raw = adata_snrna_raw[:, selected].copy()

    from ._cell2location_backend.cell2location_utils import RegressionModel

    RegressionModel.setup_anndata(
        adata_snrna_raw,
        labels_key=celltype_key,
    )
    mod = RegressionModel(adata_snrna_raw)

    # Use all data for training (validation not implemented yet, train_size=1)
    mod.train(max_epochs=250, batch_size=200, train_size=1, lr=0.002)
    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_snrna_raw = mod.export_posterior(adata_snrna_raw, sample_kwargs={"num_samples": 1000, "batch_size": 2500})

    # export estimated expression in each cluster
    if "means_per_cluster_mu_fg" in adata_snrna_raw.varm.keys():
        inf_aver = adata_snrna_raw.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in adata_snrna_raw.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = adata_snrna_raw.var[
            [f"means_per_cluster_mu_fg_{i}" for i in adata_snrna_raw.uns["mod"]["factor_names"]]
        ].copy()
    inf_aver.columns = adata_snrna_raw.uns["mod"]["factor_names"]

    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # create and train the model
    from ._cell2location_backend.cell2location_utils import Cell2location

    # prepare anndata for cell2location model
    Cell2location.setup_anndata(adata=adata_vis)
    # scvi.data.view_anndata_setup(adata_vis)

    mod = Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection (using default here):
        detection_alpha=200,
    )

    mod.train(
        max_epochs=30000,
        # train using full data (batch_size=None)
        batch_size=None,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1,
    )

    adata_vis = mod.export_posterior(adata_vis, sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs})
    adata_vis.obs[adata_vis.uns["mod"]["factor_names"]] = adata_vis.obsm["q05_cell_abundance_w_sf"]

    return adata_vis.obs[adata_vis.uns["mod"]["factor_names"]]
