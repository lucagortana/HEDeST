from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from ._spatialDWLS_backend.spatialDWLS_utils import *


logger = logging.getLogger(__name__)


def annotate_cells_core(
    sc_adata: ad.AnnData,
    adata_vis: ad.AnnData,
    celltype_key: str,
    target_sum: int = 1e4,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
    n_neighbors: int = 15,
    n_pca: int = 10,
    resolution: float = 0.4,
    method: str = "wilcoxon",
    n_genes: int = 100,
):

    adata_vis.raw = adata_vis.copy()  # Preserve original data
    logger.info("Preprocessing Visium data...")
    sc.pp.normalize_total(adata_vis, target_sum=target_sum)
    sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, n_top_genes=n_top_genes, flavor=flavor)
    adata_vis = adata_vis[:, adata_vis.var.highly_variable]

    logger.info("Computing PCA...")
    # Convert to dense array if sparse, keep as-is if already dense
    if hasattr(adata_vis.X, "toarray"):
        adata_vis.X = adata_vis.X.toarray()
    else:
        adata_vis.X = np.asarray(adata_vis.X)
    sc.pp.pca(adata_vis, svd_solver="arpack")
    logger.info("Computing neighbors...")
    sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, n_pcs=n_pca)
    logger.info("Computing UMAP...")
    sc.tl.umap(adata_vis)
    logger.info("Computing Leiden clustering...")
    sc.tl.leiden(adata_vis, resolution=resolution)

    logger.info("Preprocessing scRNA-seq reference data...")
    sc_adata.raw = sc_adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=target_sum)
    sc.pp.log1p(sc_adata)
    sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes, flavor=flavor)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]

    logger.info("PCA on scRNA-seq reference data...")
    sc.pp.pca(sc_adata, svd_solver="arpack")

    # Add celltype_final annotation
    logger.info("Identifying marker genes per cell type...")
    sc_adata.obs["leiden_clus"] = sc_adata.obs[celltype_key].astype(str)
    sc.tl.rank_genes_groups(
        sc_adata,
        groupby="leiden_clus",
        method=method,  # Giotto uses scran, we use wilcoxon as approximation
        n_genes=n_genes,
    )

    # Extract marker genes
    logger.info("Extracting marker genes and computing cell-type signatures...")
    marker_df = pd.DataFrame(
        {
            group: sc_adata.uns["rank_genes_groups"]["names"][group]
            for group in sc_adata.uns["rank_genes_groups"]["names"].dtype.names
        }
    )
    Sig_scran = marker_df.melt()["value"].dropna().unique()

    # Convert to dense array if sparse, keep as-is if already dense
    if hasattr(sc_adata.X, "toarray"):
        sc_X = sc_adata.X.toarray()
    else:
        sc_X = sc_adata.X
    norm_exp = np.expm1(sc_X)
    Sig_exp = []

    # Compute mean per cell type
    logger.info("Computing cell-type signature expression profiles...")
    for cluster in sc_adata.obs["leiden_clus"].unique():
        cluster_mask = sc_adata.obs["leiden_clus"] == cluster
        cluster_mean = norm_exp[cluster_mask, :].mean(axis=0)
        Sig_exp.append(cluster_mean)

    Sig_exp = np.array(Sig_exp).T  # shape: genes x clusters
    Sig_exp_df = pd.DataFrame(Sig_exp, index=sc_adata.var_names, columns=sc_adata.obs["leiden_clus"].unique())

    # Sig_exp_df = Sig_exp_df.loc[Sig_exp_df.index.isin(Sig_scran)] # Keep only marker genes
    logger.info("Retaining %d marker genes for deconvolution.", Sig_exp_df.shape[0])
    X = adata_vis.raw.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr_df = pd.DataFrame(X, index=adata_vis.raw.obs_names, columns=adata_vis.raw.var_names).T
    log_expr_df = adata_vis.to_df().T

    logger.info("Start spatialDWLS deconvolution...")
    results = runDWLSDeconv(
        expr_df=expr_df, log_expr_df=log_expr_df, cluster_info=adata_vis.obs["leiden"].tolist(), ct_exp_df=Sig_exp_df
    )
    return results
