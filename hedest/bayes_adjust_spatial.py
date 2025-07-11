from __future__ import annotations

import math
import pickle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from hedest.basics import revert_dict
from hedest.config import TqdmToLogger

tqdm_out = TqdmToLogger(logger, level="INFO")


class CellProbDataset(Dataset):
    def __init__(self, p_cell: torch.Tensor, p_local: torch.Tensor, beta: torch.Tensor):
        self.p_cell = p_cell  # (N, n_types)
        self.p_local = p_local  # (N, n_types)
        self.beta = beta  # (N,)

    def __len__(self):
        return self.p_cell.size(0)

    def __getitem__(self, idx):
        return self.p_cell[idx], self.p_local[idx], self.beta[idx], idx


class BayesianAdjustmentSpatial:
    r"""
    Bayesian adjustment of per‑cell type probabilities that leverages both
    **local spot‑level composition** and **global (slide‑level) priors**.

    This implementation extends the original *BayesianAdjustment* by also
    adjusting cells that fall *outside* Visium spots:

    1.  For cells **inside a spot** we reuse the spot‑level proportions.
    2.  For cells **outside** we build a *cell‑specific* local proportion
        vector by averaging the nearest ≤3 spots contained within a search
        radius of **2 × spot_diameter**.  The contribution of each spot is
        weighted linearly with distance – a spot sitting exactly on the
        cell gets weight 1, while a spot at the edge of the radius gets
        weight 0.  If no spot lies in range the probability vector is left
        untouched.

    Parameters
    ----------
    cell_prob_df : pd.DataFrame
        *n_cells × n_types* matrix with raw model probabilities.
    spot_dict : Dict[str, List[str]]
        Mapping *spot ID → list(cell IDs)*.
    spot_prop_df : pd.DataFrame
        *n_spots × n_types* matrix with spot‑level cell‑type fractions.
    global_prop : pd.Series
        Global slide‑wide cell‑type fractions (same ordering as columns).
    adata : AnnData
        Visium AnnData object (full‑resolution coordinates required).
    adata_name : str
        Key under ``adata.uns['spatial']`` holding the scalefactors.
    seg_dict : Dict
        Cell segmentation dict – nuclei centroids accessed via
        ``seg_dict['nuc'][str(cell_id)]['centroid']``.
    beta : float, default 0
        Interpolation between adjusted (0 → full adj., 1 → keep original).
    batch_size : int, default 256
        Minibatch size for GPU throughput.
    eps : float, default 1e‑6
        Numerical stability constant.
    device : Union[str, torch.device], optional
        Computation device.  Defaults to "cuda" if available.
    """

    def __init__(
        self,
        cell_prob_df: pd.DataFrame,
        spot_dict: Dict[str, List[str]],
        spot_prop_df: pd.DataFrame,
        global_prop: pd.Series,
        adata: AnnData,
        adata_name: str,
        seg_dict: Dict,
        beta: float = 0.0,
        batch_size: int = 256,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if list(cell_prob_df.columns) != list(spot_prop_df.columns):
            raise ValueError("cell_prob_df and spot_prop_df must share identical columns order")

        # Basic attributes --------------------------------------------------
        self.beta_global = float(beta)
        self.batch_size = batch_size
        self.eps = eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Store dataframes --------------------------------------------------
        self.cell_prob_df = cell_prob_df.copy()
        self.spot_prop_df = spot_prop_df.copy()
        self.global_prop = global_prop.copy()

        # Index maps --------------------------------------------------------
        self.spot_to_cells = {k: [str(c) for c in v] for k, v in spot_dict.items()}
        self.cell_to_spot = revert_dict(self.spot_to_cells)

        self.spot_coords = adata.obsm["spatial"].astype("float64")  # (n_spots, 2)
        self.spot_ids_order = adata.obs_names.to_numpy()
        self.spot_id_to_idx = {sid: i for i, sid in enumerate(self.spot_ids_order)}

        # Geometry parameters ----------------------------------------------
        self.spot_diameter = float(adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"])
        self.R = 2.0 * self.spot_diameter
        self.kdtree = cKDTree(self.spot_coords)

        # Torch global prior -----------------------------------------------
        self.p_c = torch.tensor(global_prop.values, dtype=torch.float32, device=self.device).clamp(min=self.eps)

        # Build tensors needed for adjustment ------------------------------
        self.adjustable_cells, self.unadjustable_cells, p_local_np, beta_np = self._prepare_local_vectors(seg_dict)

        self.p_cell = torch.tensor(
            self.cell_prob_df.loc[self.adjustable_cells].values, dtype=torch.float32, device=self.device
        )
        self.p_local = torch.tensor(p_local_np, dtype=torch.float32, device=self.device)
        self.beta_cell = torch.tensor(beta_np, dtype=torch.float32, device=self.device)

        # Save tensors to .pt files
        torch.save(self.p_local, "p_local.pt")
        torch.save(self.beta_cell, "beta_cell.pt")

        with open("adjus.pkl", "wb") as f:
            pickle.dump(self.adjustable_cells, f)

        with open("non_adjust.pkl", "wb") as f:
            pickle.dump(self.unadjustable_cells, f)

    def _prepare_local_vectors(self, seg_dict: Dict) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        _ = self.spot_prop_df.shape[1]
        spot_prop_np = self.spot_prop_df.to_numpy(dtype=np.float64)

        def svec(idx: int) -> np.ndarray:
            return spot_prop_np[idx]

        adjustable, unadjustable = [], []
        local_vecs: List[np.ndarray] = []
        beta_list: List[float] = []

        for cell in self.cell_prob_df.index:
            cid = str(cell)

            # ---------------- inside a spot ----------------
            if cid in self.cell_to_spot:
                spot_id = self.cell_to_spot[cid]
                s_idx = self.spot_id_to_idx[spot_id]
                local_vecs.append(svec(s_idx))
                beta_list.append(self.beta_global)
                adjustable.append(cell)
                continue

            # ---------------- outside: need neighbours -----
            try:
                centroid = np.asarray(seg_dict["nuc"][cid]["centroid"], dtype=np.float64)
                assert centroid.shape == (2,)
            except Exception:
                unadjustable.append(cell)
                continue

            dists, idxs = self.kdtree.query(centroid, k=3, distance_upper_bound=self.R)
            neighbours = [
                (int(i), float(d))
                for i, d in zip(np.atleast_1d(idxs), np.atleast_1d(dists))
                if i != self.kdtree.n and math.isfinite(d)
            ]

            if not neighbours:
                unadjustable.append(cell)
                continue

            # ------- exactly one neighbour (distance‑aware β) ------
            if len(neighbours) == 1:
                idx0, d0 = neighbours[0]
                w = max((self.R - d0) / self.R, 0.0)  # 0‥1
                local_vecs.append(svec(idx0))
                beta_list.append(1.0 - w)  # far ⇒ β≈1, close ⇒ β≈0
                adjustable.append(cell)
                continue

            # ------- two or three neighbours -----------------------
            R = self.R
            weights = np.array([(R - d) / R for _, d in neighbours], dtype=np.float64)
            weights = np.clip(weights, 0.0, 1.0)
            vecs = np.stack([svec(i) for i, _ in neighbours], axis=0)
            weighted = (weights[:, None] * vecs).sum(axis=0)
            norm = weights.sum()
            if norm <= self.eps:
                unadjustable.append(cell)
                continue
            local_vecs.append(weighted / norm)
            beta_list.append(self.beta_global)
            adjustable.append(cell)

        return (
            adjustable,
            unadjustable,
            np.stack(local_vecs, axis=0),  # (N_adj, n_types)
            np.array(beta_list, dtype=np.float32),  # (N_adj,)
        )

    def _alpha(self, p_cell: torch.Tensor, p_local: torch.Tensor) -> torch.Tensor:
        sim = torch.sum(p_cell * (p_local / self.p_c)).clamp(min=self.eps)
        return 1.0 / sim

    def adjust(self) -> pd.DataFrame:
        dataset = CellProbDataset(self.p_cell, self.p_local, self.beta_cell)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        adj = torch.zeros_like(self.p_cell)

        for p_cell_batch, p_loc_batch, beta_batch, idx in tqdm(loader, desc="Adjusting"):
            # alpha per cell (vectorised)
            sim = torch.sum(p_cell_batch * (p_loc_batch / self.p_c), dim=1).clamp(min=self.eps)
            alpha = (1.0 / sim).clamp(max=1e6)

            p_adj = p_cell_batch * alpha.unsqueeze(1) * (p_loc_batch / self.p_c)
            p_final = (1.0 - beta_batch.unsqueeze(1)) * p_adj + beta_batch.unsqueeze(1) * p_cell_batch
            adj[idx] = p_final

        # Merge with untouched cells --------------------------------------
        adj_df = pd.DataFrame(adj.cpu().numpy(), index=self.adjustable_cells, columns=self.cell_prob_df.columns)
        untouched_df = self.cell_prob_df.loc[self.unadjustable_cells]
        out = pd.concat([adj_df, untouched_df])
        return out.loc[self.cell_prob_df.index]
