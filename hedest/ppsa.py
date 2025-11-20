from __future__ import annotations

import json
import math
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
from tqdm import tqdm

from hedest.config import TqdmToLogger
from hedest.dataset import CellProbDataset
from hedest.dataset import CellProbDatasetNaive
from hedest.utils import revert_dict

tqdm_out = TqdmToLogger(logger, level="INFO")


class PPSAdjustment:
    r"""
    Prior Probability Shift adjustment of per‑cell type probabilities that leverages both
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
    json_path : str
        Path to the JSON file with segmentation data.
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
        json_path: str,
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

        # Index maps --------------------------------------------------------
        self.cell_to_spot = revert_dict(spot_dict)

        self.spot_coords = adata.obsm["spatial"].astype("float64")  # (n_spots, 2)
        self.spot_ids_order = adata.obs_names.to_numpy()
        self.spot_id_to_idx = {sid: i for i, sid in enumerate(self.spot_ids_order)}

        # Store dataframes --------------------------------------------------
        self.cell_prob_df = cell_prob_df.copy()
        self.spot_prop_df = spot_prop_df.copy()
        self.spot_prop_df = self.spot_prop_df.reindex(adata.obs_names, copy=False)
        self.global_prop = global_prop.copy()

        # Geometry parameters ----------------------------------------------
        self.spot_diameter = float(adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"])
        self.R = 2.0 * self.spot_diameter
        self.kdtree = cKDTree(self.spot_coords)

        # Torch global prior -----------------------------------------------
        self.p_c = torch.tensor(global_prop.values, dtype=torch.float32, device=self.device).clamp(min=self.eps)

        # Build tensors needed for adjustment ------------------------------
        with open(json_path) as json_file:
            seg_dict = json.load(json_file)
        self.adjustable_cells, self.unadjustable_cells, p_local_np, beta_np = self._prepare_local_vectors(seg_dict)

        self.p_cell = torch.tensor(
            self.cell_prob_df.loc[self.adjustable_cells].values, dtype=torch.float32, device=self.device
        )
        self.p_local = torch.tensor(p_local_np, dtype=torch.float32, device=self.device)
        self.beta_cell = torch.tensor(beta_np, dtype=torch.float32, device=self.device)

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


class PPSANaive:
    """
    Performs PPSA on cell probabilities.

    Attributes:
        cell_prob_df: DataFrame of cell probabilities.
        cell_prob_tensor: Tensor representation of cell probabilities.
        spot_prop_tensor: Tensor of spot-level proportions.
        p_c: Global cell type proportions as a tensor.
        spot_ids: List of indices mapping cells to their spots.
        batch_size: Batch size for processing.
        device: Device used for computation ("cuda" or "cpu").
    """

    def __init__(
        self,
        cell_prob_df: pd.DataFrame,
        spot_dict: Dict[str, List[str]],
        spot_prop_df: pd.DataFrame,
        global_prop: pd.Series,
        beta: float = 0.0,
        batch_size: int = 256,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes BayesianAdjustment.

        Args:
            cell_prob_df: DataFrame of cell probabilities.
            spot_dict: Dictionary mapping spot IDs to cell IDs.
            spot_prop_df: DataFrame of spot proportions.
            global_prop: Global proportions of cell types over the slide.
            beta: Hyperparameter to regularize adjustment.
            batch_size: Batch size for processing.
            eps: Small value to avoid division by zero.
            device: Device for computation.
        """

        self.inverse_spot_dict = revert_dict(spot_dict)

        self.cell_prob_df = cell_prob_df
        self.adjustable_cells = [cell_id for cell_id in cell_prob_df.index if cell_id in self.inverse_spot_dict]
        self.unadjustable_cells = [cell_id for cell_id in cell_prob_df.index if cell_id not in self.inverse_spot_dict]

        self.cell_prob_tensor = torch.tensor(cell_prob_df.loc[self.adjustable_cells].values, dtype=torch.float32)
        self.spot_prop_tensor = torch.tensor(spot_prop_df.values, dtype=torch.float32)
        self.p_c = torch.tensor(global_prop.values, dtype=torch.float32)

        spot_index_map = {spot_id: idx for idx, spot_id in enumerate(spot_prop_df.index)}
        self.spot_ids = [spot_index_map[self.inverse_spot_dict[cell_id]] for cell_id in self.adjustable_cells]

        self.beta = beta
        self.batch_size = batch_size
        self.eps = eps
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.cell_prob_tensor = self.cell_prob_tensor.to(self.device)
        self.spot_prop_tensor = self.spot_prop_tensor.to(self.device)
        self.p_c = self.p_c.to(self.device)
        self.p_c = self.p_c.clamp(min=self.eps)

    def _alpha(self, p_c_x: torch.Tensor, p_tilde_c: torch.Tensor) -> float:
        """
        Computes the alpha adjustment factor for a cell.

        Args:
            p_c_x: Predicted probability vector for a single cell.
            p_tilde_c: Proportion vector for the local spot.

        Returns:
            Alpha adjustment factor.
        """
        similarity = torch.sum(p_c_x * (p_tilde_c / self.p_c))
        similarity = torch.clamp(similarity, min=self.eps)
        alpha_x = 1 / similarity
        return alpha_x

    def adjust(self) -> pd.DataFrame:
        """
        Adjusts cell probabilities using Bayesian adjustment.

        Returns:
            DataFrame of adjusted cell probabilities.
        """

        dataset = CellProbDatasetNaive(self.cell_prob_tensor, self.spot_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        p_tilde_c_x = torch.zeros_like(self.cell_prob_tensor).to(self.device)

        for cell_probs, spot_ids, idx in tqdm(dataloader, file=tqdm_out, desc="Adjusting cell probabilities"):
            cell_probs = cell_probs.to(self.device)
            p_tilde_c_batch = self.spot_prop_tensor[spot_ids].to(self.device)

            alphas = torch.tensor([self._alpha(cell_probs[i], p_tilde_c_batch[i]) for i in range(len(cell_probs))])
            alphas = alphas.to(self.device)

            for i in range(len(cell_probs)):
                if torch.isinf(alphas[i]):
                    adjusted_probs = cell_probs[i]
                else:
                    adjusted_probs = cell_probs[i] * alphas[i] * (p_tilde_c_batch[i] / self.p_c)
                p_tilde_c_x[idx[i]] = (1 - self.beta) * adjusted_probs + self.beta * cell_probs[i]

        adjusted_df = pd.DataFrame(
            p_tilde_c_x.cpu().numpy(),
            index=self.adjustable_cells,
            columns=self.cell_prob_df.columns,
        )

        unadjusted_df = self.cell_prob_df.loc[self.unadjustable_cells]

        p_tilde_c_x_df = pd.concat([adjusted_df, unadjusted_df])
        p_tilde_c_x_df = p_tilde_c_x_df.loc[self.cell_prob_df.index]

        return p_tilde_c_x_df
