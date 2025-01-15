from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from deconvplugin.basics import revert_dict
from deconvplugin.config import TqdmToLogger

tqdm_out = TqdmToLogger(logger, level="INFO")


class CellProbDataset(Dataset):
    """
    Dataset for cell probabilities with corresponding spot IDs.

    Attributes:
        cell_prob_tensor: Tensor of cell probabilities.
        spot_ids: List of spot IDs corresponding to each cell.
    """

    def __init__(self, cell_prob_tensor: torch.Tensor, spot_ids: List[int]):
        """
        Initializes the CellProbDataset.

        Args:
            cell_prob_tensor: Tensor containing probabilities for each cell.
            spot_ids: List of spot IDs corresponding to each cell.
        """

        self.cell_prob_tensor = cell_prob_tensor
        self.spot_ids = spot_ids

    def __len__(self) -> int:
        """
        Returns the number of cells in the dataset.
        """

        return len(self.cell_prob_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Retrieves a cell's probabilities, spot ID, and index.

        Args:
            idx: Index of the cell.

        Returns:
            Tuple containing the cell probabilities, spot ID, and index.
        """

        return self.cell_prob_tensor[idx], self.spot_ids[idx], idx


class BayesianAdjustment:
    """
    Performs Bayesian adjustment on cell probabilities.

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
        batch_size: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes BayesianAdjustment.

        Args:
            cell_prob_df: DataFrame of cell probabilities.
            spot_dict: Dictionary mapping spot IDs to cell IDs.
            spot_prop_df: DataFrame of spot proportions.
            global_prop: Global proportions of cell types over the slide.
            batch_size: Batch size for processing.
            device: Device for computation.
        """

        self.inverse_spot_dict = revert_dict(spot_dict)

        self.cell_prob_df = cell_prob_df
        self.cell_prob_tensor = torch.tensor(self.cell_prob_df.values, dtype=torch.float32)
        self.spot_prop_tensor = torch.tensor(spot_prop_df.values, dtype=torch.float32)
        self.p_c = torch.tensor(global_prop.values, dtype=torch.float32)

        spot_index_map = {spot_id: idx for idx, spot_id in enumerate(spot_prop_df.index)}
        self.spot_ids = [spot_index_map[self.inverse_spot_dict[cell_id]] for cell_id in cell_prob_df.index]

        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.cell_prob_tensor = self.cell_prob_tensor.to(self.device)
        self.spot_prop_tensor = self.spot_prop_tensor.to(self.device)
        self.p_c = self.p_c.to(self.device)

    def _alpha(self, p_c_x: torch.Tensor, p_tilde_c: torch.Tensor) -> float:
        """
        Computes the alpha adjustment factor for a cell.

        Args:
            p_c_x: Predicted probability vector for a single cell.
            p_tilde_c: Proportion vector for the local spot.

        Returns:
            Alpha adjustment factor.
        """

        alpha_x = 1 / torch.sum(p_c_x * (p_tilde_c / self.p_c))
        return alpha_x

    def forward(self) -> pd.DataFrame:
        """
        Adjusts cell probabilities using Bayesian adjustment.

        Returns:
            DataFrame of adjusted cell probabilities.
        """

        dataset = CellProbDataset(self.cell_prob_tensor, self.spot_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        p_tilde_c_x = torch.zeros_like(self.cell_prob_tensor).to(self.device)

        for cell_probs, spot_ids, idx in tqdm(dataloader, file=tqdm_out, desc="Adjusting cell probabilities"):
            cell_probs = cell_probs.to(self.device)
            p_tilde_c_batch = self.spot_prop_tensor[spot_ids].to(self.device)

            alphas = torch.tensor([self._alpha(cell_probs[i], p_tilde_c_batch[i]) for i in range(len(cell_probs))])
            alphas = alphas.to(self.device)

            for i in range(len(cell_probs)):
                adjusted_probs = cell_probs[i] * alphas[i] * (p_tilde_c_batch[i] / self.p_c)
                p_tilde_c_x[idx[i]] = adjusted_probs

        p_tilde_c_x_df = pd.DataFrame(
            p_tilde_c_x.cpu().numpy(), index=self.cell_prob_df.index, columns=self.cell_prob_df.columns
        )

        return p_tilde_c_x_df
