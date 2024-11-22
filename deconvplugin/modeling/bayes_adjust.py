from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from deconvplugin.basics import revert_dict


class CellProbDataset(Dataset):
    def __init__(self, cell_prob_tensor, spot_ids):
        self.cell_prob_tensor = cell_prob_tensor
        self.spot_ids = spot_ids

    def __len__(self):
        return len(self.cell_prob_tensor)

    def __getitem__(self, idx):
        return self.cell_prob_tensor[idx], self.spot_ids[idx], idx


class BayesianAdjustment:
    def __init__(self, cell_prob_df, spot_dict, spot_prop_df, global_prop, batch_size=256, device=None):
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

    def _alpha(self, p_c_x, p_tilde_c):
        alpha_x = 1 / torch.sum(p_c_x * (p_tilde_c / self.p_c))
        return alpha_x

    def forward(self):
        dataset = CellProbDataset(self.cell_prob_tensor, self.spot_ids)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        p_tilde_c_x = torch.zeros_like(self.cell_prob_tensor).to(self.device)

        for cell_probs, spot_ids, idx in tqdm(data_loader, desc="Adjusting cell probabilities"):
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
