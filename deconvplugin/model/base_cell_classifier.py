from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from deconvplugin.loss import ROT
from deconvplugin.loss import weighted_kl_divergence
from deconvplugin.loss import weighted_l1_loss
from deconvplugin.loss import weighted_l2_loss


class BaseCellClassifier(nn.Module, ABC):
    def __init__(
        self,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Base class for image classifiers.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def compute_loss(
        self,
        outputs: Tensor,
        true_proportions: Tensor,
        weights: Optional[torch.Tensor] = None,
        agg: str = "proba",
        divergence: str = "l1",
        reduction: str = "mean",
        alpha: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the combined loss of the model, which includes a divergence loss
        and an optional maximum probability loss.

        Args:
            outputs (torch.Tensor): The output probabilities from the model (num_cells x num_classes).
            true_proportions (torch.Tensor): The ground-truth class proportions (num_classes).
            weights (torch.Tensor, optional): Weights for each class. Default is uniform weights.
            agg (str): Method to aggregate cell predictions into spot proportions.
                    Options are "proba" (mean probabilities) or "onehot" (one-hot encoded class predictions).
            divergence (str): Type of divergence loss to use. Options are "l1", "l2", "kl", or "rot".
            reduction (str): Reduction method for the divergence loss. Options are "mean" or "sum".
            alpha (float): Weight for the max probability loss. Should be in the range [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Total loss combining divergence and max probability loss.
                - Maximum probability loss.
                - Divergence loss.
        """

        if weights is None:
            weights = torch.ones_like(true_proportions).to(self.device)

        if divergence == "rot":
            return ROT(outputs, true_proportions, alpha=alpha, weights=weights)

        max_prob_loss = -torch.mean(torch.log(outputs.max(dim=1)[0]))

        if agg == "proba":
            pred_proportions = outputs.mean(dim=0)
        elif agg == "onehot":
            predicted_classes = torch.argmax(outputs, dim=1)
            one_hot_preds = torch.nn.functional.one_hot(predicted_classes, num_classes=outputs.size(1))
            pred_proportions = one_hot_preds.float().sum(dim=0) / outputs.size(0)
        else:
            raise ValueError(f"Invalid aggregation method: {agg}. Use 'proba' or 'onehot'.")

        if divergence == "l1":
            divergence_loss = weighted_l1_loss(pred_proportions, true_proportions, weights, reduction=reduction)
        elif divergence == "l2":
            divergence_loss = weighted_l2_loss(pred_proportions, true_proportions, weights, reduction=reduction)
        elif divergence == "kl":
            divergence_loss = weighted_kl_divergence(pred_proportions, true_proportions, weights, reduction=reduction)
        else:
            raise ValueError(f"Invalid divergence type: {divergence}. Use 'l1', 'l2', 'kl', or 'rot'.")

        loss = alpha * max_prob_loss + (1 - alpha) * divergence_loss
        return loss, max_prob_loss, divergence_loss
