from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.loss import ROT
from module.loss import weighted_kl_divergence
from module.loss import weighted_l1_loss
from module.loss import weighted_l2_loss


class CellClassifier(nn.Module):
    def __init__(self, size_edge, num_classes, dropout_prob=0.5, device=torch.device("cpu")):
        super(CellClassifier, self).__init__()

        self.size_edge = size_edge
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.device = device

        # Convolutional layers
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            # Size 8@axa
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Size 8@(a/2)x(a/2)
            nn.Conv2d(8, 16, 3, padding=1),
            # Size 16@(a/2)x(a/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Size 16@(a/4)x(a/4)
            nn.Conv2d(16, 32, 3, padding=1),
            # Size 32@(a/4)x(a/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Size 32@(a/8)x(a/8)
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(
                int(32 * (self.size_edge / 8) ** 2), int(8 * (self.size_edge / 8) ** 2)
            ),  # 32 * (a/8) * (a/8), 8 * (a/8) * (a/8)
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                int(8 * (self.size_edge / 8) ** 2), int((self.size_edge / 8) ** 2)
            ),  # 8 * (a/8) * (a/8), (a/8) * (a/8)
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(int((self.size_edge / 8) ** 2), self.num_classes)  # (a/8) * (a/8), num_classes

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)

    def loss_comb(
        self, outputs, true_proportions, weights=None, agg="proba", divergence="l1", reduction="mean", alpha=0
    ):
        """
        Computes the loss of the model.

        Args:
            outputs: The output of the model
            true_proportions: The true proportions of the classes
            weights: The weights of the classes. If None, the weights are set to 1.
            agg: The aggregation method of the output. Can be "proba" or "onehot".
            divergence: The divergence to use. Can be "l1", "l2", "kl" or "rot".
            reduction: The reduction method of the loss. Can be "mean" or "sum".
            alpha: The weight of the max probability loss. If not 0, we recommend a very low
                   value (~... for l1 and l2, ~... for kl). It also depends on 'weights'.
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

        return alpha * max_prob_loss + (1 - alpha) * divergence_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            max_probs, predicted_classes = torch.max(self.forward(x), dim=1)

        return predicted_classes, max_probs
