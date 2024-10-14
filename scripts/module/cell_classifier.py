from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def loss_comb(self, outputs, true_proportions, agg="mean", alpha=0.5):

        max_prob_loss = -torch.mean(torch.log(outputs.max(dim=1)[0]))
        if agg == "mean":
            pred_proportions = outputs.mean(dim=0)
        elif agg == "onehot":
            predicted_classes = torch.argmax(outputs, dim=1)
            one_hot_preds = torch.nn.functional.one_hot(predicted_classes, num_classes=outputs.size(1))
            pred_proportions = one_hot_preds.float().sum(dim=0) / outputs.size(0)

        divergence_loss = F.mse_loss(pred_proportions, true_proportions) * 1e2

        # Combined loss
        loss = alpha * max_prob_loss + (1 - alpha) * divergence_loss

        return loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            max_probs, predicted_classes = torch.max(self.forward(x), dim=1)

        return predicted_classes, max_probs


# class CellClassifier(nn.Module):
#     def __init__(self, num_classes=5, dropout_prob=0.5):
#         super(CellClassifier, self).__init__()

#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 8 * 8, 256)
#         self.bn_fc1 = nn.BatchNorm1d(256)

#         self.fc2 = nn.Linear(256, self.num_classes)

#         # Dropout layers
#         self.dropout_conv = nn.Dropout2d(p=self.dropout_prob)
#         self.dropout_fc = nn.Dropout(p=self.dropout_prob)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
#         x = self.dropout_conv(x)

#         x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
#         x = self.dropout_conv(x)

#         x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
#         x = self.dropout_conv(x)

#         x = x.view(x.size(0), -1)  # Flatten the tensor

#         x = F.relu(self.bn_fc1(self.fc1(x)))
#         x = self.dropout_fc(x)

#         x = self.fc2(x)

#         return F.softmax(x, dim=1)

#     def loss_comb(self, outputs, true_proportions, alpha=0.5):

#         max_prob_loss = - torch.mean(torch.log(outputs.max(dim=1)[0]))
#         pred_proportions = outputs.mean(dim=0)
#         divergence_loss = F.mse_loss(pred_proportions, true_proportions)

#         # Combined loss
#         loss = alpha * max_prob_loss + (1 - alpha) * divergence_loss

#         return loss
