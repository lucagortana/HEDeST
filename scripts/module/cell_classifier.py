from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.loss import ROT
from module.loss import weighted_kl_divergence
from module.loss import weighted_l1_loss
from module.loss import weighted_l2_loss


class ResidualBlock(nn.Module):
    """from https://github.com/samcw/ResNet18-Pytorch/tree/master"""

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """from https://github.com/samcw/ResNet18-Pytorch/tree/master"""

    def __init__(self, ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.layers(x)


class CellClassifier(nn.Module):
    def __init__(self, size_edge, num_classes, mtype="resnet18", device=torch.device("cpu")):
        super(CellClassifier, self).__init__()

        self.size_edge = size_edge
        self.num_classes = num_classes
        self.mtype = mtype
        self.device = device

        if mtype == "resnet18":
            self.feature_extractor = ResNet(ResidualBlock)
        elif mtype == "convnet":
            self.feature_extractor = ConvNet()
        else:
            raise ValueError(f"Invalid model type: {mtype}. Use 'resnet18' or 'convnet'.")

        self.fc = nn.Sequential(
            nn.Linear(
                int(32 * (self.size_edge / 8) ** 2), int(8 * (self.size_edge / 8) ** 2)
            ),  # 32 * (a/8) * (a/8), 8 * (a/8) * (a/8)
            nn.ReLU(),
            nn.Linear(int(8 * (self.size_edge / 8) ** 2), self.num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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

        loss = alpha * max_prob_loss + (1 - alpha) * divergence_loss
        return loss, max_prob_loss, divergence_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            max_probs, predicted_classes = torch.max(self.forward(x), dim=1)

        return predicted_classes, max_probs
