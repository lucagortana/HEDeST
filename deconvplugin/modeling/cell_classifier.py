from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from deconvplugin.modeling.loss import ROT
from deconvplugin.modeling.loss import weighted_kl_divergence
from deconvplugin.modeling.loss import weighted_l1_loss
from deconvplugin.modeling.loss import weighted_l2_loss


class ResidualBlock(nn.Module):
    """
    A residual block for ResNet. Adapted from https://github.com/samcw/ResNet18-Pytorch/tree/master.

    Attributes:
        left: A sequential layer containing two convolutional layers with BatchNorm and ReLU activation.
        shortcut: A shortcut connection layer that applies identity or projection.
    """

    def __init__(self, inchannel: int, outchannel: int, stride: int = 1) -> None:
        """
        Initializes the ResidualBlock.

        Args:
            inchannel (int): Number of input channels.
            outchannel (int): Number of output channels.
            stride (int): Stride for the convolutional layer. Default is 1.
        """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """

        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    A ResNet model. Adapted from https://github.com/samcw/ResNet18-Pytorch/tree/master.

    Attributes:
        inchannel (int): Number of input channels.
        conv1: Initial convolutional layer.
        layer1: First residual layer.
        layer2: Second residual layer.
        layer3: Third residual layer.
        layer4: Fourth residual layer.
    """

    def __init__(self, ResidualBlock: Type[ResidualBlock]) -> None:
        """
        Initializes the ResNet model.

        Args:
            ResidualBlock (Type[ResidualBlock]): Residual block to use in the layers.
        """

        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

    def make_layer(self, block: Type[ResidualBlock], channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Creates a residual layer with multiple blocks.

        Args:
            block (Type[ResidualBlock]): Residual block class.
            channels (int): Number of channels for the layer.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Sequential layer containing the blocks.
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ResNet model.
        """

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


class ConvNet(nn.Module):
    """
    A simple Convolutional Neural Network.

    Attributes:
        layers: Sequential layer containing convolutional, batch normalization, ReLU, and pooling layers.
    """

    def __init__(self):
        """
        Initializes the ConvNet model.
        """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ConvNet layers.
        """

        return self.layers(x)


class CellClassifier(nn.Module):
    """
    A classifier for cell images using either ResNet or ConvNet as the backbone.

    Attributes:
        edge_size: The edge size of input images.
        num_classes: Number of output classes.
        mtype: The type of model backbone ("resnet18" or "convnet").
        device: The device on which to run the model.
        feature_extractor: The chosen backbone model.
        fc: Fully connected layers for classification.
    """

    def __init__(
        self,
        edge_size: int,
        num_classes: int,
        mtype: str = "resnet18",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initializes the CellClassifier.

        Args:
            edge_size (int): The edge size of the input image.
            num_classes (int): Number of classes for classification.
            mtype (str): Type of the model backbone. Default is "resnet18".
            device (torch.device): Device to run the model. Default is CPU.
        """

        super(CellClassifier, self).__init__()

        self.edge_size = edge_size
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
                int(32 * (self.edge_size / 8) ** 2), int(8 * (self.edge_size / 8) ** 2)
            ),  # 32 * (a/8) * (a/8), 8 * (a/8) * (a/8)
            nn.ReLU(),
            nn.Linear(int(8 * (self.edge_size / 8) ** 2), self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class probabilities.
        """

        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def loss_comb(
        self,
        outputs: torch.Tensor,
        true_proportions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        agg: str = "proba",
        divergence: str = "l1",
        reduction: str = "mean",
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the class labels and associated probabilities for input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Predicted class indices for each input (batch_size).
                - Associated probabilities for the predicted classes (batch_size).
        """

        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            max_probs, predicted_classes = torch.max(self.forward(x), dim=1)

        return predicted_classes, max_probs
