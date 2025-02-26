from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from deconvplugin.model.base_cell_classifier import BaseCellClassifier


class CellClassifier(BaseCellClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        hidden_dims: list = [512, 256],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Image classifier using a pretrained model from timm or a custom ConvNet.

        Args:
            model_name (str): Name of the pretrained model (e.g., "resnet18", "hf-hub:imagenet/h_optimus-0",
                              or "convnet").
            num_classes (int): Number of output classes.
            hidden_dims (list): List of hidden dimensions for the fully connected layers.
            device (torch.device): Device to run the model on.
        """
        super().__init__(num_classes, device)
        self.model_name = model_name
        self.hidden_dims = hidden_dims

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_output_dim = self.backbone(torch.randn(1, 3, 64, 64)).shape[1]

        # Add fully connected layers for transfer learning
        self.fc_layers = nn.Sequential()
        input_dim = backbone_output_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.fc_layers.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
            self.fc_layers.add_module(f"relu_{i}", nn.ReLU())
            input_dim = hidden_dim

        # Final classification layer
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        features = self.fc_layers(features)

        return F.softmax(self.classifier(features), dim=1)


# if model_name == "convnet":
#     # Use custom ConvNet
#     self.backbone = ConvNet()
#     backbone_output_dim = int(32 * (edge_size / 8) ** 2)  # Output size of ConvNet

# class ConvNet(nn.Module):
#     """
#     A simple Convolutional Neural Network.

#     Attributes:
#         layers: Sequential layer containing convolutional, batch normalization, ReLU, and pooling layers.
#     """

#     def __init__(self):
#         """
#         Initializes the ConvNet model.
#         """

#         super(ConvNet, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 8, 3, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(8, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Defines the forward pass.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor after applying the ConvNet layers.
#         """

#         return torch.flatten(self.layers(x), 1)
