from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.models import resnet18

from deconvplugin.model.base_cell_classifier import BaseCellClassifier


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

        return torch.flatten(self.layers(x), 1)


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
        self.size_edge = 64

        if self.model_name == "convnet":
            # Use custom ConvNet
            self.backbone = ConvNet()

            self.fc_layers = nn.Sequential(
                nn.Linear(
                    int(32 * (self.size_edge / 8) ** 2), int(8 * (self.size_edge / 8) ** 2)
                ),  # 32 * (a/8) * (a/8), 8 * (a/8) * (a/8)
                nn.ReLU(),
            )

            self.classifier = nn.Linear(int(8 * (self.size_edge / 8) ** 2), self.num_classes)

        elif self.model_name == "quick":
            self.backbone = nn.Sequential()
            input_dim = 2048
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.backbone.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
                self.backbone.add_module(f"relu_{i}", nn.ReLU())
                # self.backbone.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
                input_dim = hidden_dim

            self.backbone.add_module("final", nn.Linear(input_dim, num_classes))

        elif self.model_name == "resnet18":

            resnet = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            input_dim = 512
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.backbone.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
                self.backbone.add_module(f"relu_{i}", nn.ReLU())
                # self.backbone.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
                input_dim = hidden_dim

            self.backbone.add_module("final", nn.Linear(input_dim, num_classes))

        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=self.num_classes)

            # for param in self.backbone.parameters():
            #     param.requires_grad = False

            # self.fc_layers = nn.Sequential()
            # input_dim = self.backbone(torch.randn(1, 3, 64, 64)).shape[1]
            # for i, hidden_dim in enumerate(self.hidden_dims):
            #     self.fc_layers.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
            #     self.fc_layers.add_module(f"relu_{i}", nn.ReLU())
            #     input_dim = hidden_dim

            # # Final classification layer
            # self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        # if self.model_name == "convnet":
        #     x = torch.flatten(x, 1)
        # features = self.fc_layers(features)

        return F.softmax(features, dim=1)
