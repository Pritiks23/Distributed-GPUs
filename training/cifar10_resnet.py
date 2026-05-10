"""
cifar10_resnet.py

Purpose:
Defines a simple ResNet model for CIFAR10 classification.

Main Technical Components:
- torchvision.models
- ResNet18
- transfer learning structure
- classification head replacement

Functionality:
- Creates a ResNet18 architecture.
- Modifies final classification layer for CIFAR10.
- Used in distributed training examples.
"""

import torch.nn as nn
from torchvision.models import resnet18


class CIFAR10ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)

        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)
