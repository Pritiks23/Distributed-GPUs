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

```python id="c8q2mn"
"""
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (CIFAR10 RESNET MODEL)
====================================================================

CORE IDEA:
-----------
This file defines the neural network used in distributed training.

It is intentionally simple so the focus stays on:
    distributed systems concepts (not model complexity)

--------------------------------------------------------------------
WHY THIS MODEL EXISTS
--------------------------------------------------------------------

In distributed training examples, we need:
- a real neural network
- a standard dataset task (CIFAR10)
- a model small enough to run on multiple GPUs quickly

So we use ResNet18 as a clean baseline.

--------------------------------------------------------------------
WHAT THIS MODEL IS DOING
--------------------------------------------------------------------

1. Start with ResNet18:
    a standard convolutional neural network architecture

2. Remove its original classifier head:
    because it was trained for ImageNet (1000 classes)

3. Replace it with CIFAR10 head:
    10 output classes instead of 1000

--------------------------------------------------------------------
KEY CHANGE IN THE CODE
--------------------------------------------------------------------

Original ResNet18:
    final layer → 1000 classes

This file changes it to:

    nn.Linear(512, 10)

Meaning:
    input features = 512
    output classes = 10 (CIFAR10)

--------------------------------------------------------------------
WHY THIS MATTERS FOR DISTRIBUTED TRAINING
--------------------------------------------------------------------

This model is used as the SAME architecture across all GPUs.

So when combined with:
- broadcast (initialization)
- all_reduce (gradients)

All GPUs are training:
    the exact same model structure

--------------------------------------------------------------------
INTUITION
--------------------------------------------------------------------

Think of this file as:

    "the shared brain architecture used by all workers"

Distributed training does NOT change the model here.

It only:
- copies it across GPUs
- synchronizes updates

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Model file = defines what is being trained

Distributed system = decides how it is trained

So this file is just:
    the shared function approximator used across all processes

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file defines a standard ResNet18 model adapted for CIFAR10 that
serves as the shared neural network architecture used across all GPUs
in distributed training.
--------------------------------------------------------------------
"""
```

