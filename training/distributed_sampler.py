"""
distributed_sampler.py

Purpose:
Demonstrates how datasets are split across GPUs/processes.

Main Technical Components:
- DistributedSampler
- DataLoader
- distributed batching
- shuffled partitioning

Functionality:
- Each process receives a different subset of the dataset.
- Prevents GPUs from processing duplicate samples.
- Core component of efficient distributed training.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


transform = transforms.Compose([
    transforms.ToTensor()
])


def create_dataloader(rank, world_size):

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    return dataloader
