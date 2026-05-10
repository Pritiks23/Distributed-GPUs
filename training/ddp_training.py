
"""
ddp_training.py

Purpose:
Demonstrates full Distributed Data Parallel training.

Main Technical Components:
- DistributedDataParallel (DDP)
- NCCL backend
- DistributedSampler
- gradient synchronization
- multi-process GPU training

Functionality:
- Launches one process per GPU.
- Synchronizes model weights.
- Splits batches across GPUs.
- Computes gradients independently.
- Synchronizes gradients with all_reduce internally.
- Updates identical model copies across GPUs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from cifar10_resnet import CIFAR10ResNet


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12358"


BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001


transform = transforms.Compose([
    transforms.ToTensor()
])


def setup(rank, world_size):

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


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
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    return dataloader

def train(rank, world_size):

    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    model = CIFAR10ResNet().to(device)

    # Synchronize initial weights
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    ddp_model = DDP(model, device_ids=[rank])

    dataloader = create_dataloader(rank, world_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        ddp_model.parameters(),
        lr=LEARNING_RATE
    )

    ddp_model.train()

    for epoch in range(EPOCHS):

        dataloader.sampler.set_epoch(epoch)

        total_loss = 0

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = ddp_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"[Rank {rank}] Epoch {epoch+1} Loss: {total_loss:.4f}")

    cleanup()


if __name__ == "__main__":

    assert torch.cuda.device_count() >= 2, "Requires at least 2 GPUs"

    world_size = 2

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
