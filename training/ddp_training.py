
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


"""
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (FULL DDP TRAINING)
====================================================================

CORE IDEA:
-----------
This file shows REAL multi-GPU training using Distributed Data Parallel (DDP).

It connects everything:
- multiple GPUs
- model replication
- data splitting
- gradient synchronization
- coordinated training

--------------------------------------------------------------------
WHAT PROBLEM THIS SOLVES
--------------------------------------------------------------------

Training on 1 GPU is slow.

So we split work across GPUs:

    GPU 0 → batch A
    GPU 1 → batch B

But we still want:
    ONE shared model that learns together

So we need a system where:
- each GPU works independently
- but stays synchronized

--------------------------------------------------------------------
THE SOLUTION: DDP
--------------------------------------------------------------------

DistributedDataParallel (DDP) does 3 key things:

1. Copies model to every GPU
2. Splits data across GPUs
3. Automatically syncs gradients

So each GPU behaves like:
    an independent trainer
but results stay identical

--------------------------------------------------------------------
STEP-BY-STEP FLOW
--------------------------------------------------------------------

1. SETUP DISTRIBUTED ENVIRONMENT
   - assign rank to each GPU
   - connect processes via NCCL

2. MOVE MODEL TO GPU
   - each process gets its own GPU

3. BROADCAST INITIAL WEIGHTS
   - ensure all models start identical

4. SPLIT DATA (DistributedSampler)
   - each GPU sees different data

5. FORWARD PASS
   - each GPU processes its batch

6. BACKWARD PASS
   - each GPU computes gradients independently

7. DDP SYNC (ALL-REDUCE INTERNALLY)
   - gradients are averaged across GPUs

8. OPTIMIZER STEP
   - identical updates applied everywhere

--------------------------------------------------------------------
WHY DistributedSampler IS IMPORTANT
--------------------------------------------------------------------

Without it:

    every GPU sees same data → wasted compute

With it:

    each GPU sees different slice of dataset

So training becomes:
    truly parallel

--------------------------------------------------------------------
WHY WE BROADCAST WEIGHTS FIRST
--------------------------------------------------------------------

Even before DDP wraps the model:

    dist.broadcast(param.data, src=0)

This ensures:
    all GPUs start from same initialization

Otherwise:
    models diverge immediately

--------------------------------------------------------------------
WHAT DDP IS DOING BEHIND THE SCENES
--------------------------------------------------------------------

When you call:

    loss.backward()

DDP automatically runs:

    all_reduce(gradients)

So you never manually sync gradients here.

This is the key abstraction:
    PyTorch hides distributed communication inside DDP

--------------------------------------------------------------------
WHY NCCL BACKEND
--------------------------------------------------------------------

NCCL is used because:
- optimized for NVIDIA GPUs
- fast GPU-to-GPU communication
- low latency gradient syncing

--------------------------------------------------------------------
INTUITION
--------------------------------------------------------------------

Think of DDP like:

    multiple workers training independently
    but forced to agree after every step

So:

    independence during computation
    synchronization after gradients

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Distributed training loop:

    split data
    forward pass (parallel)
    backward pass (parallel)
    sync gradients (DDP)
    update weights (identical)

Result:

    multiple GPUs behave like ONE giant GPU

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file demonstrates full multi-GPU distributed training where each GPU
computes gradients independently on different data, and DistributedDataParallel
automatically synchronizes those gradients so all models stay identical.
--------------------------------------------------------------------
"""
