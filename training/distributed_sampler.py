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

"""
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (DISTRIBUTED SAMPLING)
====================================================================

CORE IDEA:
-----------
This file shows how data is split across multiple GPUs so each one
processes a unique part of the dataset.

--------------------------------------------------------------------
WHAT PROBLEM IT SOLVES
--------------------------------------------------------------------

If every GPU reads the full dataset:

    GPU 0 → full CIFAR10
    GPU 1 → full CIFAR10

Then:
- compute is duplicated
- training is inefficient
- scaling gives no benefit

We need:

    each GPU sees different data

--------------------------------------------------------------------
THE SOLUTION: DISTRIBUTED SAMPLER
--------------------------------------------------------------------

DistributedSampler ensures:

    dataset is partitioned across processes

So:

    GPU 0 → chunk A of data
    GPU 1 → chunk B of data

No overlap.

--------------------------------------------------------------------
WHY SHUFFLING MATTERS
--------------------------------------------------------------------

shuffle=True ensures:
- data is randomly distributed each epoch
- avoids learning bias from fixed ordering

Each epoch:
    data split changes slightly

--------------------------------------------------------------------
WHAT DataLoader DOES HERE
--------------------------------------------------------------------

DataLoader is just:
- batching mechanism
- worker pipeline

But the sampler controls:
    WHICH samples each GPU sees

So sampler = distribution logic
DataLoader = batching engine

--------------------------------------------------------------------
KEY INTUITION
--------------------------------------------------------------------

Think of it like:

    splitting homework across students

Without sampler:
    everyone does the same work

With sampler:
    each student gets a different subset

--------------------------------------------------------------------
WHY THIS IS CRITICAL FOR DDP
--------------------------------------------------------------------

DDP assumes:

    each GPU processes different data

Otherwise:
- gradients become redundant
- training becomes inefficient
- scaling breaks

So DistributedSampler is what makes DDP actually work correctly.

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Distributed training pipeline:

    dataset
        ↓
    DistributedSampler (splits data)
        ↓
    DataLoader (creates batches)
        ↓
    GPU processes unique batches

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file shows how DistributedSampler splits a dataset across GPUs so
each process trains on a unique subset of data, enabling efficient and
non-redundant distributed training.
--------------------------------------------------------------------
"""
