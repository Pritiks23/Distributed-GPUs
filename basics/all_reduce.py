
"""
all_reduce.py

Purpose:
Demonstrates gradient synchronization using all_reduce.

Main Technical Components:
- collective communication
- dist.all_reduce()
- distributed gradient averaging
- ReduceOp.SUM

Functionality:
- Each process creates a different tensor.
- all_reduce sums tensors across all processes.
- Gradients are averaged to simulate DDP training.
"""

import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12357"


def all_reduce_demo(rank, world_size):

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    gradient = t.tensor([float(rank + 1)])

    print(f"[Rank {rank}] Local gradient: {gradient}")

    dist.all_reduce(
        gradient,
        op=dist.ReduceOp.SUM
    )

    print(f"[Rank {rank}] Summed gradient: {gradient}")

    gradient /= world_size

    print(f"[Rank {rank}] Averaged gradient: {gradient}")

    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 2

    mp.spawn(
        all_reduce_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

"""
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (ALL_REDUCE IN DISTRIBUTED TRAINING)
====================================================================

CORE IDEA:
-----------
This file demonstrates the most important operation in distributed
deep learning: ALL_REDUCE.

While BROADCAST synchronizes INITIAL model weights,
ALL_REDUCE synchronizes LEARNING SIGNALS (gradients) during training.

This is the core mechanism that makes multi-GPU training actually work.

--------------------------------------------------------------------
THE PROBLEM BEING SOLVED
--------------------------------------------------------------------

In distributed training, each GPU processes a different subset of data:

    GPU 0 → batch A
    GPU 1 → batch B

Each GPU computes gradients independently:

    GPU 0 gradient = g1
    GPU 1 gradient = g2

But if we applied these separately, each model would drift apart.

That breaks training consistency.

So we need:

    A way to combine gradients from ALL GPUs
    and ensure EVERY GPU uses the SAME update.

--------------------------------------------------------------------
THE SOLUTION: ALL_REDUCE
--------------------------------------------------------------------

ALL_REDUCE does exactly this:

    1. Collect values from all processes
    2. Combine them using a reduction operation (SUM here)
    3. Distribute the result back to ALL processes

Code:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

Meaning:
    "Take all gradients across GPUs, sum them, and give the result to everyone"

--------------------------------------------------------------------
WHAT HAPPENS STEP BY STEP
--------------------------------------------------------------------

Before all_reduce:

    Rank 0 → gradient = 1
    Rank 1 → gradient = 2

After SUM reduction:

    Both ranks receive:
        gradient = 3

Then we manually average:

    gradient = 3 / world_size = 1.5

Now both GPUs apply the SAME update.

--------------------------------------------------------------------
WHY THIS MATTERS IN DEEP LEARNING
--------------------------------------------------------------------

Each GPU computes gradients on different data,
but we want:

    One consistent global gradient update

So ALL_REDUCE creates the equivalent of:

    "What would happen if we trained on the full batch at once?"

But distributed across multiple GPUs.

--------------------------------------------------------------------
KEY INSIGHT
--------------------------------------------------------------------

ALL_REDUCE is what makes distributed training equivalent to:

    single GPU training with a large batch size

Even though computation is split, the math is unified.

--------------------------------------------------------------------
WHY WE USE SUM + DIVIDE
--------------------------------------------------------------------

We use:

    SUM across GPUs
    then divide by world_size

This produces:

    mean gradient

This ensures training is:
- stable
- scale-invariant
- consistent across number of GPUs

--------------------------------------------------------------------
WHAT dist.ReduceOp.SUM ACTUALLY MEANS
--------------------------------------------------------------------

It defines the aggregation rule:

    SUM → add all tensors together
    (other options: MIN, MAX, PRODUCT, etc.)

But in deep learning:

    SUM + AVERAGING = standard SGD behavior

--------------------------------------------------------------------
HOW THIS CONNECTS TO DDP
--------------------------------------------------------------------

In PyTorch DistributedDataParallel (DDP):

    You NEVER call all_reduce manually.

Instead:
    PyTorch automatically performs all_reduce
    on gradients after loss.backward()

So internally DDP does:

    for each parameter:
        all_reduce(param.grad)

This file is exposing that hidden mechanism.

--------------------------------------------------------------------
RELATIONSHIP TO BROADCAST FILE
--------------------------------------------------------------------

Broadcast:
    ONE → MANY (initialize model)

All_reduce:
    MANY → ONE → MANY (synchronize learning)

Together they form:

    BROADCAST → consistent start
    ALL_REDUCE → consistent learning

--------------------------------------------------------------------
INTUITION
--------------------------------------------------------------------

Think of training like group decision making:

    Each GPU = person giving an opinion (gradient)

Without communication:
    everyone updates differently → chaos

With ALL_REDUCE:
    everyone agrees on the same final decision

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Distributed training loop:

    1. each GPU computes gradients locally
    2. gradients are ALL_REDUCE'd (synchronized)
    3. optimizer step is identical everywhere

Result:
    multiple GPUs behave like ONE large GPU

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file demonstrates how distributed training synchronizes learning
by using all_reduce to combine gradients across all GPUs so every
process applies the same update, making multi-GPU training equivalent
to large-batch single-GPU training.
--------------------------------------------------------------------
"""
