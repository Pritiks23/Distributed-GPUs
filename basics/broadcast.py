
"""
broadcast.py

Purpose:
Demonstrates broadcast collective communication.

Main Technical Components:
- collective communication
- dist.broadcast()
- synchronization across processes
- model weight initialization concepts

Functionality:
- Rank 0 creates a tensor.
- The tensor is copied to all other processes.
- Simulates how distributed training synchronizes model weights.
"""

import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12356"


def broadcast_demo(rank, world_size):

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        tensor = t.tensor([42.0])
    else:
        tensor = t.tensor([0.0])

    print(f"[Rank {rank}] BEFORE broadcast: {tensor}")

    dist.broadcast(tensor, src=0)

    print(f"[Rank {rank}] AFTER broadcast: {tensor}")

    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 2

    mp.spawn(
        broadcast_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


"""
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (BROADCAST IN DISTRIBUTED TRAINING)
====================================================================

CORE IDEA:
-----------
This file demonstrates how multiple independent processes (or GPUs)
stay synchronized during distributed deep learning using BROADCAST.

In distributed training, there is NO shared memory between GPUs.
Each GPU runs its own Python process with its own:
- model copy
- memory space
- random initialization

So without communication, every GPU diverges immediately.

--------------------------------------------------------------------
THE PROBLEM BEING SOLVED
--------------------------------------------------------------------

When training starts across multiple processes:

    Rank 0 model weights = random initialization A
    Rank 1 model weights = random initialization B

Even though the code is identical, randomness differs.

This causes:
- models to diverge immediately
- gradients to become inconsistent
- training instability or complete failure

So we need a way to FORCE consistency at startup.

--------------------------------------------------------------------
THE SOLUTION: BROADCAST
--------------------------------------------------------------------

Broadcast solves this fundamental problem:

    ONE process becomes the source of truth (rank 0)
    ALL other processes copy its tensor exactly

Code meaning:
    dist.broadcast(tensor, src=0)

This means:
    "Take rank 0's tensor and overwrite ALL other ranks with it"

--------------------------------------------------------------------
WHAT BROADCAST ACTUALLY DOES
--------------------------------------------------------------------

Before broadcast:

    Rank 0 → tensor([42.])
    Rank 1 → tensor([0.])

After broadcast:

    Rank 0 → tensor([42.])
    Rank 1 → tensor([42.])

Now all processes are synchronized.

--------------------------------------------------------------------
WHY RANK 0?
--------------------------------------------------------------------

Rank 0 is not special in capability — it is just:
- chosen as the initializer
- reference state holder
- source of truth

Any rank could technically be used, but rank 0 is convention.

--------------------------------------------------------------------
WHY WE INTENTIONALLY USE DIFFERENT VALUES
--------------------------------------------------------------------

We purposely set:

    Rank 0 → 42
    Rank 1 → 0

Even though unrealistic, this is done to clearly show:

    "Broadcast actually overwrites local memory"

If both started equal, the effect would be invisible.

--------------------------------------------------------------------
WHAT THIS TEACHES ABOUT DISTRIBUTED SYSTEMS
--------------------------------------------------------------------

This file is NOT just PyTorch code.

It teaches a core distributed systems principle:

    THERE IS NO SHARED MEMORY BETWEEN PROCESSES

Each process has:
- independent RAM
- independent execution
- independent model state

So synchronization must be EXPLICIT:
    - broadcast
    - send / recv
    - all_reduce

Nothing is automatic.

--------------------------------------------------------------------
WHY dist.destroy_process_group() EXISTS
--------------------------------------------------------------------

This line:

    dist.destroy_process_group()

means:
    "Shut down the communication network between processes"

Without it:
- sockets remain open
- processes may hang
- future runs can break

Think of it as cleaning up the distributed "network session".

--------------------------------------------------------------------
REAL ROLE IN DEEP LEARNING (IMPORTANT)
--------------------------------------------------------------------

In real distributed training systems (DDP):

Broadcast is used at initialization time:

    Step 1: Rank 0 initializes model weights
    Step 2: Broadcast weights to all GPUs
    Step 3: All GPUs start identical

This ensures:
    - identical starting model
    - reproducible training behavior
    - correct gradient synchronization later

--------------------------------------------------------------------
HOW THIS CONNECTS TO FULL DISTRIBUTED TRAINING
--------------------------------------------------------------------

Broadcast is only the FIRST building block.

Full training also uses:

1. BROADCAST
   → synchronize initial weights

2. ALL_REDUCE
   → synchronize gradients during training

3. DISTRIBUTED SAMPLER
   → split dataset across GPUs

Together they form:

    Distributed Data Parallel (DDP)

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Think of distributed training like this:

    Multiple workers (GPUs)
        |
        | each computes independently
        |
        v
    COMMUNICATION LAYER (broadcast / all_reduce)
        |
        v
    SYNCHRONIZED MODEL STATE

Without communication → chaos
With communication → parallel learning works

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file demonstrates how distributed training ensures all processes
begin with identical model parameters by using broadcast from a
single source process (rank 0), forming the foundation of synchronized
multi-GPU learning systems.
--------------------------------------------------------------------
"""
