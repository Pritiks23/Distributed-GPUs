
"""
send_receive.py

Purpose:
Demonstrates the most basic form of distributed communication.

Main Technical Components:
- torch.distributed
- process ranks
- point-to-point communication
- dist.send()
- dist.recv()

Functionality:
- Rank 0 creates and sends a tensor.
- Rank 1 receives and overwrites its tensor.
- Demonstrates direct communication between processes.
"""

import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


def send_receive(rank, world_size):

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        tensor = t.tensor([999.0])

        print(f"[Rank {rank}] Sending: {tensor}")

        dist.send(tensor=tensor, dst=1)

    elif rank == 1:
        tensor = t.tensor([0.0])

        print(f"[Rank {rank}] Before recv: {tensor}")

        dist.recv(tensor=tensor, src=0)

        print(f"[Rank {rank}] After recv: {tensor}")

    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 2

    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    """
====================================================================
BIG PICTURE: WHY THIS FILE EXISTS (SEND / RECV IN DISTRIBUTED TRAINING)
====================================================================

CORE IDEA:
-----------
This file demonstrates the most fundamental building block of all
distributed systems: direct communication between processes.

Before you can understand broadcast or all_reduce, you need to understand:
    how two independent processes talk to each other directly.

--------------------------------------------------------------------
THE PROBLEM BEING SOLVED
--------------------------------------------------------------------

In distributed training, each GPU runs independently:

    Rank 0 → separate process
    Rank 1 → separate process

There is NO shared memory between them.

So if Rank 0 has data that Rank 1 needs:
    it cannot just "access it"

We need explicit communication.

--------------------------------------------------------------------
THE SOLUTION: SEND / RECV
--------------------------------------------------------------------

This is the lowest-level communication primitive:

    dist.send() → push data to another process
    dist.recv() → block and wait for data from another process

Unlike broadcast/all_reduce:
    this is ONE-TO-ONE communication

--------------------------------------------------------------------
WHAT THE CODE IS DOING
--------------------------------------------------------------------

Rank 0:
    creates tensor [999]
    sends it to Rank 1

Rank 1:
    starts with tensor [0]
    waits for message
    receives tensor from Rank 0
    overwrites its value

--------------------------------------------------------------------
KEY BEHAVIOR: BLOCKING COMMUNICATION
--------------------------------------------------------------------

recv is BLOCKING:

    Rank 1 will WAIT until Rank 0 sends data

This is important because:
- prevents race conditions
- enforces synchronization between processes

--------------------------------------------------------------------
WHAT YOU LEARN FROM THIS FILE
--------------------------------------------------------------------

This file teaches:

1. Processes are fully independent
2. There is no shared memory
3. Communication must be explicit
4. Data transfer is directional (A → B)

--------------------------------------------------------------------
WHY THIS MATTERS FOR DEEP LEARNING
--------------------------------------------------------------------

Even though send/recv is not used directly in DDP training,
it forms the conceptual foundation for:

    broadcast (1 → N)
    gather (N → 1)
    all_reduce (N → N)

All higher-level operations are built on this idea:
    explicit message passing between processes

--------------------------------------------------------------------
INTUITION
--------------------------------------------------------------------

Think of it like two workers:

    Worker A writes a note
    Worker B cannot see it unless A physically hands it over

send/recv = physically handing over data

--------------------------------------------------------------------
FINAL MENTAL MODEL
--------------------------------------------------------------------

Distributed systems = collection of isolated workers

To coordinate them:
    you must explicitly send messages between them

send/recv is the simplest version of that idea.

--------------------------------------------------------------------
ONE-SENTENCE SUMMARY
--------------------------------------------------------------------

This file demonstrates the most basic distributed communication pattern
where one process directly sends data to another process using explicit
message passing, forming the foundation for all higher-level distributed
training operations.
--------------------------------------------------------------------
"""
