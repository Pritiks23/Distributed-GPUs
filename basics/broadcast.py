
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
