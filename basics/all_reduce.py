
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
