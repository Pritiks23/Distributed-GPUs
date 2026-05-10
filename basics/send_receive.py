
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
