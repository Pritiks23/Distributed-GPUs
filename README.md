# Distributed Training Fundamentals

This repository demonstrates the core building blocks behind distributed deep learning using PyTorch Distributed.

The repo progresses from:

1. Basic point-to-point communication
2. Collective communication primitives
3. Distributed data loading
4. Distributed gradient synchronization
5. Full Distributed Data Parallel (DDP) training

Main concepts covered:
- send / recv
- broadcast
- all_reduce
- DistributedSampler
- gradient synchronization
- NCCL vs GLOO
- DistributedDataParallel (DDP)

## Install

```bash
pip install -r requirements.txt
