# Distributed Training Pipeline (PyTorch DDP Reference Implementation)

## Overview

This project implements a minimal, explicit Distributed Data Parallel (DDP) training system in PyTorch using NCCL/GLOO backends. It is designed to mirror internal behavior of PyTorch’s DDP abstraction by exposing communication and synchronization primitives directly.

---

## Architecture

The system is composed of the following layers:

### 1. Process Model
- One process per GPU
- Rank-based execution using `torch.multiprocessing`
- Explicit process group initialization via `torch.distributed`

### 2. Communication Layer
- `broadcast` → model weight synchronization
- `all_reduce` → gradient aggregation across ranks
- `send/recv` → point-to-point messaging primitives

### 3. Data Layer
- `DistributedSampler` partitions dataset across ranks
- Ensures non-overlapping mini-batches per GPU
- Enables deterministic distributed data loading

---

## Training Flow

### Initialization
- Each process initializes identical model architecture
- Rank 0 broadcasts initial parameters to all ranks

### Forward / Backward Pass
- Each GPU processes independent mini-batch
- Gradients computed locally per rank

### Synchronization
- Gradients are aggregated using `all_reduce` (sum reduction)
- Averaged gradients are applied identically across all models

### Optimization Step
- Each rank executes identical optimizer step
- Ensures parameter consistency across all GPUs

---

## Correctness Properties

This implementation guarantees:

- Synchronized model parameters across all ranks
- Equivalent optimization trajectory to single-device training
- Deterministic gradient aggregation under fixed sampling

---

## Notes

- NCCL backend is used for GPU-to-GPU communication
- GLOO backend is used for CPU-based simulation/testing
- This implementation exposes internal mechanisms typically abstracted by `DistributedDataParallel`

---

## Purpose

This system is intended as an educational reference implementation for understanding the internal mechanics of PyTorch DDP and NCCL-based distributed training systems.
