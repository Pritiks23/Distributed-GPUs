
# Distributed Training Fundamentals (PyTorch DDP Deep Dive)
<img width="542" height="804" alt="Distributed" src="https://github.com/user-attachments/assets/49411f1a-c6b5-4112-8aaa-620066927d0e" />



## Overview

This repository is a step-by-step breakdown of how distributed deep learning actually works under the hood using PyTorch.

It starts from the lowest-level communication primitives (`send/recv`) and builds up to full-scale **Distributed Data Parallel (DDP)** training with multiple GPUs.

The goal is not just to run distributed training, but to understand:

> how independent processes coordinate to behave like a single training system.

---

## Core Idea of Distributed Training

In single-GPU training:
- one process
- one model
- one dataset

In distributed training:
- multiple processes (one per GPU)
- each process runs the same model code
- each processes different data
- communication keeps everything synchronized

The system only works because of **explicit communication primitives**.

---

## Key Insight (Most Important Concept)

Distributed training is NOT:
> multiple models learning independently

It IS:
> multiple workers computing independently but staying synchronized through communication

So:

- computation is parallel
- learning is unified

---

## Communication Primitives (Foundation Layer)

### 1. `send / recv` (Point-to-point communication)

- simplest form of distributed communication
- one process sends data to another
- fully explicit, blocking communication

**Conceptual meaning:**
> one worker directly passes memory to another worker

This teaches:
- no shared memory exists between processes
- all coordination is explicit

---
<img width="564" height="437" alt="Broadcast" src="https://github.com/user-attachments/assets/d901bdea-71a1-4164-a36a-4f358e5be194" />

### 2. `broadcast` (1 → N synchronization)

- rank 0 becomes source of truth
- all other processes copy its tensor

**Used for:**
- model initialization
- parameter synchronization before training

**Conceptual meaning:**
> ensure all workers start from identical state

---
<img width="564" height="479" alt="Allreduce" src="https://github.com/user-attachments/assets/c37e9e02-4a94-4e4a-98b2-db94f807406d" />

### 3. `all_reduce` (N → N gradient synchronization)

- all processes compute local gradients
- gradients are summed (or averaged)
- result is distributed back to all processes

**Used for:**
- gradient synchronization in training

**Conceptual meaning:**
> convert multiple independent gradients into one shared update signal

This is the core operation behind DDP.

---

## Data Parallelism (`DistributedSampler`)
<img width="564" height="531" alt="Dataparallelism" src="https://github.com/user-attachments/assets/9a05d50e-c931-4a26-a56d-953241ef0ed7" />

Each GPU must see different data.

Without sampling:
- every GPU processes the full dataset → wasted compute

With `DistributedSampler`:
- dataset is partitioned across GPUs
- each GPU sees a unique subset

**Conceptual meaning:**
> parallelism comes from splitting data, not splitting the model

---

## Full Training System (DDP)

### What DDP does

DDP combines everything:

1. **Model replication**
   - same model on every GPU

2. **Data partitioning**
   - each GPU gets different mini-batches

3. **Independent computation**
   - forward + backward pass per GPU

4. **Automatic synchronization**
   - gradients are all-reduced internally
   - all models stay identical

---

## Training Flow (End-to-End)

### Step 1: Initialization
- processes launched (one per GPU)
- model created on each GPU
- weights broadcast from rank 0

---

### Step 2: Data Splitting
- `DistributedSampler` ensures unique batches per GPU

---

### Step 3: Forward Pass
- each GPU computes outputs independently

---

### Step 4: Backward Pass
- each GPU computes gradients locally

---

### Step 5: Gradient Synchronization (DDP Core)
- gradients are all-reduced across GPUs
- every GPU receives identical averaged gradients

---

### Step 6: Optimizer Step
- each GPU applies identical update
- models remain perfectly synchronized

---

## Key Design Principle

Even though computation is distributed:

> the model state is always globally consistent

This is achieved through synchronization after every training step.

---

## Mental Model

Think of distributed training as:

- multiple workers
- each doing partial computation
- but forced to agree after every step

So the system behaves like:

> one large GPU, split across many machines

---

## Why This Matters

This architecture enables:

- faster training (parallel compute)
- larger batch sizes
- scalable deep learning systems

Without communication primitives (broadcast + all_reduce), distributed training would not be possible.

---

## Repository Progression

This repo builds understanding in order:

1. `send_receive.py` → raw process communication
2. `broadcast.py` → state synchronization
3. `all_reduce.py` → gradient aggregation
4. `distributed_sampler.py` → data parallelism
5. `ddp_training.py` → full system

---

## Final Takeaway

Distributed training is not about running multiple models.

It is about:

> coordinating multiple independent processes so they behave like a single coherent optimization system through structured communication.
