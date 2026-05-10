
# Distributed Training Explained

## Core Idea

Distributed training allows multiple GPUs to train the same model simultaneously.

Each GPU:
1. Receives different data
2. Computes gradients independently
3. Synchronizes gradients
4. Updates identical model weights

---

## Key Distributed Operations

### send / recv

Point-to-point communication.

```python
dist.send()
dist.recv()
