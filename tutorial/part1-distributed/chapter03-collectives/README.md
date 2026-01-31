# Chapter 3: Collective Communication Operations

> *"In distributed training, all_reduce is the workhorse. Everything else is warm-up."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain what each collective operation does: broadcast, scatter, gather, all_gather, reduce, all_reduce
- Implement gradient synchronization using all_reduce
- Choose the right collective for different scenarios
- Compose primitives to build complex operations (distributed softmax)

## Prerequisites

- Completed Chapters 1 & 2
- Understanding of rank, world_size, and basic communication
- Basic linear algebra (matrix operations)

## Concept Overview

### What are Collective Operations?

Unlike point-to-point (`send`/`recv`) where two specific processes communicate, **collective operations** involve *all* processes in a group simultaneously. They're the building blocks of distributed deep learning.

### The Collective Operation Zoo

| Operation | Description | Data Flow |
|-----------|-------------|-----------|
| **broadcast** | One process sends to all | `[A] → [A] [A] [A] [A]` |
| **scatter** | Split and distribute | `[A B C D] → [A] [B] [C] [D]` |
| **gather** | Collect to one process | `[A] [B] [C] [D] → [A B C D]` |
| **all_gather** | Collect to all processes | `[A] [B] [C] [D] → [ABCD] [ABCD] [ABCD] [ABCD]` |
| **reduce** | Aggregate to one process | `[1] [2] [3] [4] → [10]` (sum) |
| **all_reduce** | Aggregate to all processes | `[1] [2] [3] [4] → [10] [10] [10] [10]` (sum) |
| **reduce_scatter** | Reduce + scatter | `[A] [B] [C] [D] → [sum(A)] [sum(B)] [sum(C)] [sum(D)]` |

### Visual Guide

```
BROADCAST (src=0):                    SCATTER (src=0):
┌───┐                                 ┌───┬───┬───┬───┐
│ A │ ─┐                              │ A │ B │ C │ D │
└───┘  │                              └─┬─┴─┬─┴─┬─┴─┬─┘
       │  ┌───┐ ┌───┐ ┌───┐ ┌───┐       │   │   │   │
       └──► A │ │ A │ │ A │ │ A │       ▼   ▼   ▼   ▼
          └───┘ └───┘ └───┘ └───┘     ┌───┐┌───┐┌───┐┌───┐
          R0    R1    R2    R3        │ A ││ B ││ C ││ D │
                                      └───┘└───┘└───┘└───┘
                                      R0   R1   R2   R3

ALL_GATHER:                           ALL_REDUCE (sum):
┌───┐ ┌───┐ ┌───┐ ┌───┐               ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ A │ │ B │ │ C │ │ D │               │ 1 │ │ 2 │ │ 3 │ │ 4 │
└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘               └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  │     │     │     │                   │     │     │     │
  └─────┴─────┴─────┘                   └─────┴─────┴─────┘
          │                                     │
          ▼                                     ▼
  ┌───────────────┐                         ┌──────┐
  │ A │ B │ C │ D │ (on all ranks)          │  10  │ (on all ranks)
  └───────────────┘                         └──────┘
```

### The Star: all_reduce

`all_reduce` is the most important collective operation in distributed training. Here's why:

In data-parallel training:
1. Each GPU has a copy of the model
2. Each GPU computes gradients on different data
3. **Gradients must be averaged across all GPUs** ← `all_reduce`!
4. Each GPU updates its model with the averaged gradients

```python
# This single line synchronizes gradients across all GPUs
dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
gradient /= world_size  # Average
```

### Reduction Operations

For `reduce` and `all_reduce`, you specify the aggregation operation:

| Operation | Python | Result |
|-----------|--------|--------|
| `ReduceOp.SUM` | `sum(values)` | Sum all |
| `ReduceOp.PRODUCT` | `prod(values)` | Multiply all |
| `ReduceOp.MIN` | `min(values)` | Minimum |
| `ReduceOp.MAX` | `max(values)` | Maximum |

### Memory Semantics: In-Place vs Out-of-Place

Some operations modify tensors in-place, others require output buffers:

```python
# all_reduce: IN-PLACE
tensor = torch.tensor([rank])
dist.all_reduce(tensor)  # tensor is modified

# all_gather: OUT-OF-PLACE
tensor = torch.tensor([rank])
gathered = [torch.zeros(1) for _ in range(world_size)]
dist.all_gather(gathered, tensor)  # tensor unchanged, gathered filled
```

## Code Walkthrough

### Script 1: collective_cheatsheet.py

This script demonstrates all major collective operations with clear before/after output. Run it to see exactly what each operation does.

### Script 2: distributed_mean.py

A practical example: computing the mean of distributed data using all_reduce. This is exactly what happens during gradient synchronization.

### Script 3: distributed_softmax.py

An advanced example showing how to compose primitives. Computing softmax across distributed data requires:
1. `all_reduce` to find the global max (for numerical stability)
2. Local exp computation
3. `all_reduce` to find the global sum
4. Local division

## When to Use What?

| Scenario | Operation | Why |
|----------|-----------|-----|
| Share hyperparameters from rank 0 | `broadcast` | One source, all need it |
| Distribute a dataset | `scatter` | Split data across workers |
| Collect predictions | `gather` | Aggregate results |
| Synchronize gradients | `all_reduce` | Everyone needs the sum |
| Share embeddings for lookup | `all_gather` | Everyone needs all data |
| Gradient bucketing | `reduce_scatter` | Efficient for large models |

## Try It Yourself

### Exercise 1: Distributed Mean (Easy)

Each process has a different number. Use `all_reduce` to compute the mean across all processes.

### Exercise 2: Distributed Argmax (Medium)

Each process has a tensor. Find the global maximum value and *which rank* has it.

Hint: Use `all_reduce` with `MAX`, then `all_gather` to find who has it.

### Exercise 3: Ring All-Reduce (Hard)

Implement `all_reduce` using only `send`/`recv` in a ring pattern:
1. Each process sends to (rank + 1) % world_size
2. Each process receives from (rank - 1) % world_size
3. Iterate until all data is aggregated

This is essentially what NCCL's ring algorithm does!

## Key Takeaways

1. **all_reduce is king** - It's the foundation of gradient synchronization
2. **Collective operations are optimized** - Don't reimplement them with send/recv
3. **Know your memory semantics** - Some ops are in-place, some aren't
4. **Composability is powerful** - Complex operations (softmax) build from primitives
5. **scatter vs broadcast** - scatter distributes different data, broadcast replicates same data

## Performance Intuition

Communication volume for N processes, each with data size D:

| Operation | Volume per process |
|-----------|-------------------|
| broadcast | D (receive) |
| scatter | D/N (receive) |
| all_gather | D * (N-1) (send + receive) |
| all_reduce | 2D * (N-1) / N (ring algorithm) |

This is why all_reduce with the ring algorithm is efficient—it has O(D) volume regardless of N (though latency scales with N).

## What's Next?

In Chapter 4, we'll dive into the actual algorithms NCCL uses (Ring, Tree, Double Binary Tree) and how to inspect GPU topology to understand communication performance.

## Further Reading

- [PyTorch Collective Operations](https://pytorch.org/docs/stable/distributed.html#collective-functions)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- Original source: [`torch/torch-distributed/codes/all-reduce-softmax.py`](../../../torch/torch-distributed/codes/all-reduce-softmax.py)
