# distributed_mean.py

> Computing global mean with all_reduce - exactly what gradient sync does!

This script demonstrates the fundamental pattern of distributed gradient synchronization: using `all_reduce` to compute global averages.

## What It Does

1. Each process has local data (simulating local gradients)
2. Uses `all_reduce` to sum all values
3. Divides by world size to get the mean
4. Every process now has the same averaged value

## The Pattern

```
Local gradients:     [1.0]  [2.0]  [3.0]  [4.0]
                       │      │      │      │
                       └──────┴──────┴──────┘
                              │
                         all_reduce (SUM)
                              │
                              ▼
Global sum:          [10.0] [10.0] [10.0] [10.0]
                              │
                         ÷ world_size
                              │
                              ▼
Global mean:         [2.5]  [2.5]  [2.5]  [2.5]
```

## Run It

```bash
python tutorial/part1-distributed/chapter03-collectives/scripts/distributed_mean.py
```

## Why This Matters

This exact pattern is used in **Distributed Data Parallel (DDP)**:

```python
# In DDP, after backward pass:
for gradient in model.gradients():
    dist.all_reduce(gradient, op=ReduceOp.SUM)
    gradient /= world_size
```

All GPUs end up with identical averaged gradients, so the model stays synchronized.

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter03-collectives/scripts/distributed_mean.py}}
```
