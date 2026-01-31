# memory_timeline.py

> Visualize GPU memory usage across RLHF phases

This script shows how memory is allocated and freed during different phases of RLHF training.

## What It Does

1. Simulates RLHF memory allocation
2. Shows memory usage for each phase
3. Identifies peak memory and bottlenecks
4. Demonstrates why phase-based swapping helps

## Run It

```bash
python tutorial/part4-rlhf/chapter14-rlhf-architecture/scripts/memory_timeline.py
```

## Example Output

```
=== RLHF Memory Timeline (70B model, 8 GPUs) ===

GPU Memory Available: 80 GB per GPU

Phase 1: Generation
  ┌─────────────────────────────────────────────────────────────┐
  │ Actor weights (TP=8):      17.5 GB                          │
  │ Reference weights (TP=8):  17.5 GB                          │
  │ Reward model (TP=8):       17.5 GB                          │
  │ KV Cache (batch=32):       20.0 GB                          │
  │ ─────────────────────────────────────                       │
  │ Total:                     72.5 GB  [OK - fits in 80 GB]    │
  └─────────────────────────────────────────────────────────────┘

Phase 2: Transition (Free KV cache, load critic)
  Memory freed:  20.0 GB (KV cache)
  Memory allocated: 17.5 GB (Critic weights)

Phase 3: Training
  ┌─────────────────────────────────────────────────────────────┐
  │ Actor weights (TP=8):      17.5 GB                          │
  │ Critic weights (TP=8):     17.5 GB                          │
  │ Actor gradients:           17.5 GB                          │
  │ Critic gradients:          17.5 GB                          │
  │ Adam states (2x):          70.0 GB  ← Offloaded!            │
  │ Activations:               10.0 GB                          │
  │ ─────────────────────────────────────                       │
  │ Without offload:          150.5 GB  [FAIL - OOM]            │
  │ With optimizer offload:    80.0 GB  [OK - barely fits]      │
  └─────────────────────────────────────────────────────────────┘

Memory Timeline:
Time →
     ┌──────────────────────────────────────────────────────────┐
 80GB│████████████████████░░░░░░░░████████████████████████████│
     │ Generation          │Swap│      Training               │
 60GB│████████████████████░░░░░░░░████████████████████████████│
     │                     │    │                              │
 40GB│████████████████████░░░░░░░░████████████████████████████│
     │                     │    │                              │
 20GB│████████████████████░░░░░░░░████████████████████████████│
     │                     │    │                              │
  0GB└──────────────────────────────────────────────────────────┘
     Legend: █ = allocated, ░ = free
```

## Why Phase-Based Memory Matters

Without swapping:
```
All 4 models + optimizer + KV cache = 200+ GB per GPU
= Out of Memory!
```

With smart swapping:
```
Generation: Models + KV cache (no optimizer) = 72 GB
Training: Models + optimizer + grads (no KV) = 80 GB
= Fits!
```

## Source Code

```python
{{#include ../../tutorial/part4-rlhf/chapter14-rlhf-architecture/scripts/memory_timeline.py}}
```
