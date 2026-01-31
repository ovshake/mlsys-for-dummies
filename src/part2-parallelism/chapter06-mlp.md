# tp_mlp.py

> A complete Tensor-Parallel MLP block with minimal communication

This script implements the Megatron-style tensor-parallel MLP, showing how to chain column-parallel and row-parallel layers to minimize communication.

## What It Does

1. Implements a tensor-parallel MLP block:
   - First linear: Column-parallel (expands hidden → 4×hidden)
   - Activation: GeLU (local, no communication)
   - Second linear: Row-parallel (contracts 4×hidden → hidden)
2. Shows that only ONE all_reduce is needed per MLP forward pass
3. Compares with naive approach (2 all_reduces)

## The Megatron Trick

```
MLP(X) = GeLU(X @ W1) @ W2

Naive: all_reduce after W1, all_reduce after W2 = 2 communications
Smart: column-parallel W1, row-parallel W2 = 1 communication!
```

**Why it works:**
- Column-parallel W1 produces split outputs: `[Y₀ | Y₁ | Y₂ | Y₃]`
- Each GPU applies GeLU locally
- Row-parallel W2 expects split inputs (which we have!)
- Only need all_reduce at the end

## Run It

```bash
python tutorial/part2-parallelism/chapter06-tensor-parallel/scripts/tp_mlp.py
```

## Architecture Visualization

```
            X (input)
               │
               ▼
     ┌─────────────────────┐
     │   Column-Parallel   │  ← W1 split by columns
     │     Linear (W1)     │     No communication
     └──────────┬──────────┘
               │
               ▼
     ┌─────────────────────┐
     │       GeLU          │  ← Local operation
     │   (no comm needed)  │
     └──────────┬──────────┘
               │
               ▼
     ┌─────────────────────┐
     │    Row-Parallel     │  ← W2 split by rows
     │     Linear (W2)     │
     └──────────┬──────────┘
               │
               ▼
     ┌─────────────────────┐
     │    all_reduce       │  ← Only sync point!
     │                     │
     └──────────┬──────────┘
               │
               ▼
            Y (output)
```

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter06-tensor-parallel/scripts/tp_mlp.py}}
```
