# pipeline_simulation.py

> A mini pipeline parallelism demo with forward pass through distributed stages

This script simulates pipeline parallelism by splitting a simple neural network across multiple processes and passing activations forward.

## What It Does

1. Creates a simple "model" (matrix multiplications) split across stages
2. Input data enters at rank 0
3. Activations flow forward through each stage via `send`/`recv`
4. Final output emerges at the last rank

## Pipeline Architecture

```
     [Input]
        │
        ▼
   ┌─────────┐
   │ Stage 0 │  ← Rank 0: First linear layer
   │  Linear │
   └────┬────┘
        │ send activations
        ▼
   ┌─────────┐
   │ Stage 1 │  ← Rank 1: Second linear layer
   │  Linear │
   └────┬────┘
        │ send activations
        ▼
   ┌─────────┐
   │ Stage 2 │  ← Rank 2: Third linear layer
   │  Linear │
   └────┬────┘
        │ send activations
        ▼
   ┌─────────┐
   │ Stage 3 │  ← Rank 3: Final layer + output
   │  Linear │
   └─────────┘
     [Output]
```

## Run It

```bash
python tutorial/part1-distributed/chapter02-point-to-point/scripts/pipeline_simulation.py
```

## Key Concepts Demonstrated

- **Pipeline parallelism** - Model split across devices
- **Activation passing** - Intermediate results flow between stages
- **Sequential dependency** - Each stage waits for the previous

## Why This Matters

Real pipeline parallelism (like GPipe or PipeDream) uses this same `send`/`recv` pattern but with:
- Micro-batching to keep all stages busy
- Backward pass for gradient computation
- Gradient checkpointing to save memory

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter02-point-to-point/scripts/pipeline_simulation.py}}
```
