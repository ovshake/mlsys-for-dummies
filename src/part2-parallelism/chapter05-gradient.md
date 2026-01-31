# gradient_sync_visualizer.py

> See exactly how gradients flow during distributed training

This script visualizes the gradient synchronization process, showing what each GPU has before and after all_reduce.

## What It Does

1. Each GPU computes gradients on its local batch
2. Displays gradients BEFORE synchronization (different on each GPU)
3. Performs all_reduce
4. Displays gradients AFTER synchronization (identical everywhere)

## Run It

```bash
python tutorial/part2-parallelism/chapter05-data-parallel/scripts/gradient_sync_visualizer.py
```

## Example Output

```
=== BEFORE Gradient Sync ===
Rank 0: layer1.weight.grad = [0.123, -0.456, 0.789, ...]
Rank 1: layer1.weight.grad = [0.234, -0.567, 0.890, ...]
Rank 2: layer1.weight.grad = [0.345, -0.678, 0.901, ...]
Rank 3: layer1.weight.grad = [0.456, -0.789, 0.012, ...]

=== Performing all_reduce... ===

=== AFTER Gradient Sync ===
Rank 0: layer1.weight.grad = [0.290, -0.623, 0.648, ...]  ← averaged
Rank 1: layer1.weight.grad = [0.290, -0.623, 0.648, ...]  ← same!
Rank 2: layer1.weight.grad = [0.290, -0.623, 0.648, ...]  ← same!
Rank 3: layer1.weight.grad = [0.290, -0.623, 0.648, ...]  ← same!
```

## The Insight

After all_reduce + averaging, every GPU has **the exact same gradient**. This is mathematically equivalent to computing the gradient on the combined batch from all GPUs.

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter05-data-parallel/scripts/gradient_sync_visualizer.py}}
```
