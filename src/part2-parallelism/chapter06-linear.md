# tp_linear.py

> Implementing Column-Parallel and Row-Parallel Linear layers from scratch

This script demonstrates the fundamental building blocks of tensor parallelism: how to split a linear layer's weight matrix across multiple GPUs.

## What It Does

1. Implements `ColumnParallelLinear` - splits weights by columns
2. Implements `RowParallelLinear` - splits weights by rows
3. Verifies that parallel execution equals sequential execution
4. Shows where communication is (and isn't) needed

## The Two Splitting Strategies

**Column-Parallel (no sync in forward):**
```
W = [W₀ | W₁ | W₂ | W₃]  ← split by columns

Y = X @ W = [X@W₀ | X@W₁ | X@W₂ | X@W₃]

Each GPU computes its slice independently!
```

**Row-Parallel (needs all_reduce):**
```
W = [W₀]   ← split by rows
    [W₁]
    [W₂]
    [W₃]

Y = X@W = X₀@W₀ + X₁@W₁ + X₂@W₂ + X₃@W₃

Requires all_reduce to sum partial results!
```

## Run It

```bash
python tutorial/part2-parallelism/chapter06-tensor-parallel/scripts/tp_linear.py
```

## Key Verification

The script verifies:
```python
# Column parallel: concatenated outputs match full computation
torch.cat([y0, y1, y2, y3], dim=-1) == X @ W_full

# Row parallel: summed outputs match full computation
y0 + y1 + y2 + y3 == X @ W_full
```

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter06-tensor-parallel/scripts/tp_linear.py}}
```
