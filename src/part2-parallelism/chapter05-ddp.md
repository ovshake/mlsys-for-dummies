# simple_ddp.py

> Understanding Distributed Data Parallel from first principles

This script implements data-parallel training both manually (with explicit all_reduce) and using PyTorch's DDP wrapper, so you can see exactly what's happening under the hood.

## What It Does

1. Creates a simple model on each process
2. **Manual approach**: Runs forward/backward, then all_reduce gradients explicitly
3. **DDP approach**: Wraps model in DDP, gradients sync automatically
4. Verifies both approaches produce identical results

## Run It

```bash
python tutorial/part2-parallelism/chapter05-data-parallel/scripts/simple_ddp.py
```

## Key Learning Points

**Manual Gradient Sync:**
```python
# After loss.backward()
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size  # Average
```

**DDP Wrapper:**
```python
model = DDP(model, device_ids=[rank])
# Gradients are synced automatically during backward()!
```

## Why DDP is Better

DDP optimizes what we do manually:
- **Overlaps communication with computation** - Starts all_reduce while backward is still running
- **Buckets gradients** - Groups small gradients for efficient communication
- **Handles edge cases** - Unused parameters, mixed precision, etc.

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter05-data-parallel/scripts/simple_ddp.py}}
```
