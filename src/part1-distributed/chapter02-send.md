# send_recv_basic.py

> Demonstrates basic point-to-point communication between processes

This script shows how to pass a tensor through a chain of processes using blocking `send` and `recv` operations.

## What It Does

1. Rank 0 creates a tensor with its rank value
2. Each rank receives from the previous rank and adds 10
3. Each rank sends to the next rank
4. Final rank prints the accumulated result

## The Chain Pattern

```
Rank 0 ──► Rank 1 ──► Rank 2 ──► Rank 3
 [0]       [0+10]     [10+10]    [20+10]
           = [10]     = [20]     = [30]
```

## Run It

```bash
python tutorial/part1-distributed/chapter02-point-to-point/scripts/send_recv_basic.py
```

## Expected Output

```
[Rank 0] Sending tensor([0.])
[Rank 1] Received tensor([0.]), adding 10, sending tensor([10.])
[Rank 2] Received tensor([10.]), adding 10, sending tensor([20.])
[Rank 3] Received tensor([20.]), final value: tensor([30.])
```

## Key Concepts Demonstrated

- **Blocking `send`/`recv`** - Operations wait until completion
- **Chain topology** - Data flows linearly through ranks
- **Conditional logic by rank** - First, middle, and last ranks have different roles

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter02-point-to-point/scripts/send_recv_basic.py}}
```
