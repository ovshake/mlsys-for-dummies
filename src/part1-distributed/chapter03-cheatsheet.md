# collective_cheatsheet.py

> A visual demonstration of all collective operations

This script is your reference guide to collective operations. It demonstrates each operation with clear before/after output so you can see exactly what happens.

## What It Does

Runs through all major collective operations:
1. **Broadcast** - One rank sends to all
2. **Scatter** - Split and distribute
3. **Gather** - Collect to one rank
4. **All-Gather** - Everyone gets everything
5. **Reduce** - Aggregate to one rank
6. **All-Reduce** - Aggregate to all ranks

## Run It

```bash
python tutorial/part1-distributed/chapter03-collectives/scripts/collective_cheatsheet.py
```

## Expected Output

```
=== BROADCAST (src=0) ===
Before: Rank 0=[42], Rank 1=[0], Rank 2=[0], Rank 3=[0]
After:  Rank 0=[42], Rank 1=[42], Rank 2=[42], Rank 3=[42]

=== SCATTER (src=0) ===
Before: Rank 0=[10,20,30,40]
After:  Rank 0=[10], Rank 1=[20], Rank 2=[30], Rank 3=[40]

=== ALL_REDUCE (sum) ===
Before: Rank 0=[1], Rank 1=[2], Rank 2=[3], Rank 3=[4]
After:  All ranks=[10] (1+2+3+4)
```

## Quick Reference

| Operation | Before | After |
|-----------|--------|-------|
| broadcast | `[A] [_] [_] [_]` | `[A] [A] [A] [A]` |
| scatter | `[ABCD] [_] [_] [_]` | `[A] [B] [C] [D]` |
| gather | `[A] [B] [C] [D]` | `[ABCD] [_] [_] [_]` |
| all_gather | `[A] [B] [C] [D]` | `[ABCD] [ABCD] [ABCD] [ABCD]` |
| reduce | `[1] [2] [3] [4]` | `[10] [_] [_] [_]` |
| all_reduce | `[1] [2] [3] [4]` | `[10] [10] [10] [10]` |

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter03-collectives/scripts/collective_cheatsheet.py}}
```
