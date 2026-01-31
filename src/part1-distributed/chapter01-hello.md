# hello_distributed.py

> Your first distributed program - see multiple processes communicate!

This is the "Hello, World!" of distributed computing. It spawns multiple processes that initialize a process group and communicate.

## What It Does

1. Spawns 4 worker processes using `mp.spawn()`
2. Each process initializes the distributed environment
3. Processes perform a simple `all_gather` to collect data from everyone
4. Each process prints what it received

## Run It

```bash
# Default: 4 processes
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py

# Custom world size
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py --world-size 8
```

## Expected Output

```
[Rank 0] Hello! I see 4 processes in the world.
[Rank 1] Hello! I see 4 processes in the world.
[Rank 2] Hello! I see 4 processes in the world.
[Rank 3] Hello! I see 4 processes in the world.
[Rank 0] Gathered values from all ranks: [0, 1, 2, 3]
...
```

## Key Concepts Demonstrated

- **`mp.spawn()`** - Creates multiple processes, automatically passing rank
- **`dist.init_process_group()`** - The handshake that enables communication
- **`dist.all_gather()`** - Collect data from all processes

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py}}
```
