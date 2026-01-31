# Chapter 2: Point-to-Point Communication

> *"Before there were collective operations, there were two processes passing notes in class."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Send tensors directly between specific processes using `send`/`recv`
- Understand blocking vs non-blocking communication (`isend`/`irecv`)
- Recognize and avoid common deadlock patterns
- Implement a simple pipeline pattern

## Prerequisites

- Completed Chapter 1: Your First Distributed Program
- Understanding of `rank` and `world_size`
- Ability to initialize a distributed process group

## Concept Overview

### What is Point-to-Point Communication?

In Chapter 1, we used `all_gather` and `all_reduce`—these are **collective operations** where everyone participates. But sometimes you need surgical precision: process 2 needs to send data specifically to process 5, and no one else.

This is **point-to-point communication**: a direct channel between two specific processes.

```
Collective (all_reduce):          Point-to-Point (send/recv):

    [0] [1] [2] [3]                    [0] ──────► [3]
      \   |   |   /                          (direct)
       \  |  |  /
        ▼ ▼ ▼ ▼
       [combined]
```

### The Four Operations

| Operation | Blocking? | Description |
|-----------|-----------|-------------|
| `send(tensor, dst)` | Yes | Send tensor to process `dst`, wait until done |
| `recv(tensor, src)` | Yes | Receive tensor from process `src`, wait until done |
| `isend(tensor, dst)` | No | Start sending, return immediately with a handle |
| `irecv(tensor, src)` | No | Start receiving, return immediately with a handle |

The "i" prefix stands for "immediate" (non-blocking).

### The Blocking vs Non-Blocking Dance

**Blocking operations** are simpler but can lead to deadlocks:

```python
# DEADLOCK! Both processes wait for each other forever
# Process 0                    # Process 1
send(tensor, dst=1)           send(tensor, dst=0)
recv(tensor, src=1)           recv(tensor, src=0)
```

Both processes are stuck on `send()`, waiting for someone to receive—but no one is receiving because everyone is sending!

**The fix**: Carefully order operations or use non-blocking variants.

```python
# CORRECT: Interleaved send/recv
# Process 0                    # Process 1
send(tensor, dst=1)           recv(tensor, src=0)
recv(tensor, src=1)           send(tensor, dst=0)
```

### Non-Blocking Operations

Non-blocking operations return a `Work` handle immediately:

```python
# isend returns immediately, data transfer happens in background
handle = dist.isend(tensor, dst=1)

# Do other work while transfer is in progress
compute_something_else()

# Wait for the transfer to complete before using the tensor
handle.wait()
```

This is essential for overlapping computation with communication—a key optimization in real systems.

### Pipeline Parallelism: Where Point-to-Point Shines

Point-to-point communication is the backbone of **pipeline parallelism**. Imagine a model split across 4 GPUs:

```
Input ──► [Stage 0] ──► [Stage 1] ──► [Stage 2] ──► [Stage 3] ──► Output
           GPU 0        GPU 1         GPU 2         GPU 3
              │            │             │             │
              └──send──────┴─────────────┴─────────────┘
                       activations flow forward
```

Each stage processes its part and `send`s the activations to the next stage. The last stage computes the loss and gradients flow backward via `send`/`recv` in the opposite direction.

## Code Walkthrough

### Script 1: send_recv_basic.py

This script demonstrates the fundamental pattern: passing a tensor through a chain of processes.

```
Rank 0 ──► Rank 1 ──► Rank 2 ──► Rank 3
   (creates)  (adds 10)  (adds 10)  (prints final)
```

Key points:
- Rank 0 **only sends** (it's the source)
- Middle ranks **receive then send** (they're relays)
- Last rank **only receives** (it's the sink)

### Script 2: async_communication.py

This script shows how to use `isend`/`irecv` for non-blocking communication:

1. Start the communication
2. Do useful work while data transfers
3. Wait for completion
4. Use the received data

This pattern is critical for hiding communication latency.

### Script 3: pipeline_simulation.py

A mini pipeline parallelism demo! We split a simple "model" (just matrix multiplications) across processes and pass activations forward.

## Common Pitfalls

### Pitfall 1: Mismatched Send/Recv

```python
# Process 0: sends to 1
dist.send(tensor, dst=1)

# Process 1: receives from 2 (WRONG!)
dist.recv(tensor, src=2)  # Will hang forever!
```

**Always ensure src/dst pairs match.**

### Pitfall 2: Buffer Reuse Before Completion

```python
handle = dist.isend(tensor, dst=1)
tensor.fill_(0)  # DANGER! Modifying buffer during transfer
handle.wait()
```

**Never modify a tensor while an async operation is in progress.**

### Pitfall 3: Forgetting to Wait

```python
handle = dist.irecv(tensor, src=0)
# Forgot handle.wait()!
print(tensor)  # Garbage data!
```

**Always call `.wait()` before using received data.**

## Try It Yourself

### Exercise 1: Ring Topology

Modify `send_recv_basic.py` to create a ring:
- Rank N sends to Rank (N+1) % world_size
- This means Rank 3 sends back to Rank 0

What value should the tensor have after going full circle?

### Exercise 2: Bidirectional Communication

Write a script where:
- Even ranks send to odd ranks
- Odd ranks send to even ranks
- All at the same time (use isend/irecv to avoid deadlock)

### Exercise 3: Measure Latency

Use `time.perf_counter()` to measure:
1. Time for a blocking `send`/`recv` pair
2. Time for an `isend`/`irecv` pair with `wait()`

Is there a difference? Why or why not?

## Key Takeaways

1. **Point-to-point is surgical** - You specify exactly which process sends and receives
2. **Blocking can deadlock** - Be very careful with `send`/`recv` ordering
3. **Non-blocking enables overlap** - `isend`/`irecv` let you compute while communicating
4. **Pipeline parallelism uses this heavily** - Activations flow forward, gradients flow backward
5. **Always wait() before using data** - Non-blocking doesn't mean the data is ready

## Mental Model: The Post Office

Think of distributed communication like a post office:

- **`send`** = Walking to the post office, handing over your package, and waiting until it's delivered
- **`isend`** = Dropping your package in a mailbox and walking away
- **`recv`** = Waiting at home until the doorbell rings
- **`irecv`** = Setting up a notification to ping you when a package arrives

The post office (NCCL/Gloo) handles the actual delivery in the background.

## What's Next?

In Chapter 3, we'll explore **collective operations** in depth—broadcast, scatter, all_gather, and the all-important all_reduce that makes gradient synchronization possible.

## Further Reading

- [PyTorch Point-to-Point Communication](https://pytorch.org/docs/stable/distributed.html#point-to-point-communication)
- Original source: [`torch/torch-distributed/codes/send-recv-sync.py`](../../../torch/torch-distributed/codes/send-recv-sync.py)
