# Chapter 1: Your First Distributed Program

> *"The journey of a thousand GPUs begins with a single `init_process_group`."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain what `rank`, `world_size`, and process groups mean
- Initialize a distributed PyTorch environment
- Run code across multiple processes that communicate with each other
- Understand why we use multiprocessing (not multithreading) for distributed training

## Prerequisites

- Python 3.8+
- PyTorch installed (`pip install torch`)
- Basic understanding of PyTorch tensors
- No GPU required (we'll use CPU fallback)

## Concept Overview

### Why Distributed Computing?

Imagine you're training a large language model. A single GPU has maybe 80GB of memory, but your model needs 500GB just for its parameters. What do you do?

The answer is **distributed computing**: spreading your computation across multiple GPUs (and multiple machines). But here's the catch—those GPUs need to talk to each other. A lot.

### The Python Problem: GIL

Python has a notorious feature called the **Global Interpreter Lock (GIL)**. It prevents true parallel execution of Python threads. For compute-intensive tasks like deep learning, this is a showstopper.

```
Thread 1: "I want to multiply matrices!"
Thread 2: "I also want to multiply matrices!"
GIL: "One at a time, please. Thread 1, you go first."
Thread 2: *waits impatiently*
```

The solution? **Multiprocessing**. Instead of threads sharing one Python interpreter, we spawn completely separate Python processes. Each process gets its own interpreter, its own memory space, and (crucially) its own GPU.

### The Distributed Vocabulary

Before we write code, let's learn the language:

| Term | Definition | Analogy |
|------|------------|---------|
| **World** | All processes participating in training | The entire team |
| **World Size** | Total number of processes | Team size |
| **Rank** | Unique ID for each process (0 to world_size-1) | Employee ID |
| **Local Rank** | Process ID within a single machine | Desk number in an office |
| **Process Group** | A subset of processes that communicate together | A project sub-team |
| **Backend** | The communication library (NCCL, Gloo, MPI) | The phone system |

```
Machine 0                    Machine 1
┌──────────────────┐        ┌──────────────────┐
│  GPU 0 (rank=0)  │        │  GPU 0 (rank=2)  │
│  GPU 1 (rank=1)  │◄──────►│  GPU 1 (rank=3)  │
└──────────────────┘        └──────────────────┘
     local_rank: 0,1             local_rank: 0,1
```

### Communication Backends

PyTorch supports three backends for inter-process communication:

| Backend | Best For | Supports CPU? | Supports GPU? |
|---------|----------|---------------|---------------|
| **NCCL** | GPU training | No | Yes (NVIDIA only) |
| **Gloo** | CPU training, fallback | Yes | Limited |
| **MPI** | HPC clusters | Yes | Yes |

**Rule of thumb**: Use NCCL for GPU training, Gloo for CPU or when NCCL isn't available.

## Code Walkthrough

### Script 1: verify_setup.py

Let's start by checking if your environment is ready for distributed computing.

This script checks:
1. Is PyTorch installed?
2. Is CUDA available?
3. Which distributed backends are supported?
4. How many GPUs do we have?

Run it with:
```bash
python tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py
```

### Script 2: hello_distributed.py

Now for the main event—your first distributed program!

The key function is `torch.distributed.init_process_group()`:

```python
import torch.distributed as dist

dist.init_process_group(
    backend="gloo",      # Communication backend
    init_method="...",   # How processes find each other
    world_size=4,        # Total number of processes
    rank=0               # This process's ID
)
```

**How do processes find each other?**

The `init_method` parameter tells processes how to rendezvous:
- `"env://"` - Use environment variables (MASTER_ADDR, MASTER_PORT)
- `"tcp://hostname:port"` - Explicit TCP address
- `"file:///path/to/file"` - Shared filesystem (for single-machine testing)

For our tutorial, we'll use `mp.spawn()` which handles this automatically.

### Understanding mp.spawn()

```python
import torch.multiprocessing as mp

def worker(rank, world_size):
    # Each process runs this function
    print(f"Hello from rank {rank}!")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size)
```

`mp.spawn()`:
1. Creates `world_size` new processes
2. Calls `worker(rank, world_size)` in each process
3. Passes `rank` as the first argument automatically

Run it with:
```bash
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py
```

You should see output from 4 different processes!

## Try It Yourself

### Exercise 1: Modify World Size

Edit `hello_distributed.py` to use `world_size=8`. What changes in the output?

### Exercise 2: Process-Specific Work

Modify the worker function so that:
- Even-ranked processes print "I handle even data!"
- Odd-ranked processes print "I handle odd data!"

<details>
<summary>Hint</summary>

```python
if rank % 2 == 0:
    print(f"Rank {rank}: I handle even data!")
else:
    print(f"Rank {rank}: I handle odd data!")
```
</details>

### Exercise 3: Investigate Environment Variables

Add code to print the following environment variables:
- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`
- `MASTER_ADDR`
- `MASTER_PORT`

What values do they have? (Hint: Use `os.environ.get("VAR_NAME", "not set")`)

## Key Takeaways

1. **Multiprocessing, not multithreading** - Python's GIL forces us to use separate processes
2. **Every process has a unique rank** - This is how you identify "who am I?"
3. **init_process_group is the handshake** - Processes can't communicate until they've all called this
4. **Choose the right backend** - NCCL for GPUs, Gloo for CPU/fallback
5. **mp.spawn handles the boilerplate** - It creates processes and passes ranks automatically

## What's Next?

In [Chapter 2](./chapter02.md), we'll learn **point-to-point communication**—how two specific processes can send data directly to each other. This is the foundation for pipeline parallelism.

## Further Reading

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Distributed Data Parallel Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
