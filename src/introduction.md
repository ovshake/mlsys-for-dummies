# ML Systems Infrastructure Tutorial

> **From distributed primitives to production RLHF: A hands-on journey through ML infrastructure**

This tutorial takes you from zero to understanding how large-scale ML systems work. If you're comfortable with PyTorch and understand transformers but wonder "how do people actually train GPT-4?", this is for you.

## Who This Is For

- **Strong ML background**: You know PyTorch, can train models, understand attention
- **New to systems**: You haven't done distributed training, don't know NCCL from TCP
- **Curious about scale**: You want to understand how 1000-GPU training actually works

## What You'll Learn

By the end of this tutorial, you'll understand:

1. **How GPUs talk to each other** - Communication primitives that enable distributed training
2. **How to parallelize training** - Data, tensor, and pipeline parallelism strategies
3. **How inference servers work** - KV cache, batching, and speculative decoding
4. **How RLHF systems are built** - The four-model dance that makes ChatGPT possible

## Tutorial Structure

### Part I: Foundations of Distributed Computing (Chapters 1-4)

Start here. These concepts are the alphabet of distributed systems.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| [Chapter 1](./part1-distributed/chapter01.md) | Your First Distributed Program | rank, world_size, process groups |
| [Chapter 2](./part1-distributed/chapter02.md) | Point-to-Point Communication | send/recv, deadlock avoidance |
| [Chapter 3](./part1-distributed/chapter03.md) | Collective Operations | all_reduce, broadcast, scatter |
| [Chapter 4](./part1-distributed/chapter04.md) | NCCL and GPU Topology | Ring/Tree algorithms, NVLink |

### Part II: Parallelism Strategies (Chapters 5-7)

Now you know the primitives. Let's use them to train models that don't fit on one GPU.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| [Chapter 5](./part2-parallelism/chapter05.md) | Data Parallelism Deep Dive | DDP, FSDP, ZeRO stages |
| [Chapter 6](./part2-parallelism/chapter06.md) | Tensor Parallelism | Column/row parallel, Megatron-style |
| [Chapter 7](./part2-parallelism/chapter07.md) | Pipeline & Expert Parallelism | 1F1B scheduling, MoE |

### Part III: LLM Inference Systems (Chapters 8-11)

Training is half the story. Serving models efficiently is the other half.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| [Chapter 8](./part3-inference/chapter08.md) | Server Anatomy | Request lifecycle, prefill/decode |
| [Chapter 9](./part3-inference/chapter09.md) | KV Cache Management | PagedAttention, RadixCache |
| [Chapter 10](./part3-inference/chapter10.md) | Scheduling & CUDA Graphs | Zero-overhead scheduling |
| [Chapter 11](./part3-inference/chapter11.md) | Speculative & Constraint Decoding | Draft models, structured output |

### Part IV: RLHF Systems (Chapters 12-14)

The grand finale: training models with human feedback.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| [Chapter 12](./part4-rlhf/chapter12.md) | RL Fundamentals for LLMs | PPO, GAE, policy gradients |
| [Chapter 13](./part4-rlhf/chapter13.md) | RLHF Computation Flow | Four models, reward calculation |
| [Chapter 14](./part4-rlhf/chapter14.md) | RLHF System Architecture | Co-located vs disaggregated |

## How to Use This Tutorial

### Prerequisites

```bash
pip install torch  # Core requirement
pip install gymnasium  # For RL chapter (optional)
```

No GPU required! All scripts have CPU fallback with the `gloo` backend.

### Learning Path

**Recommended order**: Follow chapters sequentially. Each builds on the previous.

**Hands-on learning**: Each chapter has:
- Conceptual explanation (the chapter page)
- Runnable scripts (linked as sub-pages)
- Exercises to try

### Running the Scripts

```bash
# Chapter 1: Your first distributed program
python tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py

# Chapter 3: Collective operations
python tutorial/part1-distributed/chapter03-collectives/scripts/collective_cheatsheet.py
```

## Quick Start: See Something Work!

Want to jump in immediately? Run this:

```bash
python tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py  # Check your environment
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py  # Your first distributed program!
```

You should see 4 processes talking to each other!

## Core Mental Models

### The Parallelism Zoo

```
Problem: Model too big?
├── Too big for memory → Data Parallelism (replicate model)
│   └── Still too big → ZeRO/FSDP (shard everything)
├── One layer too big → Tensor Parallelism (split layers)
└── All layers too big → Pipeline Parallelism (split model)

Problem: Model is MoE?
└── Add Expert Parallelism (distribute experts)
```

### The Memory Hierarchy

```
Fast ──────────────────────────────────────────► Slow
GPU L2   GPU HBM   CPU RAM   NVMe SSD   Network

90TB/s   3TB/s     200GB/s   7GB/s      50GB/s

Goal: Keep computation in fast memory
Strategy: Overlap communication with computation
```

### The Inference Pipeline

```
Request → Tokenizer → Scheduler → Model Runner → Detokenizer → Response
                         ↓
              [Prefill: Process prompt]
                         ↓
              [Decode: Generate tokens]
                         ↓
              [KV Cache: Remember context]
```

---

*"The best way to understand distributed systems is to build one. The second best way is this tutorial."*

Happy learning!
