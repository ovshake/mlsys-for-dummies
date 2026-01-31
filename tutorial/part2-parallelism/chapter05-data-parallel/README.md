# Chapter 5: Data Parallelism Deep Dive

> *"Data parallelism is the gateway drug of distributed training. It's deceptively simple, yet optimizing it is an art."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement basic data-parallel training manually
- Explain how PyTorch DDP works under the hood
- Understand ZeRO stages and their memory tradeoffs
- Choose between DDP, FSDP, and DeepSpeed for your use case

## Prerequisites

- Completed Part I (Distributed Computing Foundations)
- Basic understanding of neural network training (forward, backward, optimizer step)
- Familiarity with PyTorch's autograd

## Concept Overview

### What is Data Parallelism?

Data parallelism is the simplest form of distributed training:
1. **Replicate** the entire model on each GPU
2. **Split** the training batch across GPUs
3. **Compute** forward and backward passes locally
4. **Synchronize** gradients across all GPUs
5. **Update** each model copy identically

```
                    Global Batch (size 256)
                    ┌───────────────────────────────┐
                    │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │
                    └─┬───┴─┬───┴─┬───┴─┬───┴─┬───┴─┬───┴─┬───┴─┬─┘
                      │     │     │     │     │     │     │     │
                      ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
                   GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7
                   ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
                   │ M │  │ M │  │ M │  │ M │  │ M │  │ M │  │ M │  │ M │
                   │ O │  │ O │  │ O │  │ O │  │ O │  │ O │  │ O │  │ O │
                   │ D │  │ D │  │ D │  │ D │  │ D │  │ D │  │ D │  │ D │
                   │ E │  │ E │  │ E │  │ E │  │ E │  │ E │  │ E │  │ E │
                   │ L │  │ L │  │ L │  │ L │  │ L │  │ L │  │ L │  │ L │
                   └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
                     │      │      │      │      │      │      │      │
                     └──────┴──────┴──────┴───┬──┴──────┴──────┴──────┘
                                              │
                                         all_reduce
                                         (gradients)
```

### The Core Insight: Gradient Averaging

Why does this work mathematically?

For a batch B split into B₀ and B₁:
```
∇L(B) = ∇L(B₀ ∪ B₁)
      = (1/|B|) Σᵢ ∇L(xᵢ)
      = (1/|B|) [Σᵢ∈B₀ ∇L(xᵢ) + Σᵢ∈B₁ ∇L(xᵢ)]
      = (|B₀|/|B|) · ∇L(B₀) + (|B₁|/|B|) · ∇L(B₁)
```

With equal splits: `∇L(B) = (∇L(B₀) + ∇L(B₁)) / 2`

This is exactly what `all_reduce(gradients, SUM) / world_size` computes!

### PyTorch DistributedDataParallel (DDP)

DDP is PyTorch's production-grade data parallelism implementation. Key features:

1. **Gradient Bucketing**: Groups small gradients into buckets for efficient all_reduce
2. **Overlap with Backward**: Starts all_reduce before backward is complete
3. **Broadcast Parameters**: Ensures all replicas start with identical weights

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")

# Create model and wrap with DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# Training loop (exactly like single-GPU!)
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients synchronized automatically!
    optimizer.step()
```

### The Memory Problem

Data parallelism replicates the entire model. For an LLM like LLaMA-70B:

| Component | Size per GPU |
|-----------|--------------|
| Parameters (FP16) | 140 GB |
| Gradients (FP16) | 140 GB |
| Optimizer states (Adam, FP32) | 560 GB |
| **Total** | **840 GB** |

No single GPU has 840 GB! This is where **ZeRO** comes in.

### ZeRO: Zero Redundancy Optimizer

ZeRO is DeepSpeed's innovation for reducing memory redundancy in data parallelism.

**ZeRO-1: Shard Optimizer States**
```
Without ZeRO:     Each GPU has full optimizer states (O₀, O₁, O₂, O₃)
With ZeRO-1:      GPU 0 has O₀, GPU 1 has O₁, GPU 2 has O₂, GPU 3 has O₃
                  Before optimizer step: all_gather optimizer states
```
Memory saved: (N-1)/N of optimizer states

**ZeRO-2: Shard Optimizer States + Gradients**
```
Without ZeRO:     Each GPU has full gradients (G₀, G₁, G₂, G₃)
With ZeRO-2:      Use reduce_scatter instead of all_reduce
                  Each GPU only keeps 1/N of gradients
```
Memory saved: (N-1)/N of gradients too

**ZeRO-3: Shard Everything (Parameters too)**
```
Without ZeRO:     Each GPU has full model (P₀, P₁, P₂, P₃)
With ZeRO-3:      GPU 0 has P₀, GPU 1 has P₁, etc.
                  Before forward/backward: all_gather needed parameters
```
Memory saved: (N-1)/N of parameters

### Memory Comparison

For a 70B parameter model with 8 GPUs:

| Strategy | Memory per GPU |
|----------|---------------|
| DDP (replicated) | 840 GB |
| ZeRO-1 | 350 GB |
| ZeRO-2 | 210 GB |
| ZeRO-3 | 105 GB |

ZeRO-3 achieves 8x memory reduction!

### FSDP: PyTorch's ZeRO Implementation

Fully Sharded Data Parallel (FSDP) is PyTorch's native implementation of ZeRO-3:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
    # sharding_strategy=ShardingStrategy.NO_SHARD,  # DDP-like
)
```

### Communication Volume Comparison

| Strategy | Forward | Backward | Optimizer |
|----------|---------|----------|-----------|
| DDP | 0 | 2D | 0 |
| ZeRO-1 | 0 | 2D | D |
| ZeRO-2 | 0 | D | D |
| ZeRO-3 | 2D | 2D | D |

Where D = model size, communication is per-GPU.

ZeRO-3 has 3x more communication than DDP, but 8x less memory!

### When to Use What?

| Scenario | Recommendation |
|----------|---------------|
| Model fits in GPU memory | DDP (fastest) |
| Model + gradients fit | ZeRO-2 / FSDP SHARD_GRAD_OP |
| Model doesn't fit | ZeRO-3 / FSDP FULL_SHARD |
| Very large models (100B+) | ZeRO-3 + tensor parallelism |

## Code Walkthrough

### Script 1: simple_ddp.py

A minimal DDP implementation to understand the basics:
- Manual gradient synchronization with all_reduce
- Comparison with automatic DDP wrapper
- Measuring communication overhead

### Script 2: gradient_sync_visualizer.py

Visualize how gradient synchronization works:
- Shows per-parameter gradients before/after sync
- Demonstrates gradient bucketing concept
- Compares sync strategies

## Try It Yourself

### Exercise 1: Manual DDP

Implement data-parallel training without using DDP wrapper:
1. Broadcast initial weights from rank 0
2. After backward(), manually all_reduce all gradients
3. Verify your implementation matches DDP

### Exercise 2: Gradient Bucketing

Modify `gradient_sync_visualizer.py` to bucket gradients:
1. Group gradients into fixed-size buckets
2. all_reduce each bucket as a single tensor
3. Measure if bucketing improves throughput

### Exercise 3: Measure Communication Overhead

Profile a DDP training run:
1. Measure time spent in forward pass
2. Measure time spent in backward pass (includes communication)
3. Calculate communication/computation ratio

## Key Takeaways

1. **DDP is the default choice** - Simple, fast, well-optimized
2. **Gradient averaging is the key insight** - Enables mathematically correct distributed training
3. **Memory is the bottleneck for LLMs** - ZeRO/FSDP trades communication for memory
4. **Choose sharding level based on model size** - Start with DDP, escalate as needed
5. **Communication overhead grows with sharding** - ZeRO-3 is 3x more communication than DDP

## The Efficiency Equation

Throughput ≈ min(Compute Throughput, Memory Bandwidth, Network Bandwidth)

- **Compute bound**: Add more GPUs with DDP
- **Memory bound**: Use ZeRO-3/FSDP
- **Network bound**: Optimize topology, reduce communication

## What's Next?

In Chapter 6, we'll explore **Tensor Parallelism**—splitting individual layers across GPUs. This is how we train layers that are too large for a single GPU even with ZeRO-3.

## Further Reading

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- Original source: [`rlhf/sys-design/readme-2-en.md`](../../../rlhf/sys-design/readme-2-en.md)
