# Chapter 4: NCCL Algorithms and GPU Topology

> *"Understanding your hardware is half the battle. The other half is making NCCL do what you want."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain how Ring and Tree algorithms work for all_reduce
- Inspect GPU topology and NVLink connections
- Understand why communication patterns matter for performance
- Choose the right NCCL settings for your hardware

## Prerequisites

- Completed [Chapters 1-3](./chapter01.md)
- Understanding of all_reduce and collective operations
- Access to a machine with NVIDIA GPU (for hands-on topology inspection)

## Concept Overview

### Why Does the Algorithm Matter?

When you call `dist.all_reduce(tensor)`, NCCL doesn't just magically synchronize data. It runs a carefully designed algorithm that determines:
1. **Who sends to whom** - The communication pattern
2. **What data flows** - Partial aggregates vs full tensors
3. **How much bandwidth is used** - Network saturation
4. **How long it takes** - Latency characteristics

Different algorithms excel in different scenarios:
- **Ring**: Great bandwidth utilization, scales with data size
- **Tree**: Lower latency for small messages, scales better with node count
- **Double Binary Tree**: Best of both worlds for large clusters

### The Ring Algorithm

Ring is the most intuitive collective algorithm. Picture GPUs arranged in a circle:

```
        ┌──────► GPU 1 ──────┐
        │                    │
        │                    ▼
     GPU 0                 GPU 2
        ▲                    │
        │                    │
        └────── GPU 3 ◄──────┘
```

**How Ring All-Reduce Works:**

Phase 1: **Scatter-Reduce** (each GPU accumulates partial sums)
```
Step 1: GPU0 sends chunk0 to GPU1, GPU1 sends chunk1 to GPU2, ...
Step 2: Recipients add their local chunk to received chunk, send result
... (N-1 steps total)
```

Phase 2: **All-Gather** (distribute the fully reduced chunks)
```
Step 1: GPU0 sends its complete chunk0 to GPU1, ...
... (N-1 steps total)
```

**Ring Complexity:**
- Total steps: 2(N-1) where N is number of GPUs
- Data per step: D/N where D is total data size
- Total data moved: 2D(N-1)/N ≈ 2D for large N

**Ring's Superpower**: Bandwidth utilization is nearly 100%! Each GPU is always sending and receiving.

### The Tree Algorithm

For large clusters, Ring's latency (O(N) steps) becomes problematic. Tree algorithms use a hierarchical structure:

```
              GPU 0 (root)
             /          \
          GPU 1        GPU 2
         /    \       /    \
      GPU 3  GPU 4  GPU 5  GPU 6
```

**How Tree Reduce Works:**

```
Step 1: Leaves (3,4,5,6) send to parents (1,2)
Step 2: Parents combine, send to root (0)
Step 3: Root has final result
```

**Tree Complexity:**
- Total steps: 2 * log2(N) (reduce up + broadcast down)
- Much better latency for small messages

**Tree's Tradeoff**: Lower bandwidth utilization (only half the GPUs active at any time).

### Double Binary Tree (for 24,000+ GPUs)

At scale (think training GPT-4), even tree algorithms hit bottlenecks. Double Binary Tree uses two complementary trees to keep all links busy:

```
Tree A:                     Tree B:
    0                           7
   / \                         / \
  1   2                       6   5
 / \ / \                     / \ / \
3  4 5  6                   0  1 2  3
```

Different GPUs are roots/leaves in each tree, balancing the load.

### NVLink: The Speed Demon

NVLink is NVIDIA's high-bandwidth interconnect for GPU-to-GPU communication:

| Generation | Bandwidth (per link) | Links per GPU |
|------------|---------------------|---------------|
| NVLink 1.0 | 20 GB/s | 4 |
| NVLink 2.0 | 25 GB/s | 6 |
| NVLink 3.0 | 25 GB/s | 12 |
| NVLink 4.0 | 25 GB/s | 18 |

For comparison, PCIe 4.0 x16 is only ~32 GB/s total!

A fully-connected 8-GPU node with NVLink 4.0 has 900 GB/s aggregate bandwidth between GPUs. This is why DGX systems are so fast for training.

### GPU Topology: The Key to Understanding Performance

Not all GPU pairs are connected equally! Use `nvidia-smi topo -m` to see your topology:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity
GPU0     X      NV4     NV4     NV4     0-31
GPU1    NV4      X      NV4     NV4     0-31
GPU2    NV4     NV4      X      NV4     0-31
GPU3    NV4     NV4     NV4      X      0-31
```

Legend:
- **X**: Self
- **NV#**: NVLink with # links
- **SYS**: Cross NUMA node (slowest)
- **NODE**: Same NUMA node, no NVLink
- **PHB**: Same PCIe host bridge
- **PXB**: Different PCIe bridges
- **PIX**: Same PCIe switch

**Rule of thumb**: More NVLinks = faster. SYS = slow, avoid if possible.

## Code Walkthrough

### Script 1: topology_inspector.py

This script inspects your GPU topology and reports:
- How many GPUs you have
- NVLink connections between GPUs
- PCIe topology
- NUMA affinity

It also suggests optimal process placement.

### Script 2: benchmark_algorithms.py

This script benchmarks different NCCL algorithms on your hardware:
- Measures all_reduce throughput
- Compares Ring vs Tree
- Shows how performance scales with message size

## NCCL Environment Variables

You can tune NCCL behavior with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NCCL_ALGO` | Algorithm: Ring, Tree, CollNetChain | Auto |
| `NCCL_PROTO` | Protocol: Simple, LL, LL128 | Auto |
| `NCCL_NTHREADS` | Threads per block | Auto |
| `NCCL_DEBUG` | Debugging output (WARN, INFO, TRACE) | WARN |
| `NCCL_DEBUG_SUBSYS` | Subsystems to debug | All |

Example: Force ring algorithm and show debug info:
```bash
NCCL_ALGO=Ring NCCL_DEBUG=INFO python train.py
```

## Try It Yourself

### Exercise 1: Inspect Your Topology

Run `topology_inspector.py` on a GPU machine and answer:
1. How many NVLinks connect GPU 0 to GPU 1?
2. Are any GPU pairs connected only via PCIe?
3. What's the CPU affinity for each GPU?

### Exercise 2: Benchmark All-Reduce

Run `benchmark_algorithms.py` with different message sizes:
- 1 KB
- 1 MB
- 100 MB

When does Ring outperform Tree? When does Tree win?

### Exercise 3: Measure the NVLink Advantage

If you have GPUs connected via NVLink AND PCIe:
1. Run all_reduce between NVLink-connected GPUs
2. Run all_reduce between PCIe-connected GPUs
3. Calculate the speedup

## Key Takeaways

1. **Ring excels at large messages** - Nearly 100% bandwidth utilization
2. **Tree excels at low latency** - O(log N) steps vs O(N)
3. **NVLink is crucial** - 10x+ faster than PCIe
4. **Topology determines performance** - Know your hardware!
5. **NCCL auto-selects** - But you can override for specific cases

## Performance Intuition

For a 1 GB all_reduce on 8 GPUs:

| Connection | Ring Bandwidth | Approximate Time |
|------------|---------------|------------------|
| NVLink 4.0 (900 GB/s) | ~450 GB/s effective | ~2.2 ms |
| PCIe 4.0 x16 (32 GB/s) | ~16 GB/s effective | ~62 ms |

That's a **28x difference** just from interconnect!

## What's Next?

In Part II, we'll use these primitives to implement actual parallelism strategies:
- [Chapter 5](../part2-parallelism/chapter05.md): Data Parallelism (DDP, FSDP, ZeRO)
- [Chapter 6](../part2-parallelism/chapter06.md): Tensor Parallelism (splitting layers)
- [Chapter 7](../part2-parallelism/chapter07.md): Pipeline Parallelism (splitting models)

## Further Reading

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVLink Architecture](https://www.nvidia.com/en-us/data-center/nvlink/)
