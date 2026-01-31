# Chapter 10: Advanced Scheduling and CUDA Graphs

> *"The fastest code is code that doesn't run. The second fastest is code you ran yesterday."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain why CPU scheduling overhead matters for inference
- Understand CUDA Graphs and when to use them
- Describe zero-overhead scheduling with FutureMap
- Identify scenarios where CUDA Graphs help vs hurt

## Prerequisites

- Completed Chapter 8-9 (Server Anatomy, KV Cache)
- Basic understanding of CPU-GPU interaction
- Familiarity with asynchronous programming

## Concept Overview

### The Scheduling Overhead Problem

LLM inference involves many small operations:
1. Schedule which requests to process
2. Prepare batch metadata
3. Launch GPU kernels
4. Wait for results
5. Process outputs

The problem? Steps 1, 2, and 5 happen on CPU while GPU waits!

```
Traditional scheduling:

Time →
CPU:  [Schedule][Prepare]............[Process]............[Schedule][Prepare]
GPU:  ............[Compute]............[idle].............[Compute]..........
                     ↑                    ↑
              GPU working           GPU waiting for CPU!
```

For small decode batches, CPU overhead can exceed GPU compute time.

### How Bad Is It?

On a high-end setup (H100 + fast CPU):
- GPU decode step: ~5-10ms
- CPU scheduling overhead: ~2-5ms

That's 20-50% overhead! For latency-sensitive applications, this is unacceptable.

### CUDA Graphs: Recording Once, Playing Many

CUDA Graphs capture a sequence of GPU operations and replay them with minimal CPU overhead.

```python
# Traditional approach (CPU launches each kernel)
for i in range(1000):
    output = model(input)  # CPU launches kernels each time

# CUDA Graph approach
# Step 1: Capture
with torch.cuda.graph(graph):
    output = model(input)  # Operations recorded, not executed

# Step 2: Replay
for i in range(1000):
    graph.replay()  # Single CPU call replays entire graph
```

**Why it's fast:**
- One CPU→GPU launch instead of many
- GPU executes pre-optimized kernel sequence
- No kernel launch latency per operation

### CUDA Graphs: The Constraints

CUDA Graphs require **static computation**:

| Allowed | Not Allowed |
|---------|-------------|
| Fixed tensor shapes | Dynamic shapes |
| Deterministic operations | Random dropout |
| Pre-allocated memory | Dynamic allocation |
| Fixed control flow | Data-dependent branching |

This is perfect for inference (fixed model) but problematic for training.

### Why Training Rarely Uses CUDA Graphs

1. **Dynamic optimizer updates**: Gradient clipping changes behavior
2. **Learning rate scheduling**: Different computations each step
3. **Gradient accumulation**: Variable number of backwards
4. **Dropout**: Random behavior
5. **Dynamic memory**: Activation checkpointing allocates/frees

### Zero-Overhead Scheduling: The FutureMap

SGLang's innovation: **overlap CPU scheduling with GPU compute**.

Traditional:
```
Batch N: [CPU Schedule N] → [GPU Compute N] → [CPU Process N]
                                              ↓
Batch N+1:                                   [CPU Schedule N+1] → [GPU Compute N+1]
```

Overlapped:
```
Batch N:   [CPU Schedule N] → [GPU Compute N] ────────────────→
                               ↓
Batch N+1:                    [CPU Schedule N+1] → [GPU Compute N+1]
                               ↑
                               Running in parallel!
```

**The challenge**: Batch N+1's inputs might depend on Batch N's outputs!

### FutureMap: Speculative Scheduling

FutureMap solves this with **symbolic references**:

```
1. CPU pre-allocates slots for Batch N's output
2. CPU schedules Batch N+1 using slot references (not actual values)
3. GPU runs Batch N, writes to pre-allocated slots
4. GPU's "resolve" kernel substitutes symbolic refs with real values
5. GPU runs Batch N+1
```

```
┌────────────────────────────────────────────────────────────────────┐
│ FutureMap Mechanism                                                 │
│                                                                    │
│  CPU Thread:                                                        │
│    1. Reserve slots for Batch N output                             │
│    2. Build Batch N+1 input with symbolic refs: [slot_42, slot_43] │
│    3. Continue scheduling (no blocking!)                           │
│                                                                    │
│  GPU Thread:                                                        │
│    1. Compute Batch N                                              │
│    2. Write results to reserved slots (42, 43)                     │
│    3. Resolve kernel: [slot_42, slot_43] → [actual_token_ids]      │
│    4. Compute Batch N+1                                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### The Complete Overlap Scheduler

```python
async def overlap_scheduler_loop(self):
    """SGLang's overlapped scheduling loop."""
    last_batch = None
    last_result = None

    while True:
        # Step 1: Schedule NEXT batch (CPU)
        # This happens WHILE previous batch is computing!
        next_batch = self.get_next_batch_to_run()

        # Step 2: Launch next batch (GPU, non-blocking)
        next_result = self.run_batch(next_batch)

        # Step 3: Process PREVIOUS batch results (CPU)
        # By now, previous batch is likely done
        if last_batch is not None:
            self.process_batch_result(last_batch, last_result)

        last_batch = next_batch
        last_result = next_result
```

### Combining CUDA Graphs with Overlap Scheduling

The ultimate optimization:
1. **CUDA Graphs** for decode batches (fixed shape, repeated)
2. **Overlap scheduling** for prefill/mixed batches
3. **FutureMap** to bridge the gap

```
Decode path (CUDA Graph):
  - Capture graph for batch sizes [1, 2, 4, 8, 16, ...]
  - Replay appropriate graph based on batch size
  - Near-zero CPU overhead

Prefill path (Overlap):
  - Variable prompt lengths
  - Use overlap scheduling with FutureMap
  - Reduced but not eliminated CPU overhead
```

## Code Walkthrough

### Script 1: cuda_graph_simple.py

Demonstrates CUDA Graphs:
- Captures a simple model forward pass
- Compares replay vs normal execution
- Shows the constraints

### Script 2: scheduling_overhead_benchmark.py

Measures scheduling overhead:
- Time breakdown: scheduling vs compute
- Impact of batch size
- Benefits of overlap scheduling

## Try It Yourself

### Exercise 1: Measure Kernel Launch Overhead

Write a benchmark that:
1. Runs 100 small matrix multiplications normally
2. Captures them in a CUDA Graph
3. Compares total time

### Exercise 2: Understand Shape Constraints

Try to capture a CUDA Graph with:
1. Fixed input shape → works
2. Different input shapes → observe behavior
3. How do real systems handle multiple shapes?

### Exercise 3: Simulate Overlap Scheduling

Implement a simple overlap scheduler:
1. Queue of "batches" (just sleep timers)
2. Measure throughput with vs without overlap
3. What's the speedup?

## Key Takeaways

1. **CPU overhead is real** - Can be 20-50% of decode time
2. **CUDA Graphs eliminate kernel launch overhead** - But need static shapes
3. **Overlap scheduling hides CPU work** - Schedule N+1 while computing N
4. **FutureMap enables speculation** - Pre-allocate outputs, resolve later
5. **Real systems combine techniques** - CUDA Graphs for decode, overlap for prefill

## The Speed Hierarchy

From fastest to slowest:
1. **CUDA Graph replay**: ~0.01ms overhead
2. **Overlap scheduled**: ~0.5ms (hidden)
3. **Normal scheduling**: ~2-5ms
4. **Naive Python loop**: ~10ms+

## When Not to Use CUDA Graphs

- Variable sequence lengths (prefill)
- Dynamic batch sizes (requests finishing)
- Debugging (graphs hide errors)
- Memory-constrained (graphs consume memory)

## What's Next?

In Chapter 11, we'll explore **Speculative and Constraint Decoding**—using draft models to speed up generation and grammar constraints to ensure structured output.

## Further Reading

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [SGLang Zero-Overhead Scheduling](https://arxiv.org/abs/2312.07104)
- Original source: [`torch/cuda-graph/readme_en.md`](../../../torch/cuda-graph/readme_en.md)
