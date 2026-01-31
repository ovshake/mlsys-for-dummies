# cuda_graph_simple.py

> Understand CUDA Graphs with a simple example

This script demonstrates how CUDA Graphs work by capturing and replaying a simple computation.

## What It Does

1. Creates a simple model (matrix multiplications)
2. Runs it normally (CPU launches each kernel)
3. Captures it as a CUDA Graph
4. Replays the graph (single launch)
5. Compares performance

## Run It

```bash
python tutorial/part3-inference/chapter10-scheduling-cuda/scripts/cuda_graph_simple.py
```

## Example Output

```
=== CUDA Graph Demo ===

Model: 3-layer MLP (1024 → 1024 → 1024 → 1024)

Normal execution (100 iterations):
  Total time: 15.2 ms
  Per iteration: 0.152 ms

CUDA Graph execution (100 iterations):
  Capture time: 0.5 ms (one-time)
  Total replay time: 3.1 ms
  Per iteration: 0.031 ms

Speedup: 4.9x faster with CUDA Graphs!

Reason: Normal execution has ~0.12ms kernel launch overhead per iteration.
CUDA Graphs amortize this to near-zero.
```

## Key Concepts

**Capture Phase:**
```python
# Record operations into a graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)  # NOT executed, just recorded
```

**Replay Phase:**
```python
# Execute the recorded graph
for _ in range(100):
    g.replay()  # Single CPU→GPU call, entire sequence runs
```

## The Constraint

CUDA Graphs need FIXED shapes. This doesn't work:

```python
# ERROR: Shape changes between iterations
for i in range(10):
    input = torch.randn(i + 1, 1024)  # Different size each time!
    output = model(input)
```

Real systems capture graphs for common shapes: [1, 2, 4, 8, 16, ...] batch sizes.

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter10-scheduling-cuda/scripts/cuda_graph_simple.py}}
```
