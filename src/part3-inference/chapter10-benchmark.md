# scheduling_overhead_benchmark.py

> Measure and visualize CPU scheduling overhead in inference

This script quantifies how much time is spent on CPU scheduling vs GPU computation.

## What It Does

1. Simulates inference batches of different sizes
2. Measures scheduling time (CPU)
3. Measures compute time (GPU)
4. Shows overhead percentage
5. Demonstrates overlap scheduling benefit

## Run It

```bash
python tutorial/part3-inference/chapter10-scheduling-cuda/scripts/scheduling_overhead_benchmark.py
```

## Example Output

```
=== Scheduling Overhead Benchmark ===

Batch Size | Schedule (ms) | Compute (ms) | Overhead %
-----------|---------------|--------------|------------
    1      |     2.1       |     5.2      |    40%
    4      |     2.3       |     6.1      |    38%
   16      |     2.8       |    12.5      |    22%
   64      |     3.5       |    45.2      |     8%

Observation: Larger batches amortize scheduling overhead.

=== With Overlap Scheduling ===

Batch Size | Effective Overhead
-----------|-------------------
    1      |     5% (hidden)
    4      |     3% (hidden)
   16      |     1% (hidden)
   64      |     0% (hidden)

Overlap scheduling hides CPU work behind GPU compute!
```

## Why Small Batches Are Hard

```
Batch size 1:
  Schedule: [====] 2ms
  Compute:  [========] 5ms
  Total:    7ms for 1 token = 143 tokens/sec

Batch size 64:
  Schedule: [====] 3.5ms
  Compute:  [======================================] 45ms
  Total:    48.5ms for 64 tokens = 1320 tokens/sec
```

Small batches spend more time scheduling than computing!

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter10-scheduling-cuda/scripts/scheduling_overhead_benchmark.py}}
```
