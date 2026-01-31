# benchmark_algorithms.py

> Benchmark NCCL algorithms on your hardware

This script measures all_reduce performance with different algorithms and message sizes, helping you understand your hardware's communication characteristics.

## What It Does

1. Runs all_reduce with various message sizes (1KB to 1GB)
2. Tests different NCCL algorithms (Ring, Tree)
3. Measures throughput (GB/s) and latency (ms)
4. Shows scaling behavior as data size increases

## Run It

```bash
# Default benchmarks
python tutorial/part1-distributed/chapter04-nccl-topology/scripts/benchmark_algorithms.py

# Force specific algorithm
NCCL_ALGO=Ring python tutorial/part1-distributed/chapter04-nccl-topology/scripts/benchmark_algorithms.py
```

## Example Output

```
=== All-Reduce Benchmark ===

Message Size | Latency (ms) | Throughput (GB/s) | Algorithm
-------------|--------------|-------------------|----------
     1 KB    |     0.05     |       0.02        |   Tree
    16 KB    |     0.06     |       0.27        |   Tree
   256 KB    |     0.12     |       2.13        |   Ring
     4 MB    |     0.89     |       4.49        |   Ring
    64 MB    |    12.50     |       5.12        |   Ring
     1 GB    |   198.00     |       5.05        |   Ring

Observations:
- Tree wins for small messages (< 256 KB): lower latency
- Ring wins for large messages (> 256 KB): better bandwidth
- Peak throughput: 5.12 GB/s (limited by PCIe)
```

## Interpreting Results

**Latency-bound** (small messages):
- Tree algorithm is better
- Dominated by startup overhead
- Actual data transfer is fast

**Bandwidth-bound** (large messages):
- Ring algorithm is better
- Near-100% bandwidth utilization
- All GPUs sending/receiving simultaneously

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter04-nccl-topology/scripts/benchmark_algorithms.py}}
```
