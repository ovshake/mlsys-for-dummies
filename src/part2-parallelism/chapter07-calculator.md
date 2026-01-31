# parallel_strategy_calculator.py

> Calculate optimal parallelism configuration for your model and hardware

This script helps you design a parallelism strategy by calculating memory requirements, communication volume, and efficiency for different configurations.

## What It Does

1. Takes model specifications (parameters, layers, hidden size)
2. Takes hardware specifications (GPU memory, count, interconnect)
3. Calculates memory per GPU for each parallelism strategy
4. Recommends optimal configuration

## Run It

```bash
python tutorial/part2-parallelism/chapter07-pipeline-expert/scripts/parallel_strategy_calculator.py
```

## Example Usage

```
=== Parallelism Strategy Calculator ===

Model: 70B parameters, 80 layers, hidden=8192
Hardware: 64 GPUs (8 per node), 80GB each, NVLink intra-node

Configuration options:

| Strategy      | TP | PP | DP | Memory/GPU | Comm Volume | Feasible? |
|---------------|----|----|----| -----------|-------------|-----------|
| Pure DP       | 1  | 1  | 64 | 840 GB     | 2D/step     | No        |
| TP=8, DP=8    | 8  | 1  | 8  | 105 GB     | 8D/layer    | No        |
| TP=8, PP=2    | 8  | 2  | 4  | 52 GB      | 8D/layer    | Yes       |
| TP=8, PP=4    | 8  | 4  | 2  | 26 GB      | 8D/layer    | Yes (rec) |

Recommended: TP=8 (within node), PP=4 (across nodes), DP=2

Reasoning:
- TP=8 uses NVLink bandwidth efficiently
- PP=4 distributes 80 layers across 4 stages (20 layers each)
- DP=2 provides batch parallelism for throughput
```

## Input Parameters

The calculator considers:
- **Model**: Parameters, layers, hidden dimension, precision
- **Hardware**: GPU count, memory per GPU, interconnect bandwidth
- **Training**: Batch size, sequence length, microbatch count

## What It Calculates

For each configuration:
- **Memory per GPU**: Parameters + gradients + optimizer + activations
- **Communication volume**: Per-step all_reduce/all_gather/send-recv
- **Bubble fraction**: For pipeline configurations
- **Feasibility**: Does it fit in GPU memory?

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter07-pipeline-expert/scripts/parallel_strategy_calculator.py}}
```
