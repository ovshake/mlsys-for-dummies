# pipeline_schedule_viz.py

> Visualize pipeline scheduling strategies and understand bubbles

This script creates ASCII visualizations of different pipeline scheduling algorithms, showing how they affect GPU utilization and memory usage.

## What It Does

1. Visualizes **Naive (Fill-Drain)** scheduling - shows massive bubbles
2. Visualizes **1F1B** scheduling - shows reduced bubbles
3. Calculates bubble fraction for each approach
4. Compares memory requirements

## Run It

```bash
python tutorial/part2-parallelism/chapter07-pipeline-expert/scripts/pipeline_schedule_viz.py
```

## Example Output

```
=== Naive Fill-Drain Schedule (P=4, M=8) ===

Time →
GPU 0: [F0][F1][F2][F3][F4][F5][F6][F7][  ][  ][  ][B7][B6][B5][B4][B3][B2][B1][B0]
GPU 1:    [F0][F1][F2][F3][F4][F5][F6][F7][  ][  ][B7][B6][B5][B4][B3][B2][B1][B0]
GPU 2:       [F0][F1][F2][F3][F4][F5][F6][F7][  ][B7][B6][B5][B4][B3][B2][B1][B0]
GPU 3:          [F0][F1][F2][F3][F4][F5][F6][F7][B7][B6][B5][B4][B3][B2][B1][B0]

Bubble fraction: 27% (3 slots idle per GPU out of 11)
Peak memory: 8 microbatches of activations

=== 1F1B Schedule (P=4, M=8) ===

Time →
GPU 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
GPU 1:    [F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][F6][B4][F7][B5][B6][B7]
GPU 2:       [F0][F1][B0][F2][B1][F3][B2][F4][B3][F5][B4][F6][B5][F7][B6][B7]
GPU 3:          [F0][B0][F1][B1][F2][B2][F3][B3][F4][B4][F5][B5][F6][B6][F7][B7]

Bubble fraction: 19%
Peak memory: 4 microbatches of activations (= P, not M!)
```

## Key Insight

1F1B achieves:
- **Lower bubble fraction** by interleaving forward/backward
- **Constant memory** by releasing activations as soon as backward is done

## Source Code

```python
{{#include ../../tutorial/part2-parallelism/chapter07-pipeline-expert/scripts/pipeline_schedule_viz.py}}
```
