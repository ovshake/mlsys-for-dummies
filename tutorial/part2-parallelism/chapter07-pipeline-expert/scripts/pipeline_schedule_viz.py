#!/usr/bin/env python3
"""
Pipeline Schedule Visualizer

This script visualizes different pipeline parallelism scheduling strategies:
- Naive (fill-drain): Simple but high bubble overhead
- 1F1B: Interleaved forward/backward for lower memory
- Shows bubble ratios and GPU utilization

Usage:
    python pipeline_schedule_viz.py
    python pipeline_schedule_viz.py --stages 4 --microbatches 8
"""

import argparse
from typing import List, Tuple


def visualize_naive_schedule(stages: int, microbatches: int) -> Tuple[str, float]:
    """
    Visualize naive fill-drain pipeline schedule.

    In naive scheduling:
    1. Forward all microbatches through the pipeline
    2. Then backward all microbatches

    This leads to large bubbles at start and end.
    """
    # Calculate timeline
    # Forward: stage i starts at time i
    # Backward: starts after all forwards complete

    total_forward_time = microbatches + stages - 1
    total_backward_time = microbatches + stages - 1
    total_time = total_forward_time + total_backward_time

    # Build visualization
    lines = []
    for stage in range(stages):
        line = ["."] * total_time

        # Forward passes
        for mb in range(microbatches):
            t = stage + mb
            line[t] = f"F{mb}"

        # Backward passes (start after all forwards)
        for mb in range(microbatches):
            t = total_forward_time + (stages - 1 - stage) + mb
            line[t] = f"B{mb}"

        lines.append(line)

    # Calculate bubble ratio
    work_per_stage = 2 * microbatches  # F + B for each microbatch
    bubble_per_stage = total_time - work_per_stage
    bubble_ratio = (stages - 1) * microbatches / (stages * microbatches + (stages - 1))

    # Format output
    output = []
    output.append(f"Time →  " + "".join([f"{i:>3}" for i in range(total_time)]))
    output.append("-" * (8 + total_time * 3))

    for stage, line in enumerate(lines):
        formatted = "".join([f"{x:>3}" for x in line])
        output.append(f"GPU {stage}:  {formatted}")

    return "\n".join(output), bubble_ratio


def visualize_1f1b_schedule(stages: int, microbatches: int) -> Tuple[str, float, int]:
    """
    Visualize 1F1B (One Forward, One Backward) pipeline schedule.

    Key insight: After warmup, each stage does 1F then 1B,
    keeping activation memory bounded.

    Memory = max(warmup_microbatches) = stages
    """
    # Timeline representation
    total_time = 2 * microbatches + 2 * (stages - 1)

    lines = []
    for stage in range(stages):
        line = ["."] * total_time
        warmup_steps = stages - stage  # First stages need more warmup

        # Warmup phase: only forwards
        for mb in range(warmup_steps):
            t = stage + mb
            if t < total_time:
                line[t] = f"F{mb}"

        # Steady state: 1F1B
        for mb in range(warmup_steps, microbatches):
            # Forward at position
            f_time = stage + mb
            if f_time < total_time:
                line[f_time] = f"F{mb}"

            # Backward at position (for earlier microbatch)
            b_mb = mb - warmup_steps
            b_time = f_time + 1 if stage == stages - 1 else f_time + 2
            if b_time < total_time:
                line[b_time] = f"B{b_mb}"

        # Cooldown: remaining backwards
        cooldown_start = microbatches - warmup_steps
        for i, mb in enumerate(range(cooldown_start, microbatches)):
            b_time = stages + microbatches - 1 + stage + i
            if b_time < total_time:
                line[b_time] = f"B{mb}"

        lines.append(line)

    # Simplified bubble calculation
    work_units = 2 * microbatches
    bubble_ratio = (stages - 1) / (microbatches + stages - 1)
    peak_memory = stages  # Peak number of activations stored

    # Format output
    output = []
    output.append(f"Time →  " + "".join([f"{i:>3}" for i in range(min(total_time, 25))]))
    output.append("-" * (8 + min(total_time, 25) * 3))

    for stage, line in enumerate(lines):
        formatted = "".join([f"{x:>3}" for x in line[:25]])
        output.append(f"GPU {stage}:  {formatted}" + ("..." if total_time > 25 else ""))

    return "\n".join(output), bubble_ratio, peak_memory


def analyze_schedules(stages: int, microbatches: int) -> None:
    """Compare different scheduling strategies."""
    print("=" * 70)
    print(" PIPELINE SCHEDULE COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration: {stages} stages, {microbatches} microbatches\n")

    # Naive schedule
    print("-" * 70)
    print(" NAIVE (Fill-Drain) SCHEDULE")
    print("-" * 70)
    print("""
Strategy: Complete all forwards, then all backwards.
Memory: Must store activations for ALL microbatches.
""")
    naive_viz, naive_bubble = visualize_naive_schedule(stages, microbatches)
    print(naive_viz)
    print(f"\nBubble ratio: {naive_bubble:.1%}")
    print(f"Peak activation memory: {microbatches} microbatches worth")

    print("\n")

    # 1F1B schedule
    print("-" * 70)
    print(" 1F1B (One Forward, One Backward) SCHEDULE")
    print("-" * 70)
    print("""
Strategy: After warmup, alternate 1 forward then 1 backward.
Memory: Only store activations for 'stages' microbatches.
""")
    fb_viz, fb_bubble, fb_memory = visualize_1f1b_schedule(stages, microbatches)
    print(fb_viz)
    print(f"\nBubble ratio: {fb_bubble:.1%}")
    print(f"Peak activation memory: {fb_memory} microbatches worth")

    # Comparison
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
{'Metric':<25} {'Naive':<20} {'1F1B':<20}
{'-'*65}
{'Bubble ratio':<25} {naive_bubble:.1%:<20} {fb_bubble:.1%:<20}
{'Peak memory':<25} {microbatches:<20} {fb_memory:<20}

Key insights:
1. 1F1B has the SAME bubble ratio but LOWER memory
2. More microbatches → lower bubble ratio (approaches 0 as M→∞)
3. Peak memory in 1F1B is bounded by pipeline depth
""")


def demonstrate_bubble_reduction() -> None:
    """Show how bubble ratio decreases with more microbatches."""
    print("\n" + "=" * 70)
    print(" BUBBLE RATIO vs MICROBATCHES")
    print("=" * 70)
    print("""
Bubble ratio = (P-1) / (M + P - 1)

Where P = pipeline stages, M = microbatches
""")

    stages = 4
    print(f"For P = {stages} stages:\n")
    print(f"{'Microbatches':<15} {'Bubble Ratio':<15} {'Efficiency':<15}")
    print("-" * 45)

    for mb in [1, 2, 4, 8, 16, 32, 64]:
        bubble = (stages - 1) / (mb + stages - 1)
        efficiency = 1 - bubble
        print(f"{mb:<15} {bubble:.1%:<15} {efficiency:.1%:<15}")

    print("""
Takeaway: Use at least 4x pipeline stages as microbatches
          for > 80% efficiency.
""")


def explain_memory_tradeoff() -> None:
    """Explain the memory-throughput tradeoff."""
    print("\n" + "=" * 70)
    print(" MEMORY vs THROUGHPUT TRADEOFF")
    print("=" * 70)
    print("""
The fundamental tradeoff in pipeline parallelism:

MORE MICROBATCHES:
  ✓ Lower bubble ratio (better throughput)
  ✗ More activation memory (naive) or same (1F1B)
  ✗ Smaller per-microbatch batch size (worse GPU utilization)

FEWER MICROBATCHES:
  ✗ Higher bubble ratio (worse throughput)
  ✓ Less activation memory
  ✓ Larger per-microbatch batch size (better GPU utilization)

1F1B ADVANTAGE:
  With 1F1B, memory is bounded by pipeline depth, NOT microbatches.
  This allows many microbatches for low bubbles without memory explosion.

Example calculation:
  Model: 24 layers, 4096 hidden dim, batch 512
  Pipeline: 4 stages (6 layers each)
  Microbatches: 16 (32 samples each)

  Naive memory: 16 × activations ≈ 16 × 32 × 4096 × 6 = 12.6 GB per stage
  1F1B memory:   4 × activations ≈  4 × 32 × 4096 × 6 =  3.1 GB per stage

  4x memory reduction!
""")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Schedule Visualizer")
    parser.add_argument("--stages", "-s", type=int, default=4,
                        help="Number of pipeline stages")
    parser.add_argument("--microbatches", "-m", type=int, default=8,
                        help="Number of microbatches")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " PIPELINE PARALLELISM SCHEDULE VISUALIZER".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    analyze_schedules(args.stages, args.microbatches)
    demonstrate_bubble_reduction()
    explain_memory_tradeoff()


if __name__ == "__main__":
    main()
