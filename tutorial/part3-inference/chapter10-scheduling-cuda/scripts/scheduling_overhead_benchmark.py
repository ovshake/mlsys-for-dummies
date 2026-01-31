#!/usr/bin/env python3
"""
Scheduling Overhead Benchmark

This script demonstrates the CPU scheduling overhead in LLM inference
and shows how overlap scheduling reduces it.

Usage:
    python scheduling_overhead_benchmark.py
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    scheduling_time_ms: float
    compute_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float


class SchedulingOverheadBenchmark:
    """
    Benchmark to measure and compare scheduling strategies.

    Simulates:
    - CPU scheduling overhead (preparing batches)
    - GPU compute time (actual model execution)
    - CPU postprocessing (handling results)
    """

    def __init__(self, scheduling_ms: float = 2.0, compute_ms: float = 10.0,
                 postprocess_ms: float = 1.0):
        self.scheduling_ms = scheduling_ms
        self.compute_ms = compute_ms
        self.postprocess_ms = postprocess_ms

    def _simulate_scheduling(self) -> float:
        """Simulate CPU scheduling work."""
        start = time.perf_counter()
        # Simulate work: preparing batch metadata, allocating, etc.
        time.sleep(self.scheduling_ms / 1000)
        return (time.perf_counter() - start) * 1000

    def _simulate_compute(self) -> float:
        """Simulate GPU compute time."""
        start = time.perf_counter()
        time.sleep(self.compute_ms / 1000)
        return (time.perf_counter() - start) * 1000

    def _simulate_postprocess(self) -> float:
        """Simulate result postprocessing."""
        start = time.perf_counter()
        time.sleep(self.postprocess_ms / 1000)
        return (time.perf_counter() - start) * 1000

    def run_sequential(self, num_batches: int) -> Tuple[float, List[BatchMetrics]]:
        """
        Run batches sequentially (traditional approach).

        Timeline: [Schedule] -> [Compute] -> [Postprocess] -> [Schedule] -> ...
        """
        metrics = []
        total_start = time.perf_counter()

        for _ in range(num_batches):
            batch_start = time.perf_counter()

            sched_time = self._simulate_scheduling()
            compute_time = self._simulate_compute()
            post_time = self._simulate_postprocess()

            batch_total = (time.perf_counter() - batch_start) * 1000

            metrics.append(BatchMetrics(
                scheduling_time_ms=sched_time,
                compute_time_ms=compute_time,
                postprocess_time_ms=post_time,
                total_time_ms=batch_total
            ))

        total_time = (time.perf_counter() - total_start) * 1000
        return total_time, metrics

    async def run_overlapped(self, num_batches: int) -> Tuple[float, List[BatchMetrics]]:
        """
        Run batches with overlap scheduling.

        Key insight: Schedule batch N+1 while batch N is computing.

        Timeline (overlapped):
        [Schedule 0] -> [Compute 0]
                        [Schedule 1] -> [Compute 1]
                                        [Postprocess 0]
                                        [Schedule 2] -> ...
        """
        metrics = []
        total_start = time.perf_counter()

        # Pipeline: we overlap scheduling with previous compute
        for i in range(num_batches):
            batch_start = time.perf_counter()

            if i == 0:
                # First batch: no overlap possible
                sched_time = self._simulate_scheduling()
                compute_time = self._simulate_compute()
                post_time = self._simulate_postprocess()
            else:
                # Subsequent batches: scheduling was done during previous compute
                sched_time = 0  # Already done (overlapped)

                # But we still need to do scheduling for NEXT batch
                # This runs in "parallel" with compute
                compute_start = time.perf_counter()

                # Simulate both happening together
                # In reality, GPU computes while CPU schedules
                # Here we take max of the two times
                parallel_time = max(self.compute_ms, self.scheduling_ms) / 1000
                time.sleep(parallel_time)

                compute_time = (time.perf_counter() - compute_start) * 1000
                post_time = self._simulate_postprocess()

            batch_total = (time.perf_counter() - batch_start) * 1000

            metrics.append(BatchMetrics(
                scheduling_time_ms=sched_time,
                compute_time_ms=compute_time,
                postprocess_time_ms=post_time,
                total_time_ms=batch_total
            ))

        total_time = (time.perf_counter() - total_start) * 1000
        return total_time, metrics


def print_results(name: str, total_time: float, metrics: List[BatchMetrics],
                  num_batches: int):
    """Print benchmark results."""
    print(f"\n{name}")
    print("-" * 50)

    avg_sched = sum(m.scheduling_time_ms for m in metrics) / len(metrics)
    avg_compute = sum(m.compute_time_ms for m in metrics) / len(metrics)
    avg_post = sum(m.postprocess_time_ms for m in metrics) / len(metrics)
    avg_total = sum(m.total_time_ms for m in metrics) / len(metrics)

    print(f"Total time: {total_time:.2f} ms")
    print(f"Throughput: {num_batches / (total_time / 1000):.2f} batches/sec")
    print(f"\nPer-batch breakdown:")
    print(f"  Scheduling: {avg_sched:.2f} ms")
    print(f"  Compute: {avg_compute:.2f} ms")
    print(f"  Postprocess: {avg_post:.2f} ms")
    print(f"  Total: {avg_total:.2f} ms")

    overhead_pct = (avg_sched + avg_post) / avg_total * 100
    print(f"\nCPU overhead: {overhead_pct:.1f}%")


def visualize_timeline(scheduling_ms: float, compute_ms: float,
                       postprocess_ms: float, num_batches: int = 4):
    """Visualize the scheduling timeline."""
    print("\n" + "=" * 70)
    print(" TIMELINE VISUALIZATION")
    print("=" * 70)

    scale = 2  # Characters per ms

    def bar(char: str, ms: float) -> str:
        return char * int(ms * scale)

    print("\nSEQUENTIAL EXECUTION:")
    print("  S = Schedule, C = Compute, P = Postprocess, . = idle\n")

    cpu_line = ""
    gpu_line = ""

    for i in range(num_batches):
        # CPU: schedule, then idle, then postprocess
        cpu_line += bar('S', scheduling_ms)
        cpu_line += bar('.', compute_ms)
        cpu_line += bar('P', postprocess_ms)

        # GPU: idle, then compute, then idle
        gpu_line += bar('.', scheduling_ms)
        gpu_line += bar('C', compute_ms)
        gpu_line += bar('.', postprocess_ms)

    print(f"  CPU: {cpu_line}")
    print(f"  GPU: {gpu_line}")

    print("\nOVERLAPPED EXECUTION:")
    print("  CPU schedules batch N+1 while GPU computes batch N\n")

    cpu_line = ""
    gpu_line = ""

    for i in range(num_batches):
        if i == 0:
            # First batch: no overlap
            cpu_line += bar('S', scheduling_ms)
            gpu_line += bar('.', scheduling_ms)

        # Overlap: CPU schedules next while GPU computes
        overlap_time = max(compute_ms, scheduling_ms)
        if scheduling_ms <= compute_ms:
            cpu_line += bar('S', scheduling_ms) + bar('.', compute_ms - scheduling_ms)
        else:
            cpu_line += bar('S', compute_ms) + bar('S', scheduling_ms - compute_ms)

        gpu_line += bar('C', overlap_time)

        # Postprocess
        cpu_line += bar('P', postprocess_ms)
        gpu_line += bar('.', postprocess_ms)

    print(f"  CPU: {cpu_line}")
    print(f"  GPU: {gpu_line}")


def main():
    parser = argparse.ArgumentParser(description="Scheduling Overhead Benchmark")
    parser.add_argument("--batches", "-b", type=int, default=20,
                        help="Number of batches to process")
    parser.add_argument("--scheduling-ms", type=float, default=2.0,
                        help="Simulated scheduling time in ms")
    parser.add_argument("--compute-ms", type=float, default=10.0,
                        help="Simulated compute time in ms")
    parser.add_argument("--postprocess-ms", type=float, default=1.0,
                        help="Simulated postprocess time in ms")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " SCHEDULING OVERHEAD BENCHMARK".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print(f"\nConfiguration:")
    print(f"  Batches: {args.batches}")
    print(f"  Scheduling time: {args.scheduling_ms} ms")
    print(f"  Compute time: {args.compute_ms} ms")
    print(f"  Postprocess time: {args.postprocess_ms} ms")

    benchmark = SchedulingOverheadBenchmark(
        scheduling_ms=args.scheduling_ms,
        compute_ms=args.compute_ms,
        postprocess_ms=args.postprocess_ms
    )

    # Run sequential
    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS")
    print("=" * 70)

    seq_time, seq_metrics = benchmark.run_sequential(args.batches)
    print_results("SEQUENTIAL (Traditional)", seq_time, seq_metrics, args.batches)

    # Run overlapped
    overlap_time, overlap_metrics = asyncio.run(
        benchmark.run_overlapped(args.batches)
    )
    print_results("OVERLAPPED (Zero-Overhead)", overlap_time, overlap_metrics, args.batches)

    # Comparison
    print("\n" + "=" * 70)
    print(" COMPARISON")
    print("=" * 70)

    speedup = seq_time / overlap_time
    time_saved = seq_time - overlap_time

    print(f"\nSequential: {seq_time:.2f} ms")
    print(f"Overlapped: {overlap_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {time_saved:.2f} ms ({time_saved/seq_time*100:.1f}%)")

    # Visualize
    visualize_timeline(args.scheduling_ms, args.compute_ms,
                       args.postprocess_ms, num_batches=4)

    # Analysis
    print("\n" + "=" * 70)
    print(" ANALYSIS")
    print("=" * 70)
    print(f"""
Key Observations:

1. OVERHEAD IMPACT
   Without overlap: {args.scheduling_ms + args.postprocess_ms} ms overhead per batch
   Total per batch: {args.scheduling_ms + args.compute_ms + args.postprocess_ms} ms
   Overhead percentage: {(args.scheduling_ms + args.postprocess_ms) / (args.scheduling_ms + args.compute_ms + args.postprocess_ms) * 100:.1f}%

2. OVERLAP BENEFIT
   Scheduling is hidden behind compute (when compute > scheduling)
   Effective overhead: {args.postprocess_ms} ms (only postprocessing)
   Overhead reduction: {(args.scheduling_ms) / (args.scheduling_ms + args.postprocess_ms) * 100:.0f}%

3. WHEN OVERLAP HELPS MOST
   - Long compute times (GPU-bound workloads)
   - Significant scheduling overhead
   - Batch decode in LLM inference

4. WHEN OVERLAP HELPS LESS
   - Very short compute times
   - Scheduling time > compute time
   - Complex dependencies between batches
""")


if __name__ == "__main__":
    main()
