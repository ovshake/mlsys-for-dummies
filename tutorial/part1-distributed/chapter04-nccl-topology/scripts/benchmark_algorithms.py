#!/usr/bin/env python3
"""
NCCL Algorithm Benchmark

This script benchmarks all_reduce performance with different:
- Message sizes (small vs large)
- Number of processes
- Backend settings

It demonstrates how performance characteristics change based on
these parameters, showing when Ring vs Tree algorithms excel.

Usage:
    python benchmark_algorithms.py
    python benchmark_algorithms.py --sizes 1000,1000000,100000000

Note: On CPU-only systems, this uses the gloo backend which
doesn't have Ring/Tree algorithm selection, but still demonstrates
how message size affects throughput.
"""

import argparse
import os
import time
from typing import List, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def format_bytes(size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_bandwidth(bytes_per_sec: float) -> str:
    """Format bandwidth into human-readable string."""
    return format_bytes(int(bytes_per_sec)) + "/s"


def benchmark_all_reduce(
    tensor: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict:
    """
    Benchmark all_reduce operation.

    Returns dict with timing statistics.
    """
    # Warmup
    for _ in range(warmup_iterations):
        dist.all_reduce(tensor.clone())

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        test_tensor = tensor.clone()

        start = time.perf_counter()
        dist.all_reduce(test_tensor)
        dist.barrier()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'median_ms': sorted(times)[len(times)//2] * 1000,
    }


def benchmark_worker(
    rank: int,
    world_size: int,
    message_sizes: List[int],
    backend: str,
    num_iterations: int
) -> None:
    """Worker function for benchmarking."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # Run benchmarks
    results = []
    for size in message_sizes:
        # Create tensor of specified size (in bytes, using float32 = 4 bytes)
        num_elements = size // 4
        tensor = torch.randn(num_elements, device=device)

        stats = benchmark_all_reduce(tensor, num_iterations=num_iterations)

        # Calculate bandwidth
        # all_reduce moves approximately 2 * size * (N-1) / N bytes (ring algorithm)
        bytes_moved = 2 * size * (world_size - 1) / world_size
        bandwidth = bytes_moved / (stats['mean_ms'] / 1000)

        results.append({
            'size': size,
            'num_elements': num_elements,
            'stats': stats,
            'bandwidth': bandwidth,
        })

        dist.barrier()

    # Only rank 0 prints results
    if rank == 0:
        print("\n" + "=" * 70)
        print(" ALL_REDUCE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Backend: {backend}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Iterations per test: {num_iterations}")
        print("=" * 70)

        print(f"\n{'Size':<12} {'Elements':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Bandwidth':<15}")
        print("-" * 70)

        for r in results:
            print(f"{format_bytes(r['size']):<12} "
                  f"{r['num_elements']:<12} "
                  f"{r['stats']['mean_ms']:<12.3f} "
                  f"{r['stats']['min_ms']:<12.3f} "
                  f"{format_bandwidth(r['bandwidth']):<15}")

        print("\n" + "=" * 70)
        print(" ANALYSIS")
        print("=" * 70)

        if len(results) >= 2:
            # Compare small vs large messages
            small = results[0]
            large = results[-1]

            small_latency = small['stats']['mean_ms']
            large_latency = large['stats']['mean_ms']
            size_ratio = large['size'] / small['size']
            latency_ratio = large_latency / small_latency

            print(f"\nLatency scaling:")
            print(f"  Message size increased {size_ratio:.0f}x")
            print(f"  Latency increased {latency_ratio:.1f}x")

            if latency_ratio < size_ratio * 0.5:
                print(f"  → Latency grows sub-linearly with size (good bandwidth utilization)")
            elif latency_ratio < size_ratio:
                print(f"  → Latency grows roughly linearly with size")
            else:
                print(f"  → Latency grows super-linearly (possible bottleneck)")

            print(f"\nBandwidth comparison:")
            print(f"  Small messages ({format_bytes(small['size'])}): {format_bandwidth(small['bandwidth'])}")
            print(f"  Large messages ({format_bytes(large['size'])}): {format_bandwidth(large['bandwidth'])}")

            if large['bandwidth'] > small['bandwidth'] * 1.5:
                print(f"  → Large messages achieve much better bandwidth utilization")
                print(f"  → This is typical: large messages amortize fixed overhead")

        print("""
Understanding the results:

1. SMALL MESSAGES (< 1 MB):
   - Dominated by latency (startup cost)
   - Tree algorithm excels here (O(log N) steps)
   - Low bandwidth utilization

2. LARGE MESSAGES (> 10 MB):
   - Dominated by bandwidth
   - Ring algorithm excels here (~100% utilization)
   - Latency becomes less important

3. NCCL AUTO-SELECTION:
   - NCCL automatically chooses Ring or Tree based on message size
   - Small: Tree (low latency)
   - Large: Ring (high bandwidth)
   - Crossover point is typically around 1-10 MB

4. THEORETICAL PEAK:
   - NVLink 4.0: ~450 GB/s effective for all_reduce
   - PCIe 4.0: ~16 GB/s effective for all_reduce
   - If your numbers are much lower, check topology!
""")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="NCCL Algorithm Benchmark")
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,10000,100000,1000000,10000000,100000000",
        help="Comma-separated message sizes in bytes (default: 1KB to 100MB)"
    )
    parser.add_argument(
        "--world-size", "-w",
        type=int,
        default=4,
        help="Number of processes (default: 4)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend (default: gloo for CPU compatibility)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Number of iterations per test (default: 50)"
    )
    args = parser.parse_args()

    message_sizes = [int(s) for s in args.sizes.split(',')]

    print("╔" + "═" * 58 + "╗")
    print("║" + " NCCL ALGORITHM BENCHMARK".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nMessage sizes: {[format_bytes(s) for s in message_sizes]}")
    print(f"World size: {args.world_size}")
    print(f"Backend: {args.backend}")
    print(f"Iterations: {args.iterations}")

    if args.backend == "nccl" and not torch.cuda.is_available():
        print("\n[WARN] NCCL backend requires CUDA. Falling back to gloo.")
        args.backend = "gloo"

    mp.spawn(
        benchmark_worker,
        args=(args.world_size, message_sizes, args.backend, args.iterations),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
