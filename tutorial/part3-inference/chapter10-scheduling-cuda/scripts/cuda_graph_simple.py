#!/usr/bin/env python3
"""
CUDA Graphs Simple Demonstration

This script demonstrates CUDA Graphs for reducing kernel launch overhead.
CUDA Graphs record GPU operations and replay them with minimal CPU overhead.

Usage:
    python cuda_graph_simple.py

Note: Requires CUDA GPU. Falls back to simulation on CPU.
"""

import argparse
import time
from typing import Tuple


def check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_with_cuda():
    """Run the demo with actual CUDA Graphs."""
    import torch
    import torch.nn as nn

    print("=" * 60)
    print(" CUDA GRAPHS DEMONSTRATION")
    print("=" * 60)
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    ).cuda()
    model.eval()

    # Fixed-size input (required for CUDA Graphs)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 512, device='cuda')
    output_tensor = torch.zeros(batch_size, 512, device='cuda')

    num_iterations = 1000

    # =========================================================================
    # Method 1: Normal execution (kernel launch per operation)
    # =========================================================================
    print("\n" + "-" * 60)
    print(" Method 1: Normal Execution")
    print("-" * 60)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            output = model(input_tensor)
    torch.cuda.synchronize()
    normal_time = time.perf_counter() - start

    print(f"Total time: {normal_time * 1000:.2f} ms")
    print(f"Per iteration: {normal_time / num_iterations * 1000:.3f} ms")

    # =========================================================================
    # Method 2: CUDA Graph capture and replay
    # =========================================================================
    print("\n" + "-" * 60)
    print(" Method 2: CUDA Graph Replay")
    print("-" * 60)

    # Create static tensors for capture
    static_input = torch.randn(batch_size, 512, device='cuda')
    static_output = torch.zeros(batch_size, 512, device='cuda')

    # Warmup for capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                static_output = model(static_input)
    torch.cuda.current_stream().wait_stream(s)

    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_output = model(static_input)

    print("Graph captured successfully!")
    print(f"Graph contains operations for: Linear → ReLU → Linear → ReLU → Linear")

    # Benchmark graph replay
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        # Copy new input to static buffer
        static_input.copy_(input_tensor)
        # Replay the graph
        graph.replay()
    torch.cuda.synchronize()
    graph_time = time.perf_counter() - start

    print(f"\nTotal time: {graph_time * 1000:.2f} ms")
    print(f"Per iteration: {graph_time / num_iterations * 1000:.3f} ms")

    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print(" COMPARISON")
    print("=" * 60)

    speedup = normal_time / graph_time
    overhead_saved = (normal_time - graph_time) / num_iterations * 1000

    print(f"\nNormal execution: {normal_time / num_iterations * 1000:.3f} ms/iter")
    print(f"CUDA Graph replay: {graph_time / num_iterations * 1000:.3f} ms/iter")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Overhead saved per iteration: {overhead_saved:.3f} ms")

    # Verify correctness
    static_input.copy_(input_tensor)
    graph.replay()
    torch.cuda.synchronize()

    with torch.no_grad():
        expected = model(input_tensor)

    diff = (static_output - expected).abs().max().item()
    print(f"\nCorrectness check (max diff): {diff:.2e}")
    print(f"Results match: {diff < 1e-5}")

    # =========================================================================
    # Demonstrate constraints
    # =========================================================================
    print("\n" + "=" * 60)
    print(" CUDA GRAPH CONSTRAINTS")
    print("=" * 60)
    print("""
CUDA Graphs REQUIRE:
  ✓ Fixed tensor shapes
  ✓ Pre-allocated output buffers
  ✓ Deterministic operations
  ✓ Static control flow

CUDA Graphs FORBID:
  ✗ Dynamic shapes (different batch sizes)
  ✗ Random operations (dropout)
  ✗ CPU-GPU synchronization points
  ✗ Memory allocation during execution

For LLM inference:
  - Decode phase: Fixed batch size → CUDA Graphs work great!
  - Prefill phase: Variable prompt lengths → Cannot use CUDA Graphs
  - Solution: Capture graphs for common batch sizes, fall back otherwise
""")


def run_simulation():
    """Simulate CUDA Graphs concept without GPU."""
    print("=" * 60)
    print(" CUDA GRAPHS SIMULATION (No GPU)")
    print("=" * 60)
    print("\nNote: Running without GPU. Demonstrating concept only.\n")

    # Simulate overhead
    kernel_launch_overhead_ms = 0.05  # Per kernel
    compute_time_ms = 0.5  # Actual compute
    num_kernels = 5  # Linear, ReLU, Linear, ReLU, Linear
    num_iterations = 1000

    # Normal execution: overhead per kernel per iteration
    normal_time = num_iterations * (num_kernels * kernel_launch_overhead_ms + compute_time_ms)

    # CUDA Graph: overhead only once for entire graph
    graph_time = num_iterations * (kernel_launch_overhead_ms + compute_time_ms)

    print("-" * 60)
    print(" SIMULATED COMPARISON")
    print("-" * 60)

    print(f"\nAssumptions:")
    print(f"  Kernel launch overhead: {kernel_launch_overhead_ms} ms")
    print(f"  Number of kernels: {num_kernels}")
    print(f"  Compute time: {compute_time_ms} ms")
    print(f"  Iterations: {num_iterations}")

    print(f"\nNormal execution:")
    print(f"  Per iteration: {num_kernels} × {kernel_launch_overhead_ms} + {compute_time_ms} = "
          f"{num_kernels * kernel_launch_overhead_ms + compute_time_ms} ms")
    print(f"  Total: {normal_time} ms")

    print(f"\nCUDA Graph replay:")
    print(f"  Per iteration: 1 × {kernel_launch_overhead_ms} + {compute_time_ms} = "
          f"{kernel_launch_overhead_ms + compute_time_ms} ms")
    print(f"  Total: {graph_time} ms")

    speedup = normal_time / graph_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Overhead reduced from {num_kernels * kernel_launch_overhead_ms} ms to "
          f"{kernel_launch_overhead_ms} ms per iteration")


def demonstrate_multiple_graphs():
    """Show how real systems handle multiple batch sizes."""
    print("\n" + "=" * 60)
    print(" HANDLING MULTIPLE BATCH SIZES")
    print("=" * 60)
    print("""
Real inference systems capture multiple CUDA Graphs:

  Graph pool:
    - batch_size=1:  [captured graph for single request decode]
    - batch_size=2:  [captured graph for 2 requests]
    - batch_size=4:  [captured graph for 4 requests]
    - batch_size=8:  [captured graph for 8 requests]
    - batch_size=16: [captured graph for 16 requests]
    ...

  At runtime:
    1. Check current batch size
    2. If graph exists for this size: replay()
    3. If not: fall back to normal execution

  Trade-offs:
    - More graphs = more GPU memory for graph storage
    - Typical: capture for powers of 2 up to max batch size
    - Padding: batch_size=5 might use batch_size=8 graph with padding
""")


def main():
    parser = argparse.ArgumentParser(description="CUDA Graphs Demo")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU simulation even if GPU available")
    args = parser.parse_args()

    has_cuda = check_cuda() and not args.force_cpu

    if has_cuda:
        run_with_cuda()
    else:
        run_simulation()

    demonstrate_multiple_graphs()


if __name__ == "__main__":
    main()
