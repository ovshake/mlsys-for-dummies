#!/usr/bin/env python3
"""
Tensor-Parallel MLP Block

This script implements a complete tensor-parallel MLP block using
the Megatron-style column→row pattern for minimal communication.

Usage:
    python tp_mlp.py
    python tp_mlp.py --tp-size 4 --hidden-size 256
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP using Megatron-style column→row parallelism.

    Structure:
        Input → [Column-Parallel Linear] → GeLU → [Row-Parallel Linear] → Output

    Communication: 1 all_reduce per forward pass (after row-parallel)
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()

        assert intermediate_size % tp_size == 0

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        self.intermediate_local = intermediate_size // tp_size

        # Column-parallel: W1 shape [hidden, intermediate // tp_size]
        self.w1 = nn.Linear(hidden_size, self.intermediate_local, bias=False)

        # Row-parallel: W2 shape [intermediate // tp_size, hidden]
        self.w2 = nn.Linear(self.intermediate_local, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for TP."""
        nn.init.xavier_uniform_(self.w1.weight)
        # Scale row-parallel weights to maintain variance after all_reduce
        nn.init.xavier_uniform_(self.w2.weight)
        self.w2.weight.data /= self.tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with minimal communication.

        Args:
            x: Input tensor of shape [batch, seq, hidden]

        Returns:
            Output tensor of shape [batch, seq, hidden]
        """
        # Step 1: Column-parallel first linear (no communication)
        h = self.w1(x)

        # Step 2: Activation (local)
        h = torch.nn.functional.gelu(h)

        # Step 3: Row-parallel second linear
        y = self.w2(h)

        # Step 4: All-reduce to sum partial products
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=self.tp_group)

        return y


class NonParallelMLP(nn.Module):
    """Standard MLP for comparison."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.w1(x))
        return self.w2(h)


def benchmark_tp_mlp(rank: int, world_size: int, hidden_size: int,
                     batch_size: int, seq_len: int, warmup: int = 10,
                     iterations: int = 100) -> Tuple[float, float]:
    """Benchmark tensor-parallel MLP."""
    device = torch.device("cpu")
    intermediate_size = hidden_size * 4

    # Create TP MLP
    tp_mlp = TensorParallelMLP(
        hidden_size, intermediate_size, world_size, rank
    ).to(device)

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = tp_mlp(x)
        dist.barrier()

    # Benchmark
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        y = tp_mlp(x)
        dist.barrier()
    total_time = time.perf_counter() - start

    return total_time / iterations, y


def verify_correctness(rank: int, world_size: int, hidden_size: int) -> None:
    """Verify TP MLP produces correct output."""
    device = torch.device("cpu")
    intermediate_size = hidden_size * 4

    if rank == 0:
        print("\n" + "=" * 60)
        print(" CORRECTNESS VERIFICATION")
        print("=" * 60)

    # Create test input (same on all ranks)
    torch.manual_seed(42)
    x = torch.randn(4, 8, hidden_size, device=device)

    # Create TP MLP with deterministic weights
    torch.manual_seed(100)
    tp_mlp = TensorParallelMLP(
        hidden_size, intermediate_size, world_size, rank
    ).to(device)

    # Forward pass
    y_tp = tp_mlp(x)

    # Gather TP weights to rank 0 for comparison
    # W1 (column-parallel)
    w1_local = tp_mlp.w1.weight.data.clone()
    w1_gathered = [torch.zeros_like(w1_local) for _ in range(world_size)]
    dist.all_gather(w1_gathered, w1_local)

    # W2 (row-parallel)
    w2_local = tp_mlp.w2.weight.data.clone()
    w2_gathered = [torch.zeros_like(w2_local) for _ in range(world_size)]
    dist.all_gather(w2_gathered, w2_local)

    if rank == 0:
        # Reconstruct full weights
        w1_full = torch.cat(w1_gathered, dim=0).T  # [hidden, intermediate]
        w2_full = torch.cat(w2_gathered, dim=1)     # [intermediate, hidden]

        # Correct for scaling
        w2_full = w2_full * world_size

        # Compute reference output
        h = torch.nn.functional.gelu(x @ w1_full.T)
        y_ref = h @ w2_full.T

        diff = (y_tp - y_ref).abs().max().item()
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {y_tp.shape}")
        print(f"Max difference from reference: {diff:.2e}")
        print(f"Correct: {diff < 1e-5}")


def analyze_communication(rank: int, world_size: int,
                          hidden_size: int, batch_size: int, seq_len: int) -> None:
    """Analyze communication costs."""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print(" COMMUNICATION ANALYSIS")
    print("=" * 60)

    bytes_per_element = 4  # float32
    elements_per_allreduce = batch_size * seq_len * hidden_size
    bytes_per_allreduce = elements_per_allreduce * bytes_per_element

    # Ring all_reduce volume
    ring_volume = 2 * bytes_per_allreduce * (world_size - 1) / world_size

    print(f"""
Configuration:
  Hidden size: {hidden_size}
  Batch size: {batch_size}
  Sequence length: {seq_len}
  TP degree: {world_size}

Per forward pass:
  All-reduce calls: 1
  Elements per all-reduce: {elements_per_allreduce:,}
  Bytes per all-reduce: {bytes_per_allreduce / 1024:.1f} KB

Communication volume (ring algorithm):
  Per GPU: {ring_volume / 1024:.1f} KB
  Total across all GPUs: {ring_volume * world_size / 1024:.1f} KB

Comparison with non-TP:
  Non-TP: 0 bytes (no communication)
  TP: {ring_volume / 1024:.1f} KB per forward

This is the price of tensor parallelism!
But we can now handle models {world_size}x larger.
""")


def compare_scaling(rank: int, world_size: int) -> None:
    """Compare TP vs non-parallel scaling."""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print(" SCALING ANALYSIS")
    print("=" * 60)
    print("""
Memory scaling with Tensor Parallelism:

For an MLP with hidden_size H and intermediate_size 4H:

Non-parallel:
  W1: H × 4H = 4H² parameters
  W2: 4H × H = 4H² parameters
  Total: 8H² parameters per GPU

With TP degree T:
  W1: H × (4H/T) = 4H²/T parameters
  W2: (4H/T) × H = 4H²/T parameters
  Total: 8H²/T parameters per GPU

Example: H=4096, T=8 (8-way TP)
  Non-parallel: 134M parameters (537 MB in FP32)
  With 8-way TP: 16.7M parameters (67 MB per GPU)

This is how we fit 70B+ parameter models on GPUs!
""")


def worker(rank: int, world_size: int, hidden_size: int,
           batch_size: int, seq_len: int) -> None:
    """Main worker function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29509"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Verify correctness
    verify_correctness(rank, world_size, hidden_size)
    dist.barrier()

    # Analyze communication
    analyze_communication(rank, world_size, hidden_size, batch_size, seq_len)
    dist.barrier()

    # Benchmark
    if rank == 0:
        print("\n" + "=" * 60)
        print(" BENCHMARK")
        print("=" * 60)

    avg_time, output = benchmark_tp_mlp(
        rank, world_size, hidden_size, batch_size, seq_len
    )

    dist.barrier()

    if rank == 0:
        print(f"\nTP MLP forward pass: {avg_time * 1000:.3f} ms")
        print(f"Output shape: {output.shape}")

    # Compare scaling
    compare_scaling(rank, world_size)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Tensor-Parallel MLP Block")
    parser.add_argument("--tp-size", "-t", type=int, default=4,
                        help="Tensor parallelism degree")
    parser.add_argument("--hidden-size", "-H", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=16,
                        help="Sequence length")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " TENSOR-PARALLEL MLP BLOCK".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nTP degree: {args.tp_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Intermediate size: {args.hidden_size * 4}")

    mp.spawn(
        worker,
        args=(args.tp_size, args.hidden_size, args.batch_size, args.seq_len),
        nprocs=args.tp_size,
        join=True
    )


if __name__ == "__main__":
    main()
