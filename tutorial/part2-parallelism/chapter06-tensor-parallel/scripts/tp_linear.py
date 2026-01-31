#!/usr/bin/env python3
"""
Tensor-Parallel Linear Layers

This script implements column-parallel and row-parallel linear layers
from scratch, showing exactly how tensor parallelism works.

Column-parallel: Split output dimension (no sync needed)
Row-parallel: Split input dimension (all_reduce needed)

Usage:
    python tp_linear.py
    python tp_linear.py --tp-size 4
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-parallel weight matrix.

    The weight matrix W is split along the output dimension:
        W_full: [in_features, out_features]
        W_local: [in_features, out_features // tp_size]

    Each GPU computes a portion of the output features.

    Forward pass:
        Y_local = X @ W_local  (no communication!)
        To get full Y, concatenate Y_local from all GPUs
    """

    def __init__(self, in_features: int, out_features: int,
                 tp_size: int, tp_rank: int):
        super().__init__()
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.out_features_local = out_features // tp_size

        # Local weight: only 1/tp_size of the columns
        self.weight = nn.Parameter(
            torch.empty(in_features, self.out_features_local)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y_local = X @ W_local

        Input: x of shape [batch, in_features]
        Output: y of shape [batch, out_features // tp_size]
        """
        return x @ self.weight

    def __repr__(self):
        return (f"ColumnParallelLinear(in={self.in_features}, "
                f"out={self.out_features_local} (local) / {self.out_features} (total), "
                f"tp_rank={self.tp_rank})")


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-parallel weight matrix.

    The weight matrix W is split along the input dimension:
        W_full: [in_features, out_features]
        W_local: [in_features // tp_size, out_features]

    Each GPU computes a partial result that must be summed.

    Forward pass:
        Y_partial = X_local @ W_local
        Y = all_reduce(Y_partial)  # Sum across all GPUs
    """

    def __init__(self, in_features: int, out_features: int,
                 tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.in_features_local = in_features // tp_size

        # Local weight: only 1/tp_size of the rows
        self.weight = nn.Parameter(
            torch.empty(self.in_features_local, out_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = all_reduce(X_local @ W_local)

        Input: x_local of shape [batch, in_features // tp_size]
        Output: y of shape [batch, out_features]
        """
        # Partial output (not yet complete)
        y_partial = x_local @ self.weight

        # Sum across all GPUs to get complete output
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM, group=self.tp_group)

        return y_partial

    def __repr__(self):
        return (f"RowParallelLinear(in={self.in_features_local} (local) / {self.in_features} (total), "
                f"out={self.out_features}, tp_rank={self.tp_rank})")


def verify_column_parallel(rank: int, world_size: int) -> None:
    """Verify column-parallel linear correctness."""
    device = torch.device("cpu")

    if rank == 0:
        print("\n" + "=" * 60)
        print(" COLUMN-PARALLEL LINEAR VERIFICATION")
        print("=" * 60)

    # Parameters
    batch_size = 4
    in_features = 8
    out_features = 8  # Must be divisible by world_size

    # Create column-parallel layer
    torch.manual_seed(42)
    col_linear = ColumnParallelLinear(
        in_features, out_features, world_size, rank
    ).to(device)

    # Create full layer for comparison (only on rank 0)
    torch.manual_seed(42)
    if rank == 0:
        full_linear = nn.Linear(in_features, out_features, bias=False).to(device)
        # Copy weights to match column-parallel weights
        full_weight = torch.empty(in_features, out_features)
        nn.init.xavier_uniform_(full_weight)

    # Gather all column-parallel weights to rank 0 for verification
    local_weight = col_linear.weight.data.clone()
    gathered_weights = [torch.zeros_like(local_weight) for _ in range(world_size)]
    dist.all_gather(gathered_weights, local_weight)

    if rank == 0:
        reconstructed_weight = torch.cat(gathered_weights, dim=1)
        print(f"\nWeight shapes:")
        print(f"  Local: {local_weight.shape}")
        print(f"  Reconstructed: {reconstructed_weight.shape}")
        print(f"  Full: {full_weight.shape}")

    # Create test input (same on all ranks)
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device)

    # Forward pass with column-parallel
    y_local = col_linear(x)

    # Gather outputs
    gathered_outputs = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, y_local)
    y_reconstructed = torch.cat(gathered_outputs, dim=1)

    # Compare with full layer (only on rank 0)
    if rank == 0:
        # Use reconstructed weight for full computation
        y_full = x @ reconstructed_weight

        diff = (y_reconstructed - y_full).abs().max().item()
        print(f"\nOutput comparison:")
        print(f"  Reconstructed shape: {y_reconstructed.shape}")
        print(f"  Full shape: {y_full.shape}")
        print(f"  Max difference: {diff:.2e}")
        print(f"  Correct: {diff < 1e-5}")


def verify_row_parallel(rank: int, world_size: int) -> None:
    """Verify row-parallel linear correctness."""
    device = torch.device("cpu")

    if rank == 0:
        print("\n" + "=" * 60)
        print(" ROW-PARALLEL LINEAR VERIFICATION")
        print("=" * 60)

    # Parameters
    batch_size = 4
    in_features = 8  # Must be divisible by world_size
    out_features = 8

    # Create row-parallel layer
    torch.manual_seed(42)
    row_linear = RowParallelLinear(
        in_features, out_features, world_size, rank
    ).to(device)

    # Create test input (full, same on all ranks)
    torch.manual_seed(123)
    x_full = torch.randn(batch_size, in_features, device=device)

    # Split input for row-parallel (each rank gets a slice)
    x_chunks = x_full.chunk(world_size, dim=1)
    x_local = x_chunks[rank]

    if rank == 0:
        print(f"\nInput shapes:")
        print(f"  Full: {x_full.shape}")
        print(f"  Local: {x_local.shape}")

    # Forward pass with row-parallel (includes all_reduce!)
    y_row_parallel = row_linear(x_local)

    # Gather all local weights to reconstruct full weight
    local_weight = row_linear.weight.data.clone()
    gathered_weights = [torch.zeros_like(local_weight) for _ in range(world_size)]
    dist.all_gather(gathered_weights, local_weight)

    if rank == 0:
        # Reconstruct full weight
        full_weight = torch.cat(gathered_weights, dim=0)

        # Compute full output for comparison
        y_full = x_full @ full_weight

        diff = (y_row_parallel - y_full).abs().max().item()
        print(f"\nOutput comparison:")
        print(f"  Row-parallel shape: {y_row_parallel.shape}")
        print(f"  Full shape: {y_full.shape}")
        print(f"  Max difference: {diff:.2e}")
        print(f"  Correct: {diff < 1e-5}")


def demonstrate_megatron_pattern(rank: int, world_size: int) -> None:
    """Demonstrate the Megatron column→row pattern."""
    device = torch.device("cpu")

    if rank == 0:
        print("\n" + "=" * 60)
        print(" MEGATRON PATTERN: Column + Row")
        print("=" * 60)
        print("""
The Megatron-LM pattern for an MLP block:

1. First linear (column-parallel):
   - Input: [batch, hidden]
   - Output: [batch, 4*hidden // tp_size]
   - No communication needed!

2. Activation (GeLU):
   - Applied locally
   - Still no communication!

3. Second linear (row-parallel):
   - Input: [batch, 4*hidden // tp_size]
   - Output: [batch, hidden]
   - ONE all_reduce to sum partial products

Result: Only 1 all_reduce per MLP block forward pass!
""")

    # Demonstrate the pattern
    batch_size = 4
    hidden_size = 8
    intermediate_size = 32  # 4x hidden

    # Column-parallel first layer
    torch.manual_seed(42 + rank)
    W1_col = ColumnParallelLinear(
        hidden_size, intermediate_size, world_size, rank
    ).to(device)

    # Row-parallel second layer
    torch.manual_seed(142 + rank)
    W2_row = RowParallelLinear(
        intermediate_size, hidden_size, world_size, rank
    ).to(device)

    # Input (same on all ranks)
    torch.manual_seed(200)
    x = torch.randn(batch_size, hidden_size, device=device)

    # Forward pass
    # Step 1: Column-parallel (no communication)
    h = W1_col(x)
    if rank == 0:
        print(f"\nAfter column-parallel W1:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {h.shape} (partitioned across {world_size} GPUs)")

    # Step 2: Activation (local)
    h = torch.relu(h)

    # Step 3: Row-parallel (one all_reduce)
    y = W2_row(h)
    if rank == 0:
        print(f"\nAfter row-parallel W2:")
        print(f"  Input shape: {h.shape}")
        print(f"  Output shape: {y.shape} (after all_reduce)")

    # Verify all ranks have the same output
    gathered_outputs = [torch.zeros_like(y) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, y)

    if rank == 0:
        all_same = all(torch.allclose(gathered_outputs[0], g) for g in gathered_outputs[1:])
        print(f"\nAll ranks have identical output: {all_same}")
        print("\nKey insight: We achieved tensor parallelism with only ONE all_reduce!")


def worker(rank: int, world_size: int) -> None:
    """Main worker function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29508"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    verify_column_parallel(rank, world_size)
    dist.barrier()

    verify_row_parallel(rank, world_size)
    dist.barrier()

    demonstrate_megatron_pattern(rank, world_size)
    dist.barrier()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Tensor-Parallel Linear Layers")
    parser.add_argument("--tp-size", "-t", type=int, default=4,
                        help="Tensor parallelism degree (default: 4)")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " TENSOR-PARALLEL LINEAR LAYERS".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nTP degree: {args.tp_size}")

    mp.spawn(worker, args=(args.tp_size,), nprocs=args.tp_size, join=True)


if __name__ == "__main__":
    main()
