#!/usr/bin/env python3
"""
Distributed Mean Computation

This script shows how to compute the mean of data distributed across
multiple processes. This is EXACTLY what happens in gradient synchronization!

Scenario:
- Each process has local data (gradients in real training)
- We want the global mean across ALL data

Two approaches:
1. all_reduce(SUM) / world_size  (simple, always works)
2. Local mean, then weighted average (more efficient for unequal sizes)

Usage:
    python distributed_mean.py
    python distributed_mean.py --data-size 1000
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def compute_mean_simple(rank: int, world_size: int, data_size: int,
                        device: torch.device) -> torch.Tensor:
    """
    Simple approach: all_reduce(SUM) / world_size

    This works when all processes have equal-sized data.
    It's what PyTorch DDP does for gradient synchronization.
    """
    # Simulate local gradients (different on each rank)
    local_data = torch.randn(data_size, device=device) + rank

    # Step 1: Sum across all processes
    total = local_data.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    # Step 2: Divide by number of processes
    mean = total / world_size

    return mean, local_data


def compute_mean_weighted(rank: int, world_size: int, local_sizes: list,
                          device: torch.device) -> torch.Tensor:
    """
    Weighted approach for unequal local sizes.

    When processes have different amounts of data (e.g., last batch smaller),
    we need to weight by the local size.
    """
    # Each process has different amount of data
    local_size = local_sizes[rank]
    local_data = torch.randn(local_size, device=device) + rank

    # Step 1: Compute local sum
    local_sum = local_data.sum()

    # Step 2: all_reduce the sum
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)

    # Step 3: all_reduce the count
    local_count = torch.tensor([float(local_size)], device=device)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    # Step 4: Global mean = global sum / global count
    global_mean = local_sum / local_count

    return global_mean, local_data


def mean_worker(rank: int, world_size: int, data_size: int, backend: str) -> None:
    """Worker demonstrating distributed mean computation."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")

    # =========================================================================
    # Demo 1: Simple Mean (equal data sizes)
    # =========================================================================
    if rank == 0:
        print("=" * 60)
        print(" DISTRIBUTED MEAN: Simple Approach (equal sizes)")
        print("=" * 60)

    torch.manual_seed(42 + rank)  # Reproducible but different per rank
    dist_mean, local_data = compute_mean_simple(rank, world_size, data_size, device)

    local_mean = local_data.mean().item()
    dist.barrier()

    print(f"[Rank {rank}] Local mean: {local_mean:.4f}, Distributed mean: {dist_mean.mean().item():.4f}")

    dist.barrier()

    if rank == 0:
        print("\n[Verification] Distributed mean should equal average of local means.")
        print("This works because all ranks have equal-sized data.\n")

    # =========================================================================
    # Demo 2: Weighted Mean (unequal data sizes)
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("=" * 60)
        print(" DISTRIBUTED MEAN: Weighted Approach (unequal sizes)")
        print("=" * 60)

    # Simulate unequal batch sizes (e.g., last batch is smaller)
    local_sizes = [data_size, data_size, data_size, data_size // 2][:world_size]

    torch.manual_seed(42 + rank)
    weighted_mean, local_data = compute_mean_weighted(rank, world_size, local_sizes, device)

    local_mean = local_data.mean().item()
    dist.barrier()

    print(f"[Rank {rank}] Size: {local_sizes[rank]}, Local mean: {local_mean:.4f}, "
          f"Weighted global mean: {weighted_mean.item():.4f}")

    dist.barrier()

    if rank == 0:
        print("\n[Verification] Weighted mean properly accounts for different sizes.")
        print("This is important when batch sizes vary!\n")

    # =========================================================================
    # Demo 3: Gradient Synchronization (the real use case)
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("=" * 60)
        print(" PRACTICAL EXAMPLE: Gradient Synchronization")
        print("=" * 60)
        print("""
In distributed data-parallel training, each GPU computes gradients
on its local batch. To train correctly, we need the AVERAGE gradient
across all batches.

Pseudo-code for DDP:
    # Forward pass (local)
    loss = model(batch)

    # Backward pass (local)
    loss.backward()  # Computes gradients locally

    # Synchronize gradients
    for param in model.parameters():
        dist.all_reduce(param.grad, op=ReduceOp.SUM)
        param.grad /= world_size

    # Optimizer step (local, but now with averaged gradients)
    optimizer.step()
""")

    # Simulate gradient computation
    torch.manual_seed(123 + rank)
    fake_gradient = torch.randn(10, device=device)

    if rank == 0:
        print("Before synchronization:")
    dist.barrier()
    print(f"  [Rank {rank}] gradient[0]: {fake_gradient[0].item():.4f}")
    dist.barrier()

    # Synchronize gradients
    dist.all_reduce(fake_gradient, op=dist.ReduceOp.SUM)
    fake_gradient /= world_size

    if rank == 0:
        print("\nAfter synchronization (all ranks have same gradient):")
    dist.barrier()
    print(f"  [Rank {rank}] gradient[0]: {fake_gradient[0].item():.4f}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Mean Computation")
    parser.add_argument(
        "--data-size", "-d",
        type=int,
        default=100,
        help="Size of local data per process (default: 100)"
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
        help="Distributed backend"
    )
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " DISTRIBUTED MEAN COMPUTATION".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    mp.spawn(
        mean_worker,
        args=(args.world_size, args.data_size, args.backend),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
