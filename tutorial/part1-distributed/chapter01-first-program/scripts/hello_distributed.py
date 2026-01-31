#!/usr/bin/env python3
"""
Your First Distributed Program!

This script demonstrates the fundamentals of distributed PyTorch:
- Process group initialization
- Rank and world_size concepts
- Simple tensor communication with all_gather

Usage:
    python hello_distributed.py
    python hello_distributed.py --world-size 8

What this script does:
1. Spawns multiple processes (default: 4)
2. Each process initializes a distributed environment
3. Processes share information about themselves
4. We demonstrate all_gather to collect data from all processes
"""

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def get_device_info() -> dict:
    """Get information about the current process's compute device."""
    if torch.cuda.is_available():
        # Get local rank (which GPU this process should use)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    return {
        "device": device,
        "device_name": device_name,
        "pid": os.getpid(),
    }


def distributed_worker(rank: int, world_size: int, backend: str) -> None:
    """
    The main function that runs in each distributed process.

    Args:
        rank: Unique identifier for this process (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('gloo' or 'nccl')
    """
    # =========================================================================
    # Step 1: Initialize the process group
    # =========================================================================
    # This is the "handshake" - all processes must call this before communicating
    # Environment variables are set by mp.spawn automatically:
    #   - MASTER_ADDR: Address of rank 0 process
    #   - MASTER_PORT: Port for communication

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # =========================================================================
    # Step 2: Get device and process information
    # =========================================================================
    info = get_device_info()
    device = info["device"]

    print(f"[Rank {rank}/{world_size}] Hello! PID={info['pid']}, Device={info['device_name']}")

    # =========================================================================
    # Step 3: Demonstrate all_gather - collect data from all processes
    # =========================================================================
    # Each process creates a tensor with its rank value
    # After all_gather, every process has all tensors

    # Create a tensor unique to this rank
    my_tensor = torch.tensor([rank * 10.0, rank * 10.0 + 1], device=device)
    print(f"[Rank {rank}] My tensor: {my_tensor.tolist()}")

    # Prepare a list to receive tensors from all ranks
    gathered_tensors: List[torch.Tensor] = [
        torch.zeros(2, device=device) for _ in range(world_size)
    ]

    # all_gather: collect my_tensor from all ranks into gathered_tensors
    dist.all_gather(gathered_tensors, my_tensor)

    # Synchronize before printing (ensures all processes complete the operation)
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 50)
        print("all_gather results (collected on all ranks):")
        for i, tensor in enumerate(gathered_tensors):
            print(f"  From rank {i}: {tensor.tolist()}")
        print("=" * 50 + "\n")

    # =========================================================================
    # Step 4: Demonstrate all_reduce - aggregate values across all processes
    # =========================================================================
    # Each process contributes its rank, and we sum them all

    my_value = torch.tensor([float(rank)], device=device)
    dist.all_reduce(my_value, op=dist.ReduceOp.SUM)

    if rank == 0:
        expected_sum = sum(range(world_size))
        print(f"all_reduce (SUM) result: {my_value.item()}")
        print(f"  Expected: 0 + 1 + ... + {world_size-1} = {expected_sum}")
        print(f"  Correct: {my_value.item() == expected_sum}\n")

    # =========================================================================
    # Step 5: Show that rank 0 is special (often used as "master")
    # =========================================================================
    if rank == 0:
        print("I am rank 0 - often called the 'master' or 'coordinator'")
        print("Common responsibilities of rank 0:")
        print("  - Logging and printing results")
        print("  - Saving checkpoints")
        print("  - Orchestrating distributed operations")

    # =========================================================================
    # Step 6: Clean up
    # =========================================================================
    # Always destroy the process group when done
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Your First Distributed Program")
    parser.add_argument(
        "--world-size", "-w",
        type=int,
        default=4,
        help="Number of processes to spawn (default: 4)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend (default: gloo for CPU compatibility)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print(" YOUR FIRST DISTRIBUTED PROGRAM")
    print("=" * 50)
    print(f"World size: {args.world_size}")
    print(f"Backend: {args.backend}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
    print("=" * 50 + "\n")

    # Spawn worker processes
    # mp.spawn will:
    #   1. Create args.world_size new processes
    #   2. Call distributed_worker(rank, world_size, backend) in each
    #   3. Pass rank=0,1,2,... to each process automatically
    mp.spawn(
        distributed_worker,
        args=(args.world_size, args.backend),
        nprocs=args.world_size,
        join=True  # Wait for all processes to complete
    )

    print("\nAll processes completed successfully!")


if __name__ == "__main__":
    main()
