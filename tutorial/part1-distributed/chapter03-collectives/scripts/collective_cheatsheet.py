#!/usr/bin/env python3
"""
Collective Operations Cheatsheet

This script demonstrates all major collective operations with clear
before/after output. Run it to understand what each operation does.

Operations covered:
- broadcast: One-to-all (same data)
- scatter: One-to-all (different data)
- gather: All-to-one
- all_gather: All-to-all (collect)
- reduce: All-to-one (aggregate)
- all_reduce: All-to-all (aggregate)

Usage:
    python collective_cheatsheet.py
    python collective_cheatsheet.py --operation all_reduce
"""

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def print_state(rank: int, world_size: int, name: str, tensor: torch.Tensor,
                is_before: bool = True) -> None:
    """Pretty print tensor state."""
    dist.barrier()  # Synchronize printing
    if is_before:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f" {name}")
            print(f"{'='*50}")
            print("BEFORE:")
        dist.barrier()
        print(f"  Rank {rank}: {tensor.tolist()}")
    else:
        dist.barrier()
        if rank == 0:
            print("AFTER:")
        dist.barrier()
        print(f"  Rank {rank}: {tensor.tolist()}")
    dist.barrier()


def demo_broadcast(rank: int, world_size: int, device: torch.device) -> None:
    """
    BROADCAST: One process sends the same data to all others.

    Use case: Share hyperparameters, model weights initialization,
              random seed from rank 0 to all processes.
    """
    # Before: only rank 0 has meaningful data
    if rank == 0:
        tensor = torch.tensor([42.0, 43.0, 44.0], device=device)
    else:
        tensor = torch.zeros(3, device=device)

    print_state(rank, world_size, "BROADCAST (src=0)", tensor, is_before=True)

    # Broadcast from rank 0 to all
    dist.broadcast(tensor, src=0)

    print_state(rank, world_size, "BROADCAST (src=0)", tensor, is_before=False)

    if rank == 0:
        print("\n[Explanation] Rank 0's data [42, 43, 44] was copied to all ranks.")


def demo_scatter(rank: int, world_size: int, device: torch.device) -> None:
    """
    SCATTER: One process distributes different chunks to each process.

    Use case: Distribute different batches of data to workers.
    """
    # Before: only rank 0 has all data
    if rank == 0:
        scatter_list = [
            torch.tensor([i * 10.0, i * 10 + 1.0], device=device)
            for i in range(world_size)
        ]
        print_state(rank, world_size, "SCATTER (src=0)", torch.stack(scatter_list), is_before=True)
    else:
        scatter_list = None
        print_state(rank, world_size, "SCATTER (src=0)", torch.zeros(2, device=device), is_before=True)

    # Receive buffer
    recv_tensor = torch.zeros(2, device=device)

    # Scatter from rank 0
    dist.scatter(recv_tensor, scatter_list=scatter_list if rank == 0 else None, src=0)

    print_state(rank, world_size, "SCATTER (src=0)", recv_tensor, is_before=False)

    if rank == 0:
        print("\n[Explanation] Rank 0 distributed different chunks to each rank:")
        print("             Rank 0 got [0,1], Rank 1 got [10,11], etc.")


def demo_gather(rank: int, world_size: int, device: torch.device) -> None:
    """
    GATHER: Collect data from all processes to one process.

    Use case: Collect results, predictions, or metrics to rank 0.
    """
    # Each rank has unique data
    tensor = torch.tensor([rank * 100.0, rank * 100 + 1.0], device=device)

    print_state(rank, world_size, "GATHER (dst=0)", tensor, is_before=True)

    # Gather to rank 0
    if rank == 0:
        gather_list = [torch.zeros(2, device=device) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(tensor, gather_list=gather_list, dst=0)

    if rank == 0:
        result = torch.stack(gather_list)
        print_state(rank, world_size, "GATHER (dst=0)", result, is_before=False)
        print("\n[Explanation] Rank 0 collected all data. Other ranks have nothing new.")
    else:
        print_state(rank, world_size, "GATHER (dst=0)", tensor, is_before=False)


def demo_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    """
    ALL_GATHER: Collect data from all processes to ALL processes.

    Use case: Share embeddings, gather activations for all-to-all attention.
    """
    # Each rank has unique data
    tensor = torch.tensor([rank + 1.0], device=device)

    print_state(rank, world_size, "ALL_GATHER", tensor, is_before=True)

    # All-gather: everyone gets everything
    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    gathered_tensor = torch.cat(gathered)
    print_state(rank, world_size, "ALL_GATHER", gathered_tensor, is_before=False)

    if rank == 0:
        print("\n[Explanation] Every rank now has [1, 2, 3, 4] (data from all ranks).")


def demo_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """
    REDUCE: Aggregate (sum/max/min/product) data from all to one process.

    Use case: Compute total loss, find global max, etc.
    """
    # Each rank has data to contribute
    tensor = torch.tensor([rank + 1.0], device=device)

    print_state(rank, world_size, "REDUCE SUM (dst=0)", tensor, is_before=True)

    # Reduce to rank 0 with sum
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    print_state(rank, world_size, "REDUCE SUM (dst=0)", tensor, is_before=False)

    if rank == 0:
        print(f"\n[Explanation] Rank 0 has sum: 1+2+3+4 = {tensor.item()}")
        print("             Other ranks' tensors are unchanged (or undefined).")


def demo_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    """
    ALL_REDUCE: Aggregate and distribute result to ALL processes.

    Use case: GRADIENT SYNCHRONIZATION! This is the heart of distributed training.
    """
    # Each rank has gradients to contribute
    tensor = torch.tensor([rank + 1.0, (rank + 1.0) * 2], device=device)

    print_state(rank, world_size, "ALL_REDUCE SUM", tensor, is_before=True)

    # All-reduce with sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print_state(rank, world_size, "ALL_REDUCE SUM", tensor, is_before=False)

    if rank == 0:
        print(f"\n[Explanation] All ranks now have the same sum!")
        print(f"             Element 0: 1+2+3+4 = 10")
        print(f"             Element 1: 2+4+6+8 = 20")
        print("             This is how gradient synchronization works!")


def demo_reduce_scatter(rank: int, world_size: int, device: torch.device) -> None:
    """
    REDUCE_SCATTER: Reduce + Scatter in one operation.

    Use case: Efficient gradient synchronization for model parallelism,
              ZeRO optimizer.
    """
    # Each rank has a tensor that will be element-wise reduced, then scattered
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device) * (rank + 1)

    print_state(rank, world_size, "REDUCE_SCATTER SUM", tensor, is_before=True)

    # Reduce-scatter
    output = torch.zeros(1, device=device)
    dist.reduce_scatter(output, [tensor[i:i+1].clone() for i in range(world_size)])

    print_state(rank, world_size, "REDUCE_SCATTER SUM", output, is_before=False)

    if rank == 0:
        print("\n[Explanation] First sums across ranks, then each rank gets one chunk.")
        print("             Rank i gets sum of position i from all ranks.")


def collective_worker(rank: int, world_size: int, operation: str, backend: str) -> None:
    """Main worker function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")

    operations = {
        "broadcast": demo_broadcast,
        "scatter": demo_scatter,
        "gather": demo_gather,
        "all_gather": demo_all_gather,
        "reduce": demo_reduce,
        "all_reduce": demo_all_reduce,
        "reduce_scatter": demo_reduce_scatter,
        "all": None,  # Special case
    }

    if operation == "all":
        for op_name, op_func in operations.items():
            if op_name != "all" and op_func is not None:
                op_func(rank, world_size, device)
                dist.barrier()
                if rank == 0:
                    print("\n" + "─" * 50)
    else:
        operations[operation](rank, world_size, device)

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Collective Operations Cheatsheet")
    parser.add_argument(
        "--operation", "-o",
        type=str,
        default="all",
        choices=["broadcast", "scatter", "gather", "all_gather",
                 "reduce", "all_reduce", "reduce_scatter", "all"],
        help="Which operation to demonstrate (default: all)"
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
    print("║" + " COLLECTIVE OPERATIONS CHEATSHEET".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"World size: {args.world_size}")
    print(f"Operation: {args.operation}")

    mp.spawn(
        collective_worker,
        args=(args.world_size, args.operation, args.backend),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
