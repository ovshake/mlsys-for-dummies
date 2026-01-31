#!/usr/bin/env python3
"""
Basic Point-to-Point Communication: The Chain Pattern

This script demonstrates send/recv in a chain topology:
    Rank 0 → Rank 1 → Rank 2 → Rank 3

Each process receives from the previous rank, adds 10, and sends to the next.

Usage:
    python send_recv_basic.py
    python send_recv_basic.py --world-size 8

Key concepts:
- Blocking send/recv
- Chain topology (avoiding deadlocks)
- Careful ordering of operations
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def chain_worker(rank: int, world_size: int, backend: str) -> None:
    """
    Worker function implementing a chain communication pattern.

    Data flows: Rank 0 → Rank 1 → Rank 2 → ... → Rank (world_size-1)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Get device (CPU for gloo, GPU for nccl)
    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # =========================================================================
    # The Chain Pattern
    # =========================================================================
    # This pattern naturally avoids deadlocks because:
    # - Rank 0 only sends (no one sends to it first)
    # - Middle ranks receive then send (in that order)
    # - Last rank only receives (no one receives from it)

    if rank == 0:
        # First process: create initial tensor and send
        tensor = torch.tensor([42.0], device=device)
        print(f"[Rank 0] Starting chain with value: {tensor.item()}")
        dist.send(tensor, dst=1)
        print(f"[Rank 0] Sent to rank 1")

    elif rank == world_size - 1:
        # Last process: receive and display final result
        tensor = torch.zeros(1, device=device)
        dist.recv(tensor, src=rank - 1)
        print(f"[Rank {rank}] Received final value: {tensor.item()}")
        print(f"\n{'='*50}")
        print(f"Chain complete!")
        print(f"Original: 42.0")
        print(f"After {world_size - 1} additions of 10: {tensor.item()}")
        print(f"Expected: {42.0 + (world_size - 1) * 10}")
        print(f"{'='*50}")

    else:
        # Middle processes: receive, add 10, send
        tensor = torch.zeros(1, device=device)
        dist.recv(tensor, src=rank - 1)
        print(f"[Rank {rank}] Received: {tensor.item()}")

        tensor += 10  # Transform the data
        print(f"[Rank {rank}] After adding 10: {tensor.item()}")

        dist.send(tensor, dst=rank + 1)
        print(f"[Rank {rank}] Sent to rank {rank + 1}")

    # Synchronize all processes before cleanup
    dist.barrier()
    dist.destroy_process_group()


def demonstrate_deadlock_pattern():
    """
    Educational function showing a deadlock pattern (DO NOT RUN).
    """
    print("""
    ⚠️  DEADLOCK PATTERN (DO NOT USE):

    # Process 0                # Process 1
    send(tensor, dst=1)        send(tensor, dst=0)
    recv(tensor, src=1)        recv(tensor, src=0)

    Both processes block on send(), waiting for the other to receive.
    Neither can proceed → DEADLOCK!

    ✓ CORRECT PATTERN (interleaved):

    # Process 0                # Process 1
    send(tensor, dst=1)        recv(tensor, src=0)
    recv(tensor, src=1)        send(tensor, dst=0)

    Process 0 sends while Process 1 receives → both can proceed.
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate chain pattern point-to-point communication"
    )
    parser.add_argument(
        "--world-size", "-w",
        type=int,
        default=4,
        help="Number of processes in the chain (default: 4)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend"
    )
    parser.add_argument(
        "--show-deadlock",
        action="store_true",
        help="Show deadlock pattern explanation (educational)"
    )
    args = parser.parse_args()

    if args.show_deadlock:
        demonstrate_deadlock_pattern()
        return

    print("=" * 50)
    print(" POINT-TO-POINT COMMUNICATION: CHAIN PATTERN")
    print("=" * 50)
    print(f"World size: {args.world_size}")
    print(f"Pattern: Rank 0 → Rank 1 → ... → Rank {args.world_size - 1}")
    print(f"Operation: Each rank adds 10 before forwarding")
    print("=" * 50 + "\n")

    mp.spawn(
        chain_worker,
        args=(args.world_size, args.backend),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
