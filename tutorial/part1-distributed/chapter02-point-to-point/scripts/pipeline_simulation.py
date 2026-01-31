#!/usr/bin/env python3
"""
Pipeline Parallelism Simulation

This script simulates how pipeline parallelism works in practice:
- A "model" is split across multiple processes (stages)
- Data flows forward through the pipeline via send/recv
- Each stage processes its part of the model

Real pipeline parallelism also has backward pass and more complex
scheduling (1F1B, interleaved), but this shows the core concept.

Usage:
    python pipeline_simulation.py
    python pipeline_simulation.py --batch-size 64 --hidden-size 128

The "model" is just a series of Linear layers:
    Input → Linear → ReLU → Linear → ReLU → ... → Output
            Stage 0         Stage 1         Stage N-1
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class PipelineStage(nn.Module):
    """
    One stage of our pipeline (a simple feed-forward block).

    In a real model like GPT, each stage might be several transformer layers.
    """

    def __init__(self, input_size: int, output_size: int, stage_id: int):
        super().__init__()
        self.stage_id = stage_id
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def pipeline_worker(
    rank: int,
    world_size: int,
    batch_size: int,
    hidden_size: int,
    num_microbatches: int,
    backend: str
) -> None:
    """
    Worker function for one pipeline stage.

    Args:
        rank: This stage's rank
        world_size: Total number of stages
        batch_size: Per-microbatch batch size
        hidden_size: Model hidden dimension
        num_microbatches: Number of microbatches to process
        backend: Distributed backend
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")

    # Create this stage's model part
    stage = PipelineStage(hidden_size, hidden_size, rank).to(device)

    # =========================================================================
    # Pipeline Forward Pass
    # =========================================================================
    # We process multiple microbatches to show the pipelining effect.
    # In practice, while Stage 1 processes microbatch 0,
    # Stage 0 can start processing microbatch 1 (pipeline filling).

    timings = []

    for mb_idx in range(num_microbatches):
        start_time = time.perf_counter()

        if rank == 0:
            # First stage: generate input (in reality, this comes from data loader)
            activations = torch.randn(batch_size, hidden_size, device=device)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Created input "
                  f"(shape: {list(activations.shape)})")
        else:
            # Other stages: receive activations from previous stage
            activations = torch.zeros(batch_size, hidden_size, device=device)
            dist.recv(activations, src=rank - 1)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Received from stage {rank - 1}")

        # Process through this stage's model part
        with torch.no_grad():
            output = stage(activations)

        if rank == world_size - 1:
            # Last stage: we're done (in reality, compute loss here)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Completed! "
                  f"Output mean: {output.mean().item():.4f}")
        else:
            # Send activations to next stage
            dist.send(output, dst=rank + 1)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Sent to stage {rank + 1}")

        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)

    # Synchronize before printing summary
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print("PIPELINE SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Stages: {world_size}")
        print(f"Microbatches: {num_microbatches}")
        print(f"Batch size per microbatch: {batch_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"\nIn a real pipeline:")
        print(f"  - Stages process different microbatches in parallel")
        print(f"  - Backward pass sends gradients in reverse")
        print(f"  - 1F1B schedule optimizes memory usage")
        print(f"{'='*60}")

    dist.destroy_process_group()


def visualize_pipeline():
    """Print a visualization of pipeline parallelism."""
    print("""
    ═══════════════════════════════════════════════════════════════════════
    PIPELINE PARALLELISM VISUALIZATION
    ═══════════════════════════════════════════════════════════════════════

    The model is split across GPUs/processes:

    Full Model:     [Embed] → [Layer 0-3] → [Layer 4-7] → [Layer 8-11] → [Head]
                        ↓           ↓            ↓             ↓           ↓
    Pipeline:       Stage 0     Stage 1      Stage 2       Stage 3     Stage 4

    Data flows through stages via send/recv:

    Time →
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  Stage 0:  [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►[MB3 Fwd]     │
    │                 │              │              │              │         │
    │                 ▼              ▼              ▼              ▼         │
    │  Stage 1:      [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►[MB3 Fwd] │
    │                     │              │              │              │     │
    │                     ▼              ▼              ▼              ▼     │
    │  Stage 2:          [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►...   │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    MB = Microbatch, Fwd = Forward pass

    Key insight: While Stage 2 processes MB0, Stage 1 processes MB1,
    and Stage 0 processes MB2. The pipeline is "full" of work!

    ═══════════════════════════════════════════════════════════════════════
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate pipeline parallelism with send/recv"
    )
    parser.add_argument(
        "--world-size", "-w",
        type=int,
        default=4,
        help="Number of pipeline stages (default: 4)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size per microbatch (default: 32)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Model hidden dimension (default: 64)"
    )
    parser.add_argument(
        "--num-microbatches", "-m",
        type=int,
        default=4,
        help="Number of microbatches to process (default: 4)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show pipeline visualization and exit"
    )
    args = parser.parse_args()

    if args.visualize:
        visualize_pipeline()
        return

    print("=" * 60)
    print(" PIPELINE PARALLELISM SIMULATION")
    print("=" * 60)
    print(f"Number of stages: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Microbatches: {args.num_microbatches}")
    print("=" * 60 + "\n")

    mp.spawn(
        pipeline_worker,
        args=(
            args.world_size,
            args.batch_size,
            args.hidden_size,
            args.num_microbatches,
            args.backend
        ),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
