#!/usr/bin/env python3
"""
Parallel Strategy Calculator

Given model specifications and hardware constraints, this script helps
you determine the optimal parallelism strategy.

It calculates:
- Memory requirements for different strategies
- Communication volumes
- Recommended configuration

Usage:
    python parallel_strategy_calculator.py
    python parallel_strategy_calculator.py --params 70 --gpus 64 --memory 80
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ModelConfig:
    """Model configuration."""
    params_billions: float
    hidden_size: int = 8192
    num_layers: int = 80
    num_heads: int = 64
    vocab_size: int = 128000
    intermediate_ratio: float = 4.0
    is_moe: bool = False
    num_experts: int = 1
    top_k: int = 1  # Experts activated per token


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    num_gpus: int
    memory_per_gpu_gb: float
    intra_node_bandwidth_gbps: float = 900  # NVLink
    inter_node_bandwidth_gbps: float = 50   # InfiniBand
    gpus_per_node: int = 8


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1024
    sequence_length: int = 4096
    dtype_bytes: int = 2  # FP16/BF16


def estimate_model_memory(config: ModelConfig, dtype_bytes: int = 2) -> dict:
    """Estimate memory requirements for model parameters."""
    # Parameter count estimation
    # Embedding: vocab_size * hidden_size
    # Per layer: 4 * hidden^2 (QKV + O) + 2 * hidden * intermediate (FFN)

    embedding_params = config.vocab_size * config.hidden_size
    attention_params = 4 * config.hidden_size ** 2  # Q, K, V, O projections
    ffn_params = 2 * config.hidden_size * int(config.hidden_size * config.intermediate_ratio)

    if config.is_moe:
        # MoE: multiply FFN by number of experts
        ffn_params *= config.num_experts

    layer_params = attention_params + ffn_params
    total_params = embedding_params + (layer_params * config.num_layers)

    # Convert to bytes
    params_bytes = total_params * dtype_bytes
    gradients_bytes = params_bytes  # Same size as params

    # Optimizer states (Adam: 2x params in FP32)
    optimizer_bytes = total_params * 4 * 2

    return {
        'params': params_bytes / 1e9,  # GB
        'gradients': gradients_bytes / 1e9,
        'optimizer': optimizer_bytes / 1e9,
        'total': (params_bytes + gradients_bytes + optimizer_bytes) / 1e9,
        'param_count': total_params,
    }


def estimate_activation_memory(
    config: ModelConfig,
    training: TrainingConfig,
    tp_degree: int = 1,
    pp_degree: int = 1
) -> float:
    """Estimate activation memory per GPU in GB."""
    batch_per_gpu = training.batch_size // (training.batch_size // pp_degree)
    seq_len = training.sequence_length
    hidden = config.hidden_size

    # Per-layer activations (simplified)
    # Input to attention, attention output, FFN intermediate, etc.
    activations_per_layer = 10 * batch_per_gpu * seq_len * hidden // tp_degree

    layers_per_stage = config.num_layers // pp_degree

    total_activation_bytes = activations_per_layer * layers_per_stage * training.dtype_bytes

    return total_activation_bytes / 1e9


def calculate_communication_volume(
    config: ModelConfig,
    training: TrainingConfig,
    tp_degree: int,
    dp_degree: int,
    pp_degree: int
) -> dict:
    """Calculate communication volume for different parallelism types."""
    batch = training.batch_size
    seq = training.sequence_length
    hidden = config.hidden_size
    dtype = training.dtype_bytes

    # TP communication: all_reduce per layer
    # 2 all_reduce per transformer layer (attention + FFN)
    tp_volume_per_layer = 4 * batch * seq * hidden * dtype * (tp_degree - 1) / tp_degree
    tp_volume_total = tp_volume_per_layer * config.num_layers / pp_degree

    # PP communication: activations between stages
    pp_volume = 2 * batch * seq * hidden * dtype  # Forward and backward

    # DP communication: gradient all_reduce
    params_per_stage = config.params_billions * 1e9 / pp_degree
    dp_volume = 2 * params_per_stage * dtype * (dp_degree - 1) / dp_degree

    return {
        'tp_per_step_gb': tp_volume_total / 1e9,
        'pp_per_step_gb': pp_volume / 1e9,
        'dp_per_step_gb': dp_volume / 1e9,
        'total_gb': (tp_volume_total + pp_volume + dp_volume) / 1e9,
    }


def find_optimal_strategy(
    model: ModelConfig,
    hardware: HardwareConfig,
    training: TrainingConfig
) -> dict:
    """Find optimal parallelism strategy given constraints."""
    mem = estimate_model_memory(model, training.dtype_bytes)

    results = []

    # Try different configurations
    for tp in [1, 2, 4, 8]:
        if tp > hardware.gpus_per_node:
            continue

        for pp in [1, 2, 4, 8, 16]:
            if tp * pp > hardware.num_gpus:
                continue

            dp = hardware.num_gpus // (tp * pp)
            if dp < 1:
                continue

            # Memory per GPU
            params_per_gpu = mem['params'] / (tp * pp)
            grads_per_gpu = mem['gradients'] / (tp * pp)
            optimizer_per_gpu = mem['optimizer'] / (tp * pp)  # With ZeRO-3

            # ZeRO-3 shards optimizer across DP
            optimizer_per_gpu = optimizer_per_gpu / dp

            activation_mem = estimate_activation_memory(model, training, tp, pp)

            total_mem = params_per_gpu + grads_per_gpu + optimizer_per_gpu + activation_mem

            # Communication
            comm = calculate_communication_volume(model, training, tp, dp, pp)

            # Estimate if TP crosses nodes
            tp_is_intra_node = tp <= hardware.gpus_per_node

            results.append({
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'memory_per_gpu': total_mem,
                'fits': total_mem < hardware.memory_per_gpu_gb * 0.9,  # 90% threshold
                'communication': comm,
                'tp_intra_node': tp_is_intra_node,
            })

    return results


def print_results(results: list, hardware: HardwareConfig) -> None:
    """Print strategy comparison."""
    print("\n" + "=" * 80)
    print(" STRATEGY COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'TP':>4} {'PP':>4} {'DP':>4} {'Mem/GPU':>10} {'Fits?':>8} "
          f"{'TP Comm':>10} {'PP Comm':>10} {'DP Comm':>10}")
    print("-" * 80)

    valid_configs = []

    for r in results:
        fits = "✓" if r['fits'] else "✗"
        tp_note = "" if r['tp_intra_node'] else "*"

        print(f"{r['tp']:>4}{tp_note:<1} {r['pp']:>3} {r['dp']:>4} "
              f"{r['memory_per_gpu']:>9.1f}GB {fits:>8} "
              f"{r['communication']['tp_per_step_gb']:>9.2f}GB "
              f"{r['communication']['pp_per_step_gb']:>9.2f}GB "
              f"{r['communication']['dp_per_step_gb']:>9.2f}GB")

        if r['fits']:
            valid_configs.append(r)

    print("\n* TP crosses node boundary (slower inter-node communication)")

    # Recommendation
    print("\n" + "=" * 80)
    print(" RECOMMENDATION")
    print("=" * 80)

    if not valid_configs:
        print("\n⚠ No configuration fits in memory!")
        print("  Consider: More GPUs, larger GPU memory, or smaller batch size")
        return

    # Sort by communication volume (prefer lower communication)
    valid_configs.sort(key=lambda x: (
        0 if x['tp_intra_node'] else 1,  # Prefer intra-node TP
        x['communication']['total_gb'],
    ))

    best = valid_configs[0]

    print(f"""
Recommended configuration:
  Tensor Parallelism (TP): {best['tp']}
  Pipeline Parallelism (PP): {best['pp']}
  Data Parallelism (DP): {best['dp']}

Memory per GPU: {best['memory_per_gpu']:.1f} GB (limit: {hardware.memory_per_gpu_gb} GB)
Total communication: {best['communication']['total_gb']:.2f} GB per step

Reasoning:
  - TP={best['tp']} {"stays within a node (NVLink speed)" if best['tp_intra_node'] else "crosses nodes (slower)"}
  - PP={best['pp']} splits model into {best['pp']} stages
  - DP={best['dp']} {"provides excellent scaling" if best['dp'] > 1 else "single replica"}
""")


def main():
    parser = argparse.ArgumentParser(description="Parallel Strategy Calculator")
    parser.add_argument("--params", "-p", type=float, default=70,
                        help="Model parameters in billions (default: 70)")
    parser.add_argument("--gpus", "-g", type=int, default=64,
                        help="Total number of GPUs (default: 64)")
    parser.add_argument("--memory", "-m", type=float, default=80,
                        help="GPU memory in GB (default: 80)")
    parser.add_argument("--batch-size", "-b", type=int, default=512,
                        help="Global batch size (default: 512)")
    parser.add_argument("--seq-len", "-s", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--moe", action="store_true",
                        help="Model is Mixture of Experts")
    parser.add_argument("--num-experts", type=int, default=64,
                        help="Number of experts for MoE (default: 64)")
    args = parser.parse_args()

    print("╔" + "═" * 78 + "╗")
    print("║" + " PARALLEL STRATEGY CALCULATOR".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Configure model
    model = ModelConfig(
        params_billions=args.params,
        is_moe=args.moe,
        num_experts=args.num_experts if args.moe else 1,
    )

    hardware = HardwareConfig(
        num_gpus=args.gpus,
        memory_per_gpu_gb=args.memory,
    )

    training = TrainingConfig(
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
    )

    # Print configuration
    print(f"\n{'─'*40}")
    print(" MODEL CONFIGURATION")
    print(f"{'─'*40}")
    mem = estimate_model_memory(model)
    print(f"Parameters: {args.params}B ({mem['param_count']/1e9:.1f}B actual)")
    print(f"Parameters memory: {mem['params']:.1f} GB")
    print(f"Gradients memory: {mem['gradients']:.1f} GB")
    print(f"Optimizer memory: {mem['optimizer']:.1f} GB")
    print(f"Total model memory: {mem['total']:.1f} GB")
    if args.moe:
        print(f"MoE: {args.num_experts} experts")

    print(f"\n{'─'*40}")
    print(" HARDWARE CONFIGURATION")
    print(f"{'─'*40}")
    print(f"GPUs: {args.gpus} ({args.gpus // 8} nodes)")
    print(f"Memory per GPU: {args.memory} GB")

    print(f"\n{'─'*40}")
    print(" TRAINING CONFIGURATION")
    print(f"{'─'*40}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")

    # Calculate strategies
    results = find_optimal_strategy(model, hardware, training)
    print_results(results, hardware)

    # Additional MoE considerations
    if args.moe:
        print("\n" + "=" * 80)
        print(" MOE-SPECIFIC CONSIDERATIONS")
        print("=" * 80)
        print(f"""
For MoE models, also consider Expert Parallelism (EP):

  With {args.num_experts} experts and 8-way EP:
  - {args.num_experts // 8} experts per GPU
  - Communication: 2 all-to-all per layer

EP vs TP for MoE:
  - EP: Keeps full expert matrices → better GEMM efficiency
  - TP: Slices experts → smaller GEMMs, worse efficiency
  - EP preferred when num_experts >> TP degree

Recommendation: Use EP instead of/in addition to TP for the FFN experts.
""")


if __name__ == "__main__":
    main()
