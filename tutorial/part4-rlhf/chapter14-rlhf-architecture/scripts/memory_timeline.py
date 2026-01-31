#!/usr/bin/env python3
"""
RLHF Memory Timeline Visualizer

This script visualizes memory usage across different phases of RLHF training,
helping understand memory requirements and bottlenecks.

Usage:
    python memory_timeline.py
    python memory_timeline.py --model-size 70
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ModelConfig:
    """Model configuration for memory estimation."""
    name: str
    params_billions: float
    hidden_size: int
    num_layers: int
    vocab_size: int = 128000


MODELS = {
    "7b": ModelConfig("7B", 7, 4096, 32),
    "13b": ModelConfig("13B", 13, 5120, 40),
    "70b": ModelConfig("70B", 70, 8192, 80),
    "405b": ModelConfig("405B", 405, 16384, 126),
}


def estimate_memory(params_b: float, dtype_bytes: int = 2) -> float:
    """Estimate memory in GB for model parameters."""
    return params_b * 1e9 * dtype_bytes / 1e9


def estimate_kv_cache(batch_size: int, seq_len: int, hidden_size: int,
                       num_layers: int, dtype_bytes: int = 2) -> float:
    """Estimate KV cache memory in GB."""
    # K and V for each layer
    kv_per_token = 2 * num_layers * hidden_size * dtype_bytes
    total = batch_size * seq_len * kv_per_token
    return total / 1e9


def estimate_optimizer_states(params_b: float) -> float:
    """Estimate Adam optimizer state memory in GB."""
    # Adam: 2 states (m, v) in FP32
    return params_b * 1e9 * 4 * 2 / 1e9


def estimate_gradients(params_b: float, dtype_bytes: int = 2) -> float:
    """Estimate gradient memory in GB."""
    return params_b * 1e9 * dtype_bytes / 1e9


def estimate_activations(batch_size: int, seq_len: int, hidden_size: int,
                          num_layers: int, dtype_bytes: int = 2) -> float:
    """Rough estimate of activation memory in GB."""
    # Simplified: ~10x hidden per layer per token
    per_layer = batch_size * seq_len * hidden_size * 10 * dtype_bytes
    return per_layer * num_layers / 1e9


def calculate_phase_memory(model: ModelConfig, batch_size: int, seq_len: int,
                            phase: str) -> Dict[str, float]:
    """Calculate memory breakdown for a specific RLHF phase."""
    memory = {}

    actor_params = estimate_memory(model.params_billions)
    critic_params = estimate_memory(model.params_billions)
    reward_params = estimate_memory(model.params_billions)
    reference_params = estimate_memory(model.params_billions)

    if phase == "generation":
        # Generation phase: Actor inference + KV cache
        memory["Actor (weights)"] = actor_params
        memory["KV Cache"] = estimate_kv_cache(
            batch_size, seq_len, model.hidden_size, model.num_layers
        )
        memory["Reward Model"] = reward_params
        memory["Reference Model"] = reference_params
        memory["Misc (buffers)"] = 2.0

    elif phase == "scoring":
        # Scoring phase: Reward model forward
        memory["Reward Model (weights)"] = reward_params
        memory["Activations"] = estimate_activations(
            batch_size, seq_len, model.hidden_size, model.num_layers
        ) * 0.2  # Inference uses less

    elif phase == "training":
        # Training phase: Actor + Critic with gradients and optimizer
        memory["Actor (weights)"] = actor_params
        memory["Critic (weights)"] = critic_params
        memory["Actor (gradients)"] = estimate_gradients(model.params_billions)
        memory["Critic (gradients)"] = estimate_gradients(model.params_billions)
        memory["Optimizer States"] = estimate_optimizer_states(model.params_billions)
        memory["Activations"] = estimate_activations(
            batch_size, seq_len, model.hidden_size, model.num_layers
        )

    elif phase == "full_rlhf":
        # Full RLHF: worst case (all models loaded)
        memory["Actor"] = actor_params
        memory["Critic"] = critic_params
        memory["Reward Model"] = reward_params
        memory["Reference Model"] = reference_params
        memory["Optimizer States"] = estimate_optimizer_states(model.params_billions)
        memory["Gradients"] = estimate_gradients(model.params_billions) * 2
        memory["KV Cache (peak)"] = estimate_kv_cache(
            batch_size, seq_len, model.hidden_size, model.num_layers
        )
        memory["Activations (peak)"] = estimate_activations(
            batch_size, seq_len, model.hidden_size, model.num_layers
        )

    return memory


def visualize_memory_bar(memory_dict: Dict[str, float], max_memory: float,
                          available: float) -> None:
    """Visualize memory as horizontal bar chart."""
    total = sum(memory_dict.values())
    scale = 50 / max_memory  # Characters per GB

    print(f"\n{'Component':<25} {'Memory':<10} {'Visualization':<50}")
    print("-" * 85)

    for name, mem in sorted(memory_dict.items(), key=lambda x: -x[1]):
        bar_len = int(mem * scale)
        bar = "█" * bar_len
        print(f"{name:<25} {mem:>7.1f} GB {bar}")

    print("-" * 85)
    print(f"{'TOTAL':<25} {total:>7.1f} GB")

    if total > available:
        print(f"\n⚠ EXCEEDS available memory ({available} GB)!")
        print(f"   Need {total/available:.1f}x GPUs or memory optimization")
    else:
        print(f"\n✓ Fits in {available} GB ({total/available*100:.0f}% utilized)")


def show_memory_timeline(model: ModelConfig, batch_size: int, seq_len: int,
                          gpu_memory: float) -> None:
    """Show memory across all RLHF phases."""
    print("\n" + "=" * 80)
    print(f" RLHF MEMORY TIMELINE: {model.name} Model")
    print("=" * 80)

    phases = ["generation", "scoring", "training"]
    max_mem = 0

    for phase in phases:
        mem = calculate_phase_memory(model, batch_size, seq_len, phase)
        phase_total = sum(mem.values())
        max_mem = max(max_mem, phase_total)

    # Now visualize
    for phase in phases:
        mem = calculate_phase_memory(model, batch_size, seq_len, phase)
        print(f"\n--- Phase: {phase.upper()} ---")
        visualize_memory_bar(mem, max_mem * 1.1, gpu_memory)


def show_scaling_analysis(model: ModelConfig, gpu_memory: float) -> None:
    """Show how memory scales with batch size."""
    print("\n" + "=" * 80)
    print(f" SCALING ANALYSIS: {model.name} Model")
    print("=" * 80)

    print(f"\nMemory breakdown (batch_size=4, seq_len=2048):\n")

    mem = calculate_phase_memory(model, 4, 2048, "full_rlhf")

    # Fixed costs (don't scale with batch)
    fixed = 0
    scaling = 0

    for name, m in mem.items():
        if "weights" in name.lower() or "optimizer" in name.lower():
            fixed += m
        else:
            scaling += m

    print(f"Fixed costs (weights, optimizer): {fixed:.1f} GB")
    print(f"Scaling costs (activations, KV): {scaling:.1f} GB")
    print(f"Total: {fixed + scaling:.1f} GB")

    print(f"\nHow scaling costs change:")
    print(f"{'Batch Size':<12} {'KV Cache':<12} {'Activations':<15} {'Total':<12}")
    print("-" * 51)

    for bs in [1, 2, 4, 8, 16, 32]:
        kv = estimate_kv_cache(bs, 2048, model.hidden_size, model.num_layers)
        act = estimate_activations(bs, 2048, model.hidden_size, model.num_layers)
        total = fixed + kv + act

        fit = "✓" if total <= gpu_memory else "✗"
        print(f"{bs:<12} {kv:<12.1f} {act:<15.1f} {total:<12.1f} {fit}")


def recommend_setup(model: ModelConfig, gpu_memory: float, num_gpus: int) -> None:
    """Recommend setup for given constraints."""
    print("\n" + "=" * 80)
    print(f" RECOMMENDED SETUP: {model.name} on {num_gpus}x {gpu_memory}GB GPUs")
    print("=" * 80)

    total_memory = gpu_memory * num_gpus

    # Estimate requirements
    mem = calculate_phase_memory(model, 4, 2048, "full_rlhf")
    required = sum(mem.values())

    print(f"\nMemory analysis:")
    print(f"  Required (naive): {required:.1f} GB")
    print(f"  Available: {total_memory:.1f} GB")

    if required <= gpu_memory:
        print(f"\n✓ Fits on single GPU")
        print(f"  Recommendation: Simple co-located setup")
    elif required <= total_memory:
        print(f"\n✓ Fits across {num_gpus} GPUs")

        # Determine parallelism
        tp_needed = max(1, int(required / gpu_memory * 0.7))  # 70% efficiency
        tp_needed = min(tp_needed, num_gpus, 8)  # Cap at 8 for TP

        print(f"  Recommendation:")
        print(f"    - Tensor Parallelism: {tp_needed}")

        remaining_gpus = num_gpus // tp_needed
        if remaining_gpus > 1:
            print(f"    - Data/Pipeline Parallelism: {remaining_gpus}")
    else:
        print(f"\n✗ Does not fit!")
        print(f"  Need {required / gpu_memory:.0f} GPUs or memory optimization")
        print(f"  Consider:")
        print(f"    - ZeRO-3 / FSDP for optimizer state sharding")
        print(f"    - Gradient checkpointing for activation memory")
        print(f"    - Disaggregated architecture")


def main():
    parser = argparse.ArgumentParser(description="RLHF Memory Timeline")
    parser.add_argument("--model-size", "-m", type=str, default="70b",
                        choices=list(MODELS.keys()),
                        help="Model size")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--gpu-memory", "-g", type=float, default=80,
                        help="GPU memory in GB")
    parser.add_argument("--num-gpus", "-n", type=int, default=8,
                        help="Number of GPUs")
    args = parser.parse_args()

    model = MODELS[args.model_size]

    print("╔" + "═" * 78 + "╗")
    print("║" + " RLHF MEMORY TIMELINE VISUALIZER".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    print(f"\nConfiguration:")
    print(f"  Model: {model.name} ({model.params_billions}B parameters)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  GPU memory: {args.gpu_memory} GB")
    print(f"  Number of GPUs: {args.num_gpus}")

    # Show timeline
    show_memory_timeline(model, args.batch_size, args.seq_len, args.gpu_memory)

    # Show scaling
    show_scaling_analysis(model, args.gpu_memory)

    # Show recommendation
    recommend_setup(model, args.gpu_memory, args.num_gpus)

    # Key insights
    print("\n" + "=" * 80)
    print(" KEY INSIGHTS")
    print("=" * 80)
    print("""
1. FOUR MODELS = 4X WEIGHT MEMORY
   RLHF needs Actor, Critic, Reward, Reference
   For 70B: 4 × 140GB = 560GB just for weights!

2. OPTIMIZER STATES DOMINATE TRAINING
   Adam needs 2× FP32 states per parameter
   For 70B with Actor+Critic: ~1.1TB

3. MEMORY PHASES DIFFER SIGNIFICANTLY
   - Generation: weights + KV cache (no gradients)
   - Training: weights + gradients + optimizer (no KV)
   Smart systems swap between phases

4. BATCH SIZE AFFECTS ACTIVATIONS
   Larger batch → more activation memory
   May need to reduce batch or checkpoint

5. SEQUENCE LENGTH AFFECTS KV CACHE
   Longer sequences → larger KV cache
   4K → 32K = 8x KV memory increase
""")


if __name__ == "__main__":
    main()
