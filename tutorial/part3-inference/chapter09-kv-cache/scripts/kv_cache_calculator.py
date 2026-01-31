#!/usr/bin/env python3
"""
KV Cache Calculator

Calculate KV cache memory requirements for different LLM configurations.
This helps understand memory constraints and capacity planning.

Usage:
    python kv_cache_calculator.py
    python kv_cache_calculator.py --model llama-70b
    python kv_cache_calculator.py --custom --layers 80 --heads 64 --dim 128
"""

import argparse
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """Model configuration for KV cache calculation."""
    name: str
    num_layers: int
    num_kv_heads: int  # KV heads (may differ from attention heads with GQA)
    head_dim: int
    vocab_size: int = 128000

    @property
    def kv_bytes_per_token(self) -> int:
        """Calculate KV cache bytes per token (for FP16)."""
        # K and V for each layer
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2  # 2 bytes for FP16


# Common model configurations
MODELS = {
    "llama-7b": ModelConfig("LLaMA-7B", num_layers=32, num_kv_heads=32, head_dim=128),
    "llama-13b": ModelConfig("LLaMA-13B", num_layers=40, num_kv_heads=40, head_dim=128),
    "llama-70b": ModelConfig("LLaMA-70B", num_layers=80, num_kv_heads=8, head_dim=128),  # GQA
    "mistral-7b": ModelConfig("Mistral-7B", num_layers=32, num_kv_heads=8, head_dim=128),  # GQA
    "qwen-72b": ModelConfig("Qwen-72B", num_layers=80, num_kv_heads=8, head_dim=128),
    "deepseek-67b": ModelConfig("DeepSeek-67B", num_layers=95, num_kv_heads=8, head_dim=128),
}


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def calculate_kv_cache(model: ModelConfig, context_lengths: list,
                       dtype_bytes: int = 2) -> Dict:
    """Calculate KV cache requirements."""
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)

    results = {
        'model': model.name,
        'layers': model.num_layers,
        'kv_heads': model.num_kv_heads,
        'head_dim': model.head_dim,
        'kv_bytes_per_token': kv_per_token,
        'contexts': {}
    }

    for ctx_len in context_lengths:
        kv_per_request = kv_per_token * ctx_len
        results['contexts'][ctx_len] = {
            'per_request': kv_per_request,
            'per_request_formatted': format_bytes(kv_per_request),
        }

    return results


def analyze_capacity(model: ModelConfig, gpu_memory_gb: float,
                     model_size_gb: float, context_length: int,
                     dtype_bytes: int = 2) -> Dict:
    """Analyze how many concurrent requests can be served."""
    # Available memory for KV cache
    overhead_gb = 2  # CUDA kernels, activations, etc.
    available_gb = gpu_memory_gb - model_size_gb - overhead_gb

    if available_gb <= 0:
        return {
            'error': 'Model does not fit in GPU memory',
            'available_gb': available_gb,
        }

    # KV cache per request
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)
    kv_per_request = kv_per_token * context_length
    kv_per_request_gb = kv_per_request / (1024 ** 3)

    # Max concurrent requests
    max_requests = int(available_gb / kv_per_request_gb)

    # With PagedAttention (assuming 95% utilization vs 50% without)
    requests_without_paging = int(max_requests * 0.5)  # 50% utilization
    requests_with_paging = int(max_requests * 0.95)    # 95% utilization

    return {
        'gpu_memory_gb': gpu_memory_gb,
        'model_size_gb': model_size_gb,
        'available_for_kv_gb': available_gb,
        'context_length': context_length,
        'kv_per_request_gb': kv_per_request_gb,
        'max_theoretical_requests': max_requests,
        'requests_without_paging': requests_without_paging,
        'requests_with_paging': requests_with_paging,
        'paging_improvement': f"{(requests_with_paging / requests_without_paging - 1) * 100:.0f}%"
    }


def compare_fragmentation(model: ModelConfig, requests: int,
                          avg_context: int, max_context: int,
                          dtype_bytes: int = 2) -> Dict:
    """Compare memory usage with and without paging."""
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)

    # Without paging: allocate max_context for each request
    memory_without_paging = requests * max_context * kv_per_token

    # With paging: only allocate what's actually used
    memory_with_paging = requests * avg_context * kv_per_token

    waste = memory_without_paging - memory_with_paging
    waste_pct = (waste / memory_without_paging) * 100

    return {
        'requests': requests,
        'avg_context': avg_context,
        'max_context': max_context,
        'memory_without_paging': format_bytes(memory_without_paging),
        'memory_with_paging': format_bytes(memory_with_paging),
        'memory_wasted': format_bytes(waste),
        'waste_percentage': f"{waste_pct:.1f}%",
    }


def print_model_comparison():
    """Print KV cache comparison for common models."""
    print("=" * 70)
    print(" KV CACHE SIZE COMPARISON ACROSS MODELS")
    print("=" * 70)

    context_lengths = [2048, 4096, 8192, 32768, 131072]

    print(f"\n{'Model':<15} {'Layers':<8} {'KV Heads':<10} "
          f"{'Per Token':<12} {'@ 8K ctx':<12} {'@ 32K ctx':<12}")
    print("-" * 70)

    for name, model in MODELS.items():
        results = calculate_kv_cache(model, context_lengths)
        per_token = format_bytes(results['kv_bytes_per_token'])
        at_8k = results['contexts'][8192]['per_request_formatted']
        at_32k = results['contexts'][32768]['per_request_formatted']

        print(f"{model.name:<15} {model.num_layers:<8} {model.num_kv_heads:<10} "
              f"{per_token:<12} {at_8k:<12} {at_32k:<12}")


def print_capacity_analysis(model_name: str, gpu_config: str):
    """Print capacity analysis for a specific configuration."""
    model = MODELS.get(model_name.lower())
    if not model:
        print(f"Unknown model: {model_name}")
        return

    # GPU configurations
    gpu_configs = {
        "h100": (80, "H100 80GB"),
        "a100": (80, "A100 80GB"),
        "a100-40": (40, "A100 40GB"),
        "4090": (24, "RTX 4090 24GB"),
    }

    # Model sizes (approximate, FP16)
    model_sizes = {
        "llama-7b": 14,
        "llama-13b": 26,
        "llama-70b": 140,
        "mistral-7b": 14,
        "qwen-72b": 144,
        "deepseek-67b": 134,
    }

    gpu_memory, gpu_name = gpu_configs.get(gpu_config, (80, "Custom"))
    model_size = model_sizes.get(model_name.lower(), 14)

    print("\n" + "=" * 70)
    print(f" CAPACITY ANALYSIS: {model.name} on {gpu_name}")
    print("=" * 70)

    for context_len in [2048, 4096, 8192, 32768]:
        capacity = analyze_capacity(model, gpu_memory, model_size, context_len)

        if 'error' in capacity:
            print(f"\n@ {context_len} context: {capacity['error']}")
            continue

        print(f"\n@ {context_len} context length:")
        print(f"  Available for KV cache: {capacity['available_for_kv_gb']:.1f} GB")
        print(f"  KV per request: {capacity['kv_per_request_gb']:.2f} GB")
        print(f"  Without PagedAttention: ~{capacity['requests_without_paging']} concurrent requests")
        print(f"  With PagedAttention: ~{capacity['requests_with_paging']} concurrent requests")
        print(f"  Improvement: {capacity['paging_improvement']}")


def print_fragmentation_analysis(model_name: str):
    """Show memory fragmentation with and without paging."""
    model = MODELS.get(model_name.lower())
    if not model:
        print(f"Unknown model: {model_name}")
        return

    print("\n" + "=" * 70)
    print(f" FRAGMENTATION ANALYSIS: {model.name}")
    print("=" * 70)

    scenarios = [
        (100, 512, 8192, "Short prompts, 8K max"),
        (50, 2048, 8192, "Medium prompts, 8K max"),
        (20, 4096, 32768, "Long prompts, 32K max"),
        (10, 8192, 131072, "Very long, 128K max"),
    ]

    for requests, avg_ctx, max_ctx, desc in scenarios:
        frag = compare_fragmentation(model, requests, avg_ctx, max_ctx)

        print(f"\nScenario: {desc}")
        print(f"  Requests: {requests}, Avg context: {avg_ctx}, Max context: {max_ctx}")
        print(f"  Without paging: {frag['memory_without_paging']}")
        print(f"  With paging: {frag['memory_with_paging']}")
        print(f"  Memory saved: {frag['memory_wasted']} ({frag['waste_percentage']} reduction)")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Calculator")
    parser.add_argument("--model", "-m", type=str, default="llama-70b",
                        choices=list(MODELS.keys()),
                        help="Model to analyze")
    parser.add_argument("--gpu", "-g", type=str, default="h100",
                        choices=["h100", "a100", "a100-40", "4090"],
                        help="GPU type")
    parser.add_argument("--custom", action="store_true",
                        help="Use custom model config")
    parser.add_argument("--layers", type=int, default=80,
                        help="Number of layers (with --custom)")
    parser.add_argument("--heads", type=int, default=8,
                        help="Number of KV heads (with --custom)")
    parser.add_argument("--dim", type=int, default=128,
                        help="Head dimension (with --custom)")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " KV CACHE CALCULATOR".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if args.custom:
        custom_model = ModelConfig(
            "Custom",
            num_layers=args.layers,
            num_kv_heads=args.heads,
            head_dim=args.dim
        )
        MODELS["custom"] = custom_model
        args.model = "custom"

    # Model comparison
    print_model_comparison()

    # Capacity analysis
    print_capacity_analysis(args.model, args.gpu)

    # Fragmentation analysis
    print_fragmentation_analysis(args.model)

    # Key insights
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. KV CACHE DOMINATES MEMORY
   - For long contexts, KV cache >> model weights
   - 70B model @ 32K context: 140GB weights vs ~10GB KV per request

2. GQA DRAMATICALLY REDUCES KV CACHE
   - LLaMA-70B uses 8 KV heads (vs 64 attention heads)
   - 8x smaller KV cache per token!

3. PAGEDATTENTION NEARLY DOUBLES CAPACITY
   - Eliminates internal fragmentation
   - 95% utilization vs ~50% without paging

4. CONTEXT LENGTH IS THE KILLER
   - 32K context: ~47 concurrent requests
   - 128K context: ~12 concurrent requests
   - Same GPU, same model!

5. QUANTIZED KV CACHE HELPS
   - FP8 KV cache: 2x more requests
   - INT8 KV cache: similar benefits
   - Some quality trade-off
""")


if __name__ == "__main__":
    main()
