#!/usr/bin/env python3
"""
Verify your environment is ready for distributed PyTorch.

Run this script to check:
- PyTorch installation
- CUDA availability
- Distributed backends
- GPU count

Usage:
    python verify_setup.py
"""

import sys


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)


def check_pytorch() -> bool:
    """Check if PyTorch is installed and print version info."""
    print_header("PyTorch Installation")
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] PyTorch location: {torch.__file__}")
        return True
    except ImportError:
        print("[FAIL] PyTorch is not installed!")
        print("       Install with: pip install torch")
        return False


def check_cuda() -> bool:
    """Check CUDA availability and GPU information."""
    print_header("CUDA / GPU Status")
    import torch

    if torch.cuda.is_available():
        print(f"[OK] CUDA is available")
        print(f"[OK] CUDA version: {torch.version.cuda}")
        print(f"[OK] cuDNN version: {torch.backends.cudnn.version()}")

        gpu_count = torch.cuda.device_count()
        print(f"[OK] Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"     GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        return True
    else:
        print("[INFO] CUDA is not available")
        print("       This is OK! We'll use CPU with 'gloo' backend")
        print("       GPU training requires NVIDIA GPU + CUDA toolkit")
        return False


def check_distributed_backends() -> dict:
    """Check which distributed backends are available."""
    print_header("Distributed Backends")
    import torch.distributed as dist

    backends = {
        "gloo": dist.is_gloo_available(),
        "nccl": dist.is_nccl_available(),
        "mpi": dist.is_mpi_available(),
    }

    for name, available in backends.items():
        status = "[OK]" if available else "[NO]"
        description = {
            "gloo": "CPU training, cross-platform",
            "nccl": "GPU training, NVIDIA only",
            "mpi": "HPC clusters",
        }
        print(f"{status} {name.upper()}: {description[name]}")

    # Recommendation
    print("\nRecommendation:")
    if backends["nccl"]:
        print("  Use 'nccl' backend for GPU training")
    if backends["gloo"]:
        print("  Use 'gloo' backend for CPU training or testing")

    return backends


def check_multiprocessing() -> bool:
    """Check multiprocessing support."""
    print_header("Multiprocessing")
    import torch.multiprocessing as mp

    # Check start methods
    methods = mp.get_all_start_methods()
    print(f"[OK] Available start methods: {methods}")
    print(f"[OK] Default start method: {mp.get_start_method()}")

    # Check if spawn is available (needed for CUDA)
    if "spawn" in methods:
        print("[OK] 'spawn' method available (required for CUDA)")
        return True
    else:
        print("[WARN] 'spawn' method not available")
        return False


def run_simple_test() -> bool:
    """Run a simple distributed test on CPU."""
    print_header("Simple Distributed Test")
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import os

    def test_worker(rank: int, world_size: int) -> None:
        """Simple test worker."""
        # Use gloo backend (works on CPU)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size
        )

        # Create a tensor and do all_reduce
        tensor = torch.tensor([rank + 1.0])
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            expected = sum(range(1, world_size + 1))
            if tensor.item() == expected:
                print(f"[OK] all_reduce test passed: {tensor.item()} == {expected}")
            else:
                print(f"[FAIL] all_reduce test failed: {tensor.item()} != {expected}")

        dist.destroy_process_group()

    try:
        world_size = 2
        mp.spawn(test_worker, args=(world_size,), nprocs=world_size, join=True)
        return True
    except Exception as e:
        print(f"[FAIL] Distributed test failed: {e}")
        return False


def main() -> None:
    """Run all verification checks."""
    print("\n" + "=" * 50)
    print(" DISTRIBUTED PYTORCH ENVIRONMENT CHECK")
    print("=" * 50)

    results = {}

    # Run checks
    results["pytorch"] = check_pytorch()
    if not results["pytorch"]:
        print("\n[ABORT] PyTorch is required. Please install it first.")
        sys.exit(1)

    results["cuda"] = check_cuda()
    results["backends"] = check_distributed_backends()
    results["multiprocessing"] = check_multiprocessing()
    results["test"] = run_simple_test()

    # Summary
    print_header("Summary")

    all_ok = all([
        results["pytorch"],
        results["backends"]["gloo"],  # At minimum we need gloo
        results["multiprocessing"],
        results["test"],
    ])

    if all_ok:
        print("[OK] Your environment is ready for distributed PyTorch!")
        print("\nNext steps:")
        print("  1. Run: python hello_distributed.py")
        print("  2. Continue to Chapter 2: Point-to-Point Communication")
    else:
        print("[WARN] Some checks failed. Review the output above.")

    # Hardware recommendation
    if results["cuda"] and results["backends"]["nccl"]:
        print("\n[TIP] You have GPU support! For best performance, use:")
        print("      backend='nccl' for GPU collective operations")
    else:
        print("\n[TIP] No GPU detected. All exercises will work on CPU with:")
        print("      backend='gloo'")


if __name__ == "__main__":
    main()
