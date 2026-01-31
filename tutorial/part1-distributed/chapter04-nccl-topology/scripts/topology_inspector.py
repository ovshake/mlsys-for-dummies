#!/usr/bin/env python3
"""
GPU Topology Inspector

This script inspects your GPU topology and provides insights about:
- Number of GPUs and their properties
- NVLink connections between GPUs
- PCIe topology
- NUMA affinity
- Recommended process placement

Usage:
    python topology_inspector.py

Note: This requires NVIDIA GPUs. On systems without GPUs, it will
display a simulated topology for educational purposes.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is available."""
    try:
        subprocess.run(
            ["nvidia-smi", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gpu_count() -> int:
    """Get the number of GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip().split('\n'))
    except:
        return 0


def get_gpu_info() -> List[Dict]:
    """Get detailed GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_mb': int(parts[2]),
                    'pci_bus': parts[3]
                })
        return gpus
    except:
        return []


def get_topology_matrix() -> Optional[str]:
    """Get the GPU topology matrix from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except:
        return None


def get_nvlink_status() -> Optional[str]:
    """Get NVLink status for GPU 0."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status", "-i", "0"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except:
        return None


def parse_topology_matrix(matrix_str: str) -> Dict[Tuple[int, int], str]:
    """Parse topology matrix into a dict of GPU pairs to connection types."""
    connections = {}
    lines = matrix_str.strip().split('\n')

    # Find the header line with GPU columns
    header_idx = None
    for i, line in enumerate(lines):
        if 'GPU0' in line or 'GPU 0' in line:
            header_idx = i
            break

    if header_idx is None:
        return connections

    # Parse the matrix
    for line in lines[header_idx + 1:]:
        if not line.strip() or 'Legend' in line:
            break

        parts = line.split()
        if not parts or not parts[0].startswith('GPU'):
            continue

        try:
            gpu_from = int(parts[0].replace('GPU', ''))
            for col_idx, conn in enumerate(parts[1:]):
                if conn in ['X', 'NV1', 'NV2', 'NV3', 'NV4', 'NV5', 'NV6',
                           'NV7', 'NV8', 'NV12', 'NV18', 'SYS', 'NODE',
                           'PHB', 'PXB', 'PIX']:
                    connections[(gpu_from, col_idx)] = conn
        except (ValueError, IndexError):
            continue

    return connections


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "═" * 60)
    print(f" {title}")
    print("═" * 60)


def print_simulated_topology() -> None:
    """Print a simulated topology for educational purposes."""
    print_header("SIMULATED GPU TOPOLOGY (No GPUs Detected)")

    print("""
This is a simulated DGX-A100 topology for educational purposes.

In a real DGX-A100 with 8 A100 GPUs:

        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X

Legend:
  X    = Self
  NV#  = Connected via NVLink (# = number of links)
  SYS  = Connected via PCIe across NUMA nodes (slowest)
  NODE = Same NUMA node, connected via PCIe
  PHB  = Same PCIe host bridge
  PXB  = Different PCIe bridges, same PCIe switch
  PIX  = Same PCIe switch

Performance implications:
  NV12 (12 NVLinks): ~300 GB/s bidirectional
  SYS:               ~12 GB/s (PCIe 4.0 x16 through CPU)

This shows why NVLink matters: 25x higher bandwidth!
""")


def analyze_topology(connections: Dict[Tuple[int, int], str], num_gpus: int) -> None:
    """Analyze and report on the topology."""
    print_header("TOPOLOGY ANALYSIS")

    # Count NVLink connections
    nvlink_pairs = []
    pcie_pairs = []

    for (g1, g2), conn in connections.items():
        if g1 < g2:  # Avoid double counting
            if conn.startswith('NV'):
                nvlink_pairs.append((g1, g2, conn))
            elif conn in ['SYS', 'NODE', 'PHB', 'PXB', 'PIX']:
                pcie_pairs.append((g1, g2, conn))

    print(f"\nNVLink Connections ({len(nvlink_pairs)} pairs):")
    if nvlink_pairs:
        for g1, g2, conn in sorted(nvlink_pairs):
            num_links = conn.replace('NV', '')
            print(f"  GPU{g1} <-> GPU{g2}: {conn} ({num_links} links)")
    else:
        print("  None detected")

    print(f"\nPCIe-only Connections ({len(pcie_pairs)} pairs):")
    if pcie_pairs:
        for g1, g2, conn in sorted(pcie_pairs):
            print(f"  GPU{g1} <-> GPU{g2}: {conn}")
    else:
        print("  None (all pairs have NVLink)")

    # Recommendations
    print_header("RECOMMENDATIONS")

    if len(nvlink_pairs) == (num_gpus * (num_gpus - 1)) // 2:
        print("✓ All GPU pairs connected via NVLink")
        print("  → Ideal for all-reduce operations")
        print("  → Can use any GPU grouping")
    elif nvlink_pairs:
        print("⚠ Mixed NVLink/PCIe topology")
        print("  → Group NVLink-connected GPUs together when possible")
        print("  → Use process groups to exploit fast connections")
    else:
        print("⚠ No NVLink detected")
        print("  → Performance will be limited by PCIe bandwidth")
        print("  → Consider using smaller batch sizes to hide communication")


def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " GPU TOPOLOGY INSPECTOR".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # Check if we have nvidia-smi
    if not check_nvidia_smi():
        print("\n[INFO] nvidia-smi not found. Showing simulated topology.")
        print_simulated_topology()
        return

    # Get GPU count
    num_gpus = get_gpu_count()
    if num_gpus == 0:
        print("\n[INFO] No NVIDIA GPUs detected. Showing simulated topology.")
        print_simulated_topology()
        return

    # Get GPU information
    print_header("GPU INFORMATION")
    gpus = get_gpu_info()
    for gpu in gpus:
        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        print(f"  Memory: {gpu['memory_mb']} MB")
        print(f"  PCI Bus: {gpu['pci_bus']}")

    # Get topology matrix
    print_header("TOPOLOGY MATRIX")
    topo_matrix = get_topology_matrix()
    if topo_matrix:
        print(topo_matrix)
        connections = parse_topology_matrix(topo_matrix)
        analyze_topology(connections, num_gpus)
    else:
        print("Could not retrieve topology matrix")

    # Get NVLink status
    nvlink_status = get_nvlink_status()
    if nvlink_status and "NVLINK" not in nvlink_status:
        print_header("NVLINK STATUS (GPU 0)")
        print(nvlink_status)

    # PyTorch CUDA information
    print_header("PYTORCH CUDA INFO")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count (torch): {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
                print(f"  Multi-processor count: {props.multi_processor_count}")
        else:
            print("PyTorch CUDA not available")
    except ImportError:
        print("PyTorch not installed")

    # Summary
    print_header("QUICK REFERENCE")
    print("""
NCCL Environment Variables for Tuning:
  NCCL_DEBUG=INFO           Show what NCCL is doing
  NCCL_ALGO=Ring            Force ring algorithm
  NCCL_ALGO=Tree            Force tree algorithm
  NCCL_NTHREADS=256         Set thread count
  NCCL_P2P_DISABLE=1        Disable peer-to-peer (for debugging)

Common Commands:
  nvidia-smi topo -m        Show topology matrix
  nvidia-smi nvlink --status Show NVLink connections
  nvidia-smi -q -d MEMORY   Show memory usage details
""")


if __name__ == "__main__":
    main()
