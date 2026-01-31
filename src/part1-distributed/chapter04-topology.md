# topology_inspector.py

> Inspect your GPU topology and understand communication paths

This script examines your GPU setup and reports on NVLink connections, PCIe topology, and NUMA affinity.

## What It Does

1. Detects available GPUs and their properties
2. Identifies NVLink connections between GPU pairs
3. Maps PCIe topology (bridges, switches)
4. Shows CPU/NUMA affinity for each GPU
5. Suggests optimal process placement

## Run It

```bash
python tutorial/part1-distributed/chapter04-nccl-topology/scripts/topology_inspector.py
```

## Example Output (8-GPU DGX)

```
=== GPU Topology Inspector ===

Found 8 GPUs:
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
  ...

NVLink Connections:
  GPU 0 <--NV12--> GPU 1
  GPU 0 <--NV12--> GPU 2
  GPU 0 <--NV12--> GPU 3
  ...

PCIe Topology:
  GPU 0-3: Same PCIe switch (fast)
  GPU 4-7: Same PCIe switch (fast)
  GPU 0-4: Cross-switch (slower)

NUMA Affinity:
  GPU 0-3: NUMA node 0 (CPUs 0-31)
  GPU 4-7: NUMA node 1 (CPUs 32-63)

Recommendations:
  - For 4-GPU jobs, use GPUs 0-3 or 4-7 (same switch)
  - For 8-GPU jobs, expect ~10% overhead from cross-switch communication
```

## Why This Matters

Understanding topology helps you:
- **Place processes optimally** - Keep communicating processes on fast interconnects
- **Debug performance issues** - Unexpectedly slow? Check if you're using PCIe instead of NVLink
- **Choose parallelism strategy** - Tensor parallel works best with NVLink

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter04-nccl-topology/scripts/topology_inspector.py}}
```
