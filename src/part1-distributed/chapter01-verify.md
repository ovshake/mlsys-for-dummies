# verify_setup.py

> Check if your environment is ready for distributed computing

This script verifies that PyTorch is installed correctly and checks for distributed computing capabilities.

## What It Does

1. Checks PyTorch installation and version
2. Detects CUDA availability and GPU count
3. Lists supported distributed backends (NCCL, Gloo, MPI)
4. Provides recommendations based on your setup

## Run It

```bash
python tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py
```

## Source Code

```python
{{#include ../../tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py}}
```
