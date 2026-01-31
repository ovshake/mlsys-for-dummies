# kv_cache_calculator.py

> Calculate KV cache memory requirements for any model

This script helps you understand how much memory your KV cache will consume and plan your deployment accordingly.

## What It Does

1. Takes model parameters (layers, heads, head_dim, dtype)
2. Calculates KV cache size per token
3. Estimates memory for different context lengths
4. Shows concurrent request capacity

## Run It

```bash
python tutorial/part3-inference/chapter09-kv-cache/scripts/kv_cache_calculator.py
```

## Example Output

```
=== KV Cache Calculator ===

Model: LLaMA-70B
  Layers: 80
  KV Heads: 8 (GQA)
  Head Dim: 128
  Dtype: FP16

KV Cache Size:
  Per token: 320 KB
  Per request (4K context): 1.28 GB
  Per request (32K context): 10.24 GB

With 80 GB GPU Memory:
  Model weights (FP16): 140 GB (requires 2+ GPUs)
  After weights on 8x H100: ~480 GB available

  Max concurrent requests:
    At 4K context: 375 requests
    At 8K context: 187 requests
    At 32K context: 46 requests

Warning: Long context dramatically reduces concurrency!
```

## The Formula

```
kv_bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes
                     ↑   ↑                             ↑
                     K+V layers                        2 for FP16
```

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter09-kv-cache/scripts/kv_cache_calculator.py}}
```
