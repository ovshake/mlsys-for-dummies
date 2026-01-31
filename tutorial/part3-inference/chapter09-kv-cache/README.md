# Chapter 9: KV Cache Management

> *"In LLM inference, memory is the new compute. And KV cache is the memory hog."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain why KV cache exists and how it accelerates inference
- Calculate KV cache size for different models and contexts
- Understand PagedAttention and its benefits
- Explain how RadixCache enables prefix sharing

## Prerequisites

- Completed Chapter 8 (Server Anatomy)
- Understanding of transformer attention mechanism
- Basic knowledge of memory management

## Concept Overview

### Why KV Cache?

In transformers, each attention layer computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

During generation, we produce one token at a time. Without caching:
- Token 1: Compute K, V for position 0
- Token 2: Compute K, V for positions 0, 1 (recompute position 0!)
- Token 3: Compute K, V for positions 0, 1, 2 (recompute again!)
- ...
- Token N: O(N²) total computations!

With KV cache:
- Token 1: Compute K₀, V₀, store in cache
- Token 2: Compute K₁, V₁, concatenate with cached K₀, V₀
- Token 3: Compute K₂, V₂, concatenate with cached
- ...
- Token N: O(N) computations

**KV cache trades memory for compute.**

### KV Cache Size Calculation

For each token, we store K and V for every layer:

```
KV per token = 2 × num_layers × num_heads × head_dim × dtype_size

Example (LLaMA-70B):
  Layers: 80
  Heads: 64 (8 KV heads with GQA)
  Head dim: 128
  Dtype: FP16 (2 bytes)

  KV per token = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB

For 32K context:
  KV per request = 320 KB × 32K = 10.24 GB
```

A single request needs **10 GB** of KV cache! This is why memory management is critical.

### The Memory Fragmentation Problem

Traditional approach: Pre-allocate maximum context length per request.

```
Request A (needs 1K): [■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□] 32K allocated
Request B (needs 2K): [■■■■■■■■□□□□□□□□□□□□□□□□□□□□□□□□] 32K allocated
Request C (needs 1K): [■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□] 32K allocated

Total allocated: 96K tokens worth of memory
Actually used: 4K tokens
Waste: 96%!
```

This is **internal fragmentation**—memory reserved but unused.

### PagedAttention: The Solution

PagedAttention (from vLLM) applies OS-style virtual memory to KV cache:

```
Physical Memory (Pages):
[Page 0][Page 1][Page 2][Page 3][Page 4][Page 5][Page 6][Page 7]

Request A (logical view):     Request B (logical view):
[Tokens 0-255][Tokens 256-511] [Tokens 0-255][Tokens 256-511][Tokens 512-767]
      ↓              ↓               ↓              ↓              ↓
   Page 2         Page 5          Page 0         Page 3         Page 7
   (physical)    (physical)      (physical)     (physical)     (physical)
```

**Key insight**: Allocate physical pages only when needed. Different requests can share the same physical memory pool.

Benefits:
- Near-zero fragmentation
- Memory utilization > 95%
- More concurrent requests

### The Three-Level KV Cache Hierarchy

Modern systems like SGLang use a three-level structure:

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: RadixCache (Logical)                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Radix Tree of Token Sequences                                │ │
│ │                                                              │ │
│ │            [root]                                           │ │
│ │           /      \                                          │ │
│ │    "What is"    "Tell me"                                   │ │
│ │       /              \                                       │ │
│ │  "the capital"    "a joke"                                  │ │
│ │                                                              │ │
│ │ Purpose: Detect prefix sharing opportunities                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: ReqToTokenPool (Mapping)                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [request_id, token_position] → physical_slot_index          │ │
│ │                                                              │ │
│ │ (req_0, pos_0) → slot_42                                    │ │
│ │ (req_0, pos_1) → slot_17                                    │ │
│ │ (req_1, pos_0) → slot_42  ← Same slot! Prefix sharing!      │ │
│ │                                                              │ │
│ │ Purpose: Map logical positions to physical memory           │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: TokenToKVPool (Physical GPU Memory)                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ slot_index → actual K,V tensors on GPU                      │ │
│ │                                                              │ │
│ │ Slot 0:  [K tensor][V tensor]                               │ │
│ │ Slot 1:  [K tensor][V tensor]                               │ │
│ │ ...                                                          │ │
│ │ Slot N:  [K tensor][V tensor]                               │ │
│ │                                                              │ │
│ │ Purpose: Store actual KV values                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### RadixCache: Automatic Prefix Sharing

RadixCache enables **automatic prefix caching**. If multiple requests share a prompt prefix, they share KV cache:

```
Request 1: "What is the capital of France?"
Request 2: "What is the capital of Germany?"
Request 3: "What is the largest planet?"

Radix Tree:
                    [root]
                      │
              "What is the"  (shared by all 3!)
                 /    \
        "capital of"  "largest planet"
            /    \          │
      "France"  "Germany"  [Request 3]
         │         │
    [Request 1] [Request 2]

Memory savings: "What is the" KV cache stored ONCE, used by 3 requests!
```

This is huge for:
- System prompts (shared across all requests)
- Few-shot examples
- Chat history prefixes

### Cache Eviction: LRU with Reference Counting

When memory is full, we need to evict cached entries. RadixCache uses:

1. **Reference counting**: Don't evict entries in use
2. **LRU (Least Recently Used)**: Evict oldest unused entries first

```python
class RadixCacheEntry:
    tokens: List[int]
    kv_indices: List[int]
    ref_count: int  # Number of requests using this
    last_access: float  # For LRU

def evict(self):
    # Find entries with ref_count == 0
    candidates = [e for e in self.entries if e.ref_count == 0]

    # Sort by last access time
    candidates.sort(key=lambda e: e.last_access)

    # Evict oldest first
    for entry in candidates:
        if self.memory_available():
            break
        self.free(entry)
```

## Code Walkthrough

### Script 1: kv_cache_calculator.py

Calculate KV cache sizes for different model configurations:
- Shows memory requirements per token, per request
- Estimates concurrent request capacity
- Compares with and without paging

### Script 2: prefix_sharing_demo.py

Demonstrates how prefix sharing works:
- Shows memory savings from shared prefixes
- Visualizes the radix tree structure
- Calculates sharing efficiency

## Memory Budget Planning

For a 70B model on 8× H100 (640 GB total):

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | 140 GB |
| CUDA kernels, etc. | ~20 GB |
| **Available for KV cache** | **~480 GB** |

With 320 KB per token per request:
- Max tokens in cache: 480 GB / 320 KB = 1.5M tokens
- At 4K avg context: ~375 concurrent requests
- At 32K context: ~47 concurrent requests

**This is why context length dramatically affects capacity!**

## Try It Yourself

### Exercise 1: Calculate Your Model's KV Cache

For your favorite model (LLaMA, Mistral, etc.):
1. Find: num_layers, num_kv_heads, head_dim
2. Calculate: KV bytes per token
3. Calculate: Max requests at 8K context with 80GB GPU

### Exercise 2: Measure Prefix Sharing Savings

Design a benchmark:
1. Create 100 requests with shared system prompt
2. Calculate memory with individual caching
3. Calculate memory with prefix sharing
4. What's the savings percentage?

### Exercise 3: Implement Simple LRU Cache

Implement a basic LRU cache for KV entries:
- Fixed capacity
- Reference counting
- Eviction when full

## Key Takeaways

1. **KV cache is massive** - Often larger than model weights for long contexts
2. **Fragmentation wastes memory** - Pre-allocation is inefficient
3. **PagedAttention solves fragmentation** - Near-100% memory utilization
4. **Prefix sharing saves memory** - Especially for system prompts
5. **Memory limits concurrency** - More memory = more concurrent requests

## Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Pre-allocation | Simple, no overhead | Massive fragmentation |
| PagedAttention | Low fragmentation | Page table overhead |
| RadixCache | Prefix sharing | Tree management overhead |
| Quantized KV | Less memory | Slight quality loss |

## What's Next?

In Chapter 10, we'll explore **Advanced Scheduling and CUDA Graphs**—how to hide scheduling overhead and maximize GPU utilization.

## Further Reading

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [SGLang RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- Original source: [`sglang/scheduler/readme-en.md`](../../../sglang/scheduler/readme-en.md)
