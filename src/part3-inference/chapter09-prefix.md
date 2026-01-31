# prefix_sharing_demo.py

> Demonstrate memory savings from shared prompt prefixes

This script shows how RadixCache saves memory by sharing KV cache for common prompt prefixes.

## What It Does

1. Creates multiple requests with shared prefixes
2. Shows memory usage WITHOUT prefix sharing
3. Shows memory usage WITH prefix sharing
4. Visualizes the radix tree structure

## Run It

```bash
python tutorial/part3-inference/chapter09-kv-cache/scripts/prefix_sharing_demo.py
```

## Example Output

```
=== Prefix Sharing Demo ===

Requests:
  1. "You are a helpful assistant. What is 2+2?"
  2. "You are a helpful assistant. Explain quantum computing."
  3. "You are a helpful assistant. Write a poem."

Shared Prefix: "You are a helpful assistant. " (7 tokens)

Memory Analysis:
  Without sharing:
    Request 1: 100 tokens × 320 KB = 32 MB
    Request 2: 120 tokens × 320 KB = 38.4 MB
    Request 3: 90 tokens × 320 KB = 28.8 MB
    Total: 99.2 MB

  With sharing:
    Shared prefix: 7 tokens × 320 KB = 2.24 MB (stored once)
    Request 1 unique: 93 tokens × 320 KB = 29.76 MB
    Request 2 unique: 113 tokens × 320 KB = 36.16 MB
    Request 3 unique: 83 tokens × 320 KB = 26.56 MB
    Total: 94.72 MB

  Savings: 4.5% (increases with more requests sharing the prefix!)

Radix Tree:
         [root]
            │
    "You are a helpful assistant."
         /      |      \
   "What is"  "Explain"  "Write"
      │          │         │
   "2+2?"   "quantum"   "a poem"
```

## Why This Matters

With 100 requests sharing a system prompt:
- Without sharing: 100× full prompt
- With sharing: 1× shared + 100× unique parts
- Savings: Up to 90%+ for long system prompts!

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter09-kv-cache/scripts/prefix_sharing_demo.py}}
```
