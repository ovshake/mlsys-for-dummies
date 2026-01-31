# Chapter 6: Tensor Parallelism from Scratch

> *"When your layer doesn't fit on one GPU, you split the layer, not just the data."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain why tensor parallelism is needed for large models
- Implement column-parallel and row-parallel linear layers
- Understand how Megatron-LM partitions transformer layers
- Calculate communication costs for different TP strategies

## Prerequisites

- Completed [Chapter 5 (Data Parallelism)](./chapter05.md)
- Linear algebra (matrix multiplication)
- Understanding of transformer architecture (attention, MLP)

## Concept Overview

### Why Tensor Parallelism?

Data parallelism replicates the entire model. But what if a single layer is too big?

Consider GPT-3's embedding layer:
- Vocabulary: 50,000 tokens
- Hidden dimension: 12,288
- Size: 50,000 × 12,288 × 2 bytes = **1.2 GB** (just for embeddings!)

For very large models, even a single linear layer might exceed GPU memory. Tensor parallelism splits individual layers across GPUs.

### The Key Insight: Matrix Multiplication is Parallelizable

Matrix multiplication `Y = XW` can be computed in parts:

**Column-wise splitting** (split W by columns):
```
W = [W₁ | W₂]  (split into left and right halves)

Y = X × [W₁ | W₂] = [X×W₁ | X×W₂] = [Y₁ | Y₂]
```
Each GPU computes part of the output. No communication needed—just concatenate!

**Row-wise splitting** (split W by rows):
```
W = [W₁]       (split into top and bottom halves)
    [W₂]

Y = X × [W₁; W₂] requires splitting X too...
```
This needs an all_reduce to combine partial results.

### Megatron-Style Tensor Parallelism

Megatron-LM (NVIDIA's framework) cleverly combines column and row splits to minimize communication:

**MLP Block** (in a transformer layer):
```
MLP(X) = GeLU(X × W₁) × W₂

GPU 0: Y₁ = GeLU(X × W₁ᶜᵒˡ⁰) × W₂ʳᵒʷ⁰
GPU 1: Y₁ = GeLU(X × W₁ᶜᵒˡ¹) × W₂ʳᵒʷ¹

Y = all_reduce(Y₀ + Y₁)
```

The trick: Column-parallel first, row-parallel second!
- After column-parallel W₁: each GPU has part of the hidden states (no comm needed)
- After row-parallel W₂: need all_reduce to sum partial products

**Only ONE all_reduce per MLP block!**

### Attention Layer Tensor Parallelism

For multi-head attention with 32 heads on 4 GPUs:
- Each GPU handles 8 attention heads
- Q, K, V projections: column-parallel (each GPU computes 8 heads worth)
- Output projection: row-parallel (combine head outputs)

```
               ┌─────────────────────────────────────────────┐
               │            Multi-Head Attention              │
               │                                              │
               │   Heads 0-7    Heads 8-15   Heads 16-23  Heads 24-31
               │   ┌─────┐     ┌─────┐      ┌─────┐      ┌─────┐
    Input X ───►   │GPU 0│     │GPU 1│      │GPU 2│      │GPU 3│
               │   └──┬──┘     └──┬──┘      └──┬──┘      └──┬──┘
               │      │           │            │            │
               │      └───────────┴────────────┴────────────┘
               │                        │
               │                   all_reduce
               │                        │
               │                        ▼
               │                    Output
               └─────────────────────────────────────────────┘
```

### Communication Analysis

For a transformer layer with tensor parallelism degree T:

| Component | Communication Volume |
|-----------|---------------------|
| MLP forward | 2 × batch × seq × hidden / T (all_reduce) |
| MLP backward | 2 × batch × seq × hidden / T (all_reduce) |
| Attention forward | 2 × batch × seq × hidden / T (all_reduce) |
| Attention backward | 2 × batch × seq × hidden / T (all_reduce) |

**Total per layer**: 8 × batch × seq × hidden × (T-1) / T bytes

This is why TP is typically used within a node (NVLink), not across nodes (slow InfiniBand).

### The Math: Column-Parallel Linear

```python
class ColumnParallelLinear:
    """
    Split the weight matrix W by columns.

    W_full shape: [in_features, out_features]
    W_local shape: [in_features, out_features // tp_size]

    Forward: Y_local = X @ W_local
    No communication needed in forward!
    """

    def forward(self, X):
        # Each GPU computes its portion of the output
        return X @ self.weight  # shape: [batch, out_features // tp_size]
```

### The Math: Row-Parallel Linear

```python
class RowParallelLinear:
    """
    Split the weight matrix W by rows.

    W_full shape: [in_features, out_features]
    W_local shape: [in_features // tp_size, out_features]

    Forward: Y_partial = X_local @ W_local
             Y = all_reduce(Y_partial)
    """

    def forward(self, X_local):
        # Each GPU has part of input, computes partial output
        Y_partial = X_local @ self.weight
        # Sum across all GPUs
        dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)
        return Y_partial
```

### Combining Column + Row: The MLP Recipe

```python
def tp_mlp_forward(X, W1_col, W2_row, tp_group):
    """
    Tensor-parallel MLP with minimal communication.

    W1 is column-parallel: [hidden, 4*hidden//tp_size]
    W2 is row-parallel: [4*hidden//tp_size, hidden]
    """
    # Step 1: Column-parallel first linear
    hidden = torch.relu(X @ W1_col)  # No comm needed!

    # Step 2: Row-parallel second linear
    output = hidden @ W2_row

    # Step 3: Only ONE all_reduce needed
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)

    return output
```

### TP vs DP: When to Use Which?

| Factor | Data Parallel | Tensor Parallel |
|--------|---------------|-----------------|
| **Granularity** | Whole model | Single layer |
| **Communication** | Gradients only | Activations every layer |
| **Scalability** | 100s of GPUs | Usually ≤8 GPUs |
| **Best for** | Batch scaling | Large layers |
| **Topology** | Cross-node OK | Intra-node (NVLink) |

**Rule of thumb**: TP within node, DP across nodes.

## Code Walkthrough

### Script 1: tp_linear.py

Implements column-parallel and row-parallel linear layers from scratch:
- Shows weight initialization and sharding
- Demonstrates forward pass with all_reduce
- Verifies correctness against non-parallel version

### Script 2: tp_mlp.py

A complete tensor-parallel MLP block:
- Combines column and row parallelism
- Shows how to minimize communication
- Compares performance with naive approach

## Common Pitfalls

### Pitfall 1: Forgetting to Split Inputs for Row-Parallel

Row-parallel expects the input to already be split. If you feed the full input, you'll get wrong results!

### Pitfall 2: Wrong Reduction Order

All_reduce must happen at the right place:
- After row-parallel layer
- NOT after column-parallel layer

### Pitfall 3: Mismatched Dimensions

When transitioning from column to row parallel:
- Column output shape: `[batch, hidden // tp_size]`
- Row input shape: `[batch, hidden // tp_size]`

These must match!

## Try It Yourself

### Exercise 1: Verify Column-Parallel Correctness

Run `tp_linear.py` and verify that:
```
concatenate(column_parallel_outputs) == full_linear_output
```

### Exercise 2: Count All-Reduces

Count the number of all_reduce calls in a full transformer layer with:
- TP degree = 4
- 12 attention heads
- 4096 hidden dimension

### Exercise 3: Measure TP Overhead

Modify `tp_mlp.py` to measure:
1. Time for matrix multiplications
2. Time for all_reduce calls
3. Communication percentage

## Key Takeaways

1. **TP splits layers, not batches** - Complementary to data parallelism
2. **Column-parallel needs no sync in forward** - Output is naturally partitioned
3. **Row-parallel needs all_reduce** - To sum partial products
4. **Megatron trick: column then row** - Minimizes communication to 2 all_reduces per MLP
5. **TP best within a node** - Needs high bandwidth (NVLink)

## Performance Intuition

For a 4-GPU TP setup with NVLink (900 GB/s total):
- MLP computation: ~1ms
- All-reduce (2MB activations): ~0.01ms

TP overhead is typically <5% within a node. But across nodes with InfiniBand (50 GB/s), it would be 10x slower!

## What's Next?

In [Chapter 7](./chapter07.md), we'll explore **Pipeline Parallelism** and **Expert Parallelism**—splitting models by layers and routing tokens to specialized experts.

## Further Reading

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [Tensor Parallelism in PyTorch](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
