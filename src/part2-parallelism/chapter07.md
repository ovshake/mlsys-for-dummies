# Chapter 7: Pipeline and Expert Parallelism

> *"When one GPU can't hold one layer (TP), we split layers. When it can't hold all layers (PP), we split the model vertically."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain how pipeline parallelism splits models across GPUs
- Implement 1F1B scheduling for efficient pipeline execution
- Understand Mixture-of-Experts (MoE) and Expert Parallelism
- Calculate the optimal parallelism strategy for a given model

## Prerequisites

- Completed [Chapters 5-6](./chapter05.md) (Data and Tensor Parallelism)
- Understanding of transformer architecture
- Basic knowledge of GPU memory hierarchy

## Concept Overview

### Pipeline Parallelism: Splitting by Layers

While tensor parallelism splits individual layers horizontally, **pipeline parallelism** splits the model vertically—each GPU holds a contiguous group of layers.

```
Full Model: [Embed] [Layer 0-5] [Layer 6-11] [Layer 12-17] [Layer 18-23] [Head]
                ↓         ↓           ↓            ↓            ↓         ↓
Pipeline:   ┌──────┐ ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐   ┌──────┐
            │GPU 0 │ │GPU 1 │   │GPU 2 │    │GPU 3 │    │GPU 4 │   │GPU 5 │
            │Stage 0│ │Stage 1│  │Stage 2│   │Stage 3│   │Stage 4│  │Stage 5│
            └──────┘ └──────┘   └──────┘    └──────┘    └──────┘   └──────┘
```

Communication: Activations flow forward, gradients flow backward (point-to-point `send`/`recv`).

### The Pipeline Bubble Problem

Naive pipeline execution has a fatal flaw: **bubbles**.

```
Time →
GPU 0: [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]
GPU 1:      [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]
GPU 2:           [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]
GPU 3:                [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]

Bubbles = empty time where GPUs are idle
```

**Bubble fraction** = (P-1) / (M + P - 1), where P = pipeline stages, M = microbatches.

For P=4, M=4: Bubble = 3/7 = 43% wasted time!

### 1F1B Scheduling: The Solution

**1F1B** (One Forward, One Backward) interleaves forward and backward passes to reduce bubbles:

```
Time →
GPU 0: [F0] [F1] [F2] [F3] [B0] [F4] [B1] [F5] [B2] [B3]
GPU 1:      [F0] [F1] [F2] [B0] [F3] [B1] [F4] [B2] [B3]
GPU 2:           [F0] [F1] [B0] [F2] [B1] [F3] [B2] [B3]
GPU 3:                [F0] [B0] [F1] [B1] [F2] [B2] [F3] [B3]
```

Key insight: Once the pipeline is "full," each GPU does one forward then one backward, keeping memory constant.

### Memory in Pipeline Parallelism

Each GPU stores:
- Model parameters for its stages
- Activations from forward pass (needed for backward)

**1F1B memory advantage**: Only need to store activations for `P` microbatches, not `M`.

### Mixture of Experts (MoE)

MoE replaces the standard FFN with multiple "expert" FFNs:

```
Standard FFN:
    Input → FFN → Output

MoE FFN:
    Input → Router → Expert 0 →
                   → Expert 1 → Weighted Sum → Output
                   → Expert 2 →
                   → Expert 3 →
```

The **router** (a small neural network) decides which experts process each token. Typically, only top-K experts (K=1 or 2) are activated per token.

**Why MoE?**
- More parameters without more FLOPs
- Each token only activates a fraction of parameters
- DeepSeek-V3: 671B parameters but only 37B activated per token!

### Expert Parallelism (EP)

When you have 64+ experts, they don't fit on one GPU. **Expert Parallelism** distributes experts across GPUs:

```
              Token Routing
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │ GPU 0 │  │ GPU 1 │  │ GPU 2 │
    │E0,E1,E2│ │E3,E4,E5│ │E6,E7,E8│
    └───────┘  └───────┘  └───────┘
        │          │          │
        └──────────┴──────────┘
              All-to-All
              (collect results)
```

Communication pattern: **All-to-All** (each GPU sends tokens to the GPUs hosting the selected experts).

### EP vs TP: A Critical Comparison

For MoE models, EP is often better than TP:

| Aspect | Tensor Parallelism | Expert Parallelism |
|--------|-------------------|-------------------|
| **What's split** | Each expert matrix | Whole experts |
| **Communication** | 2 all-reduce per layer | 2 all-to-all per layer |
| **Volume** | 2 × batch × seq × hidden | 2 × k × batch × seq × hidden / N |
| **Compute efficiency** | Low (small GEMMs) | High (full expert GEMMs) |

**Key insight**: TP slices already small expert matrices, making GEMMs inefficient. EP keeps expert matrices whole.

### Communication Volume Deep Dive

For TP with degree T on an MoE layer:
```
Volume = 2S (all-reduce, activations of size S)
```

For EP with N experts, k activated:
```
Volume = 2kS/N (all-to-all, only k/N of tokens go to each GPU)
```

When k << N (sparse activation), EP wins on communication too!

### Combining Parallelisms: The 3D Approach

Real large-model training uses multiple parallelism strategies:

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PARALLELISM                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                PIPELINE PARALLELISM                   │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │TENSOR PARA- │ │TENSOR PARA- │ │TENSOR PARA- │     │  │
│  │  │LLELISM      │ │LLELISM      │ │LLELISM      │     │  │
│  │  │ (Stage 0)   │→│ (Stage 1)   │→│ (Stage 2)   │     │  │
│  │  │  8 GPUs     │ │  8 GPUs     │ │  8 GPUs     │     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                     (Replica 0)                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                      ... more replicas ...            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Rule of thumb**:
- TP: Within a node (needs NVLink)
- PP: Across nodes (lower bandwidth OK)
- DP: Scales indefinitely (gradient sync)

## Code Walkthrough

### Script 1: pipeline_schedule_viz.py

Visualizes different pipeline schedules:
- Naive (fill-drain)
- 1F1B
- Interleaved 1F1B

Shows bubble ratios and memory usage.

### Script 2: parallel_strategy_calculator.py

Given model specs and hardware, calculates:
- Memory per GPU for each parallelism strategy
- Communication volume
- Recommended configuration

## Try It Yourself

### Exercise 1: Calculate Bubble Fraction

For a 4-stage pipeline with 16 microbatches:
1. What's the bubble fraction with naive scheduling?
2. How does it improve with 1F1B?

### Exercise 2: MoE Communication Analysis

For an MoE layer with:
- 64 experts
- top-2 routing (k=2)
- 4096 hidden dimension
- Batch of 4 × 1024 tokens

Calculate communication volume for:
1. 8-way TP (splitting each expert)
2. 8-way EP (8 experts per GPU)

### Exercise 3: Design a Parallelism Strategy

You have:
- 70B parameter dense model
- 64 H100 GPUs (8 per node)
- 80GB memory per GPU

Design a parallelism strategy. Consider:
- Model size: ~140GB in FP16
- Activations and gradients
- Communication patterns

## Key Takeaways

1. **PP splits the model by layers** - Point-to-point communication only
2. **Bubbles are the enemy** - 1F1B scheduling minimizes idle time
3. **MoE = sparse activation** - More parameters, same compute
4. **EP beats TP for MoE** - Keeps expert matrices whole
5. **Combine strategies** - Real systems use TP + PP + DP + EP

## The Parallelism Decision Tree

```
Is one layer too big for one GPU?
├─ Yes → Use Tensor Parallelism (within node)
└─ No
    └─ Is the full model too big for one GPU?
       ├─ Yes → Use Pipeline Parallelism (across nodes OK)
       │        + Use Tensor Parallelism if layers are large
       └─ No
           └─ Use Data Parallelism (scales indefinitely)

Is the model MoE?
├─ Yes → Add Expert Parallelism (across nodes OK)
└─ No → Continue with above strategy
```

## What's Next?

In Part III, we'll dive into **LLM Inference Systems**—how to efficiently serve models after training. This includes KV cache management, batching strategies, and speculative decoding.

## Further Reading

- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [PipeDream Paper](https://arxiv.org/abs/1806.03377)
- [Switch Transformer (MoE)](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
