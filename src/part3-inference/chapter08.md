# Chapter 8: Anatomy of an LLM Inference Server

> *"Training is a sprint. Inference is a marathon that never ends."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Trace the lifecycle of a request through an inference server
- Explain the roles of Tokenizer, Scheduler, and Model Runner
- Understand why inference is fundamentally different from training
- Identify bottlenecks in inference serving

## Prerequisites

- Completed Part II (Parallelism Strategies)
- Basic understanding of transformer architecture
- Familiarity with REST APIs

## Concept Overview

### Training vs Inference: A Tale of Two Challenges

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Update model weights | Generate tokens |
| **Batch size** | Fixed (large) | Dynamic (varies) |
| **Latency** | Irrelevant | Critical |
| **Throughput** | Samples/second | Tokens/second |
| **Memory** | Dominated by gradients | Dominated by KV cache |
| **Workload** | Homogeneous | Heterogeneous |

Training processes fixed batches for hours. Inference serves arbitrary requests in milliseconds.

### The Inference Pipeline

When you send a prompt to an LLM, here's what happens:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LLM INFERENCE SERVER                               │
│                                                                          │
│  HTTP Request ─────────────────────────────────────────► HTTP Response   │
│       │                                                        ▲         │
│       ▼                                                        │         │
│  ┌─────────────┐    ┌───────────────┐    ┌─────────────────┐  │         │
│  │ API Adapter │───►│TokenizerMgr   │───►│   Scheduler     │  │         │
│  │             │    │(tokenize)     │    │(batch requests) │  │         │
│  └─────────────┘    └───────────────┘    └───────┬─────────┘  │         │
│                                                   │            │         │
│                                                   ▼            │         │
│                                           ┌─────────────────┐  │         │
│                                           │  Model Runner   │  │         │
│                                           │ (GPU compute)   │  │         │
│                                           └───────┬─────────┘  │         │
│                                                   │            │         │
│                                                   ▼            │         │
│                                           ┌─────────────────┐  │         │
│                                           │DetokenizerMgr   │──┘         │
│                                           │(tokens→text)    │            │
│                                           └─────────────────┘            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Deep Dive

#### 1. API Adapter

Translates HTTP requests into internal format:
- Parses JSON body
- Validates parameters (temperature, max_tokens, etc.)
- Creates `GenerateRequest` object

```python
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # Validate and convert to internal format
    generate_request = convert_to_internal(request)
    # Send to tokenizer manager
    return await tokenizer_manager.generate(generate_request)
```

#### 2. Tokenizer Manager

Handles text ↔ token conversion:
- Tokenizes input prompt
- Manages vocabulary and special tokens
- Queues tokenized requests for scheduler

#### 3. Scheduler

The brain of the inference server:
- Manages request queue
- Decides which requests to batch together
- Allocates KV cache memory
- Chooses between prefill and decode

The scheduler is so important it gets its own chapters (9-10)!

#### 4. Model Runner

Executes the actual neural network:
- Loads model weights
- Runs forward pass
- Samples next token

#### 5. Detokenizer Manager

Converts tokens back to text:
- Decodes token IDs to strings
- Handles streaming output
- Manages stop sequences

### The Two Phases of Inference

LLM inference has two distinct phases:

**Phase 1: Prefill (Prompt Processing)**
```
Input:  "What is the capital of France?"
        [token_0, token_1, token_2, ..., token_n]

Output: KV cache for all tokens + first generated token

Compute: Parallelizable (all tokens at once)
Memory: Write n entries to KV cache
```

**Phase 2: Decode (Token Generation)**
```
Input:  Previously generated token + KV cache
        [token_i]

Output: Next token

Compute: Sequential (one token at a time)
Memory: Read from KV cache, write 1 entry
```

```
Time →

Prefill:  [===================] (process all prompt tokens)
                               ↓
Decode:   [=] [=] [=] [=] [=] [=] [=] [=] ...
          t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈
```

**Key insight**: Prefill is compute-bound, decode is memory-bound.

### Why Batching is Complicated

Training batches are simple: same sequence length, process together.

Inference batches are hard:
- Requests arrive at different times
- Different prompt lengths
- Different desired output lengths
- Some requests finish mid-batch

**Continuous batching** solves this:
```
Time →
Request A: [====prefill====][d][d][d][d][d][done]
Request B:         [prefill][d][d][d][d][d][d][d][d]...
Request C:                      [====prefill====][d][d]...

Batched execution:
[A+B prefill] [A+B decode] [A+B+C] [B+C decode] ...
```

### Memory: The Inference Bottleneck

For a 70B parameter model serving requests:

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | 140 GB |
| KV cache (per request) | ~2 GB for 32K context |
| Activations | ~1 GB |

With 140 GB of weights and 80 GB GPU memory... we need tensor parallelism just to load the model!

And each request needs its own KV cache. Serving 100 concurrent requests at 32K context would need 200 GB just for KV cache!

This is why KV cache management ([Chapter 9](./chapter09.md)) is critical.

## Code Walkthrough

### Script: minimal_inference_server.py

A simplified inference server showing the core components:
- Request queue management
- Simple batching
- Token-by-token generation

This isn't production-ready but demonstrates the architecture.

## Key Metrics

When evaluating inference servers:

| Metric | Definition | Target |
|--------|------------|--------|
| **TTFT** | Time To First Token | < 500ms |
| **ITL** | Inter-Token Latency | < 50ms |
| **Throughput** | Tokens/second | Maximize |
| **Concurrency** | Simultaneous requests | Maximize |

Trade-offs:
- Higher concurrency → higher memory → larger batches → higher throughput
- Higher concurrency → more KV cache → potential OOM
- Larger batches → higher latency per request

## Try It Yourself

### Exercise 1: Trace a Request

Using the `minimal_inference_server.py`:
1. Add logging to each component
2. Trace a single request through the system
3. Measure time spent in each stage

### Exercise 2: Measure Prefill vs Decode

Profile inference to measure:
1. Time for prefill (prompt processing)
2. Time per decode step
3. How does prompt length affect prefill time?
4. How does batch size affect decode time?

### Exercise 3: Calculate KV Cache Size

For a model with:
- 32 layers
- 8192 hidden dimension
- 128 heads
- 32K max sequence length

Calculate:
1. KV cache size per token
2. KV cache size for one 32K request
3. Max concurrent requests with 80 GB memory (after model weights)

## Key Takeaways

1. **Inference is a pipeline** - Multiple stages, each can be a bottleneck
2. **Prefill vs Decode** - Different compute characteristics, different optimizations
3. **Memory dominates** - KV cache limits concurrency
4. **Batching is complex** - Continuous batching enables high throughput
5. **Latency matters** - Unlike training, users are waiting

## The Inference Optimization Hierarchy

```
Level 0: Model works (correctness)
    ↓
Level 1: Model fits in memory (quantization, TP)
    ↓
Level 2: Efficient memory management (KV cache, paging)
    ↓
Level 3: Efficient batching (continuous batching)
    ↓
Level 4: Kernel optimizations (FlashAttention, CUDA graphs)
    ↓
Level 5: Speculative decoding (draft models)
```

## What's Next?

In [Chapter 9](./chapter09.md), we'll dive deep into **KV Cache Management**—how systems like PagedAttention and RadixCache enable serving many concurrent requests efficiently.

## Further Reading

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [SGLang Paper](https://arxiv.org/abs/2312.07104)
