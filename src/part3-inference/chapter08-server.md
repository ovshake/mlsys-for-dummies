# minimal_inference_server.py

> A simplified LLM inference server demonstrating core architecture

This script implements a minimal inference server showing the key components: request handling, batching, and token generation.

## What It Does

1. Creates a simple request queue
2. Implements basic batching logic
3. Simulates the prefill/decode loop
4. Demonstrates streaming output

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Minimal Inference Server               │
│                                                  │
│  Request Queue ──► Batcher ──► Model ──► Output │
│                                                  │
│  Components:                                     │
│  - RequestQueue: FIFO queue for incoming prompts│
│  - SimpleBatcher: Groups requests for GPU        │
│  - MockModel: Simulates forward pass            │
│  - Generator: Token-by-token output loop        │
└─────────────────────────────────────────────────┘
```

## Run It

```bash
python tutorial/part3-inference/chapter08-server-anatomy/scripts/minimal_inference_server.py
```

## Key Learning Points

**Request Lifecycle:**
```python
# 1. Request arrives
request = Request(prompt="Hello, world!")

# 2. Tokenize
tokens = tokenizer.encode(request.prompt)

# 3. Add to queue
queue.add(request)

# 4. Batch processing
batch = batcher.get_next_batch()

# 5. Prefill (process prompt)
kv_cache = model.prefill(batch)

# 6. Decode (generate tokens)
while not done:
    next_token = model.decode(kv_cache)
    yield next_token
```

## What This Demonstrates

- **Separation of concerns**: Each component has a single responsibility
- **Queue management**: Requests are processed fairly
- **Batching strategy**: Multiple requests share GPU
- **Two-phase inference**: Prefill then decode

## What's Missing (Real Systems)

- KV cache management (Chapter 9)
- CUDA graph optimization (Chapter 10)
- Speculative decoding (Chapter 11)
- Tensor parallelism for large models
- Production error handling

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter08-server-anatomy/scripts/minimal_inference_server.py}}
```
