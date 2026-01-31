# Chapter 14: RLHF System Architecture

> *"The difference between a working RLHF system and an efficient one is whether you can fit four models on your GPUs."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Compare co-located vs disaggregated RLHF architectures
- Explain weight update mechanisms between training and inference engines
- Understand the hybrid engine approach (verl)
- Design an RLHF system for a given hardware setup

## Prerequisites

- Completed [Chapters 12-13](./chapter12.md) (RL Fundamentals, RLHF Flow)
- Understanding of distributed training (Part II)
- Familiarity with inference systems (Part III)

## Concept Overview

### The RLHF Systems Challenge

RLHF requires:
1. **Generation** (inference): Actor generates responses
2. **Scoring** (inference): Reward model evaluates
3. **Training** (training): PPO updates actor and critic

These have different optimal configurations:
- Generation: Large batch, high throughput
- Training: Gradient synchronization, memory for optimizer

Naively running both on the same GPUs wastes resources.

### Architecture Options

| Architecture | Description | Pros | Cons |
|--------------|-------------|------|------|
| **Co-located** | All models on same GPUs | Simple, no transfer | Memory constrained |
| **Disaggregated** | Separate GPU groups | Optimized per workload | Network transfer |
| **Hybrid** | Smart resource sharing | Best utilization | Complex implementation |

### Architecture 1: Co-located (slime, verl)

All models share the same GPUs, swapping memory between phases.

```
GPU 0-7 (same GPUs for everything):

Phase 1 - Generation:
┌────────────────────────────────────────────┐
│  Actor weights + KV cache for inference    │
│  (Reference and Reward also loaded)        │
└────────────────────────────────────────────┘

Phase 2 - Training:
┌────────────────────────────────────────────┐
│  Actor + Critic weights + gradients +      │
│  optimizer states + activations            │
└────────────────────────────────────────────┘
```

**Memory swapping**: After generation, KV cache is freed. Optimizer states loaded.

**Advantage**: No network transfer for weight updates.
**Disadvantage**: Cannot parallelize generation and training.

### Architecture 2: Disaggregated (OpenRLHF)

Separate GPU groups for different tasks.

```
Training Cluster (GPUs 0-31):          Inference Cluster (GPUs 32-63):
┌───────────────────────────┐         ┌───────────────────────────┐
│  Actor training           │         │  Actor inference          │
│  Critic training          │         │  (generation)             │
│  Gradients + optimizer    │ ◄────── │                           │
└───────────────────────────┘ weights └───────────────────────────┘
              │                                    ▲
              │              ┌───────────────────────────┐
              │              │  Reward Model            │
              └─────────────►│  (scoring)               │
                   prompts   └───────────────────────────┘
```

**Weight transfer**: After training, send updated weights to inference cluster.

**Advantage**: Generation and training can overlap.
**Disadvantage**: Network bandwidth for weight transfer.

### Architecture 3: Hybrid Engine (verl)

verl's innovation: Keep weights in GPU memory, switch between training and inference modes.

```
Same GPUs, Different Modes:

Training Mode:
┌────────────────────────────────────────────┐
│  FSDP sharded weights                      │
│  Full gradients and optimizer states       │
│  Backpropagation-ready tensors             │
└────────────────────────────────────────────┘
                    │
                    │ mode switch (no data movement!)
                    ▼
Inference Mode:
┌────────────────────────────────────────────┐
│  Same weights, viewed for inference        │
│  KV cache allocated                        │
│  No gradient tracking                      │
└────────────────────────────────────────────┘
```

**Key insight**: Tensor memory is reused between modes. Only metadata changes.

### Weight Update Mechanisms

How to get updated weights from training to inference?

**Method 1: Disk-based (simplest)**
```python
# After training
torch.save(actor.state_dict(), "checkpoint.pt")

# Inference engine loads
actor.load_state_dict(torch.load("checkpoint.pt"))
```
- Pros: Works always, supports different cluster sizes
- Cons: I/O bound, slow for large models

**Method 2: NCCL-based (disaggregated)**
```python
# Training rank 0 gathers full weights
full_weights = gather_weights(training_group)

# Send to inference rank 0
dist.send(full_weights, dst=inference_rank_0)

# Inference rank 0 broadcasts
dist.broadcast(full_weights, src=0, group=inference_group)
```
- Pros: Fast with good network
- Cons: Requires connectivity between clusters

**Method 3: Shared memory (co-located)**
```python
# verl approach: Share GPU memory via CUDA IPC
handle = tensor._cuda_ipc_handle()  # Get memory handle
serialized = serialize(handle)      # Not the data, just the pointer!

# Other process
tensor = deserialize(serialized)    # Reconstructs tensor from handle
# tensor points to SAME GPU memory - zero copy!
```
- Pros: Zero data movement
- Cons: Only works on same GPU

### The verl Weight Update Deep Dive

verl's weight update is elegant:

1. **Training finishes**: Actor weights are FSDP-sharded across GPUs
2. **Gather to full**: FSDP `FULL_STATE_DICT` gathers to rank 0
3. **Serialize handle**: Create CUDA IPC handle (just a pointer)
4. **Share handle**: Send handle to inference engine (tiny data!)
5. **Reconstruct tensor**: Inference engine creates tensor from handle
6. **Same memory**: Both engines now reference identical GPU memory

```
Training Engine                    Inference Engine
     │                                    │
     │  FSDP gathers                      │
     ▼                                    │
[Full tensor on GPU]                      │
     │                                    │
     │  Get IPC handle                    │
     ▼                                    │
[Handle: ptr=0x7f.., size=1GB]           │
     │                                    │
     │  Send handle (few bytes!)          │
     └───────────────────────────────────►│
                                          │  Reconstruct from handle
                                          ▼
                              [Same GPU memory, new tensor object]
```

### Memory Timeline in Hybrid Engine

```
Time →

Phase 1: Generation
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory: [Actor weights][KV Cache][Reward Model][Reference]  │
└─────────────────────────────────────────────────────────────────┘

Phase 2: Prepare for Training
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory: [Actor weights][Critic weights][Free space...]      │
│             (KV cache freed, RM and Ref offloaded)              │
└─────────────────────────────────────────────────────────────────┘

Phase 3: Training
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory: [Actor][Critic][Actor grads][Critic grads]          │
│             [Adam states][Activations]                          │
└─────────────────────────────────────────────────────────────────┘

Phase 4: Back to Generation
┌─────────────────────────────────────────────────────────────────┐
│ GPU Memory: [Updated Actor][KV Cache][RM][Ref]                  │
│             (optimizer states offloaded)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison: verl vs OpenRLHF vs slime

| Feature | verl | OpenRLHF | slime |
|---------|------|----------|-------|
| Architecture | Hybrid | Disaggregated | Co-located |
| Weight transfer | IPC handles | NCCL/Disk | Disk or tensor |
| Generation engine | Custom | vLLM | SGLang |
| Training engine | Custom SPMD | Ray + DeepSpeed | Megatron |
| Memory efficiency | High | Medium | High |
| Scaling | Complex | Simpler | Complex |

## Code Walkthrough

### Script 1: weight_update_demo.py

Demonstrates weight update mechanisms:
- Simulates different transfer methods
- Compares overhead

### Script 2: memory_timeline.py

Visualizes memory usage across RLHF phases:
- Shows peak memory per phase
- Identifies bottlenecks

## System Design Guidelines

### For Small Models (7B)

```
Single 8-GPU node:
- Co-located approach
- All 4 models fit with TP=1
- Simple implementation
```

### For Medium Models (70B)

```
Multi-node setup:
- Disaggregated or Hybrid
- Actor/Critic: TP=8, PP=2 (16 GPUs)
- Reward/Reference: TP=8 (8 GPUs each)
- Total: 32+ GPUs
```

### For Large Models (400B+)

```
Large cluster:
- Definitely disaggregated
- Separate clusters for training and inference
- Async weight updates
- Consider gradient checkpointing
```

## Try It Yourself

### Exercise 1: Memory Planning

For a 70B model RLHF setup:
1. Calculate memory per GPU for co-located (8 GPUs)
2. Calculate memory per GPU for disaggregated (32 GPUs)
3. Which fits? What trade-offs?

### Exercise 2: Weight Transfer Bandwidth

If weight transfer takes 10 seconds for 140GB:
1. What's the transfer bandwidth?
2. How does this compare to training iteration time?
3. Can we overlap transfer with anything?

### Exercise 3: Design an RLHF System

You have: 64 H100 GPUs across 8 nodes
Model: 70B parameters

Design:
1. Training parallelism (TP, PP, DP)
2. Inference parallelism
3. Weight update mechanism
4. Memory budget per GPU

## Key Takeaways

1. **Architecture choice depends on scale** - Co-located for small, disaggregated for large
2. **Weight transfer is critical** - IPC handles enable zero-copy on same GPU
3. **Memory phases are distinct** - Generation and training have different needs
4. **Hybrid engines maximize utilization** - Same GPUs, different modes
5. **Real systems combine techniques** - No one-size-fits-all

## The RLHF Systems Maturity Model

```
Level 1: Naive Co-location
  └─► All models loaded always
  └─► Works but memory inefficient

Level 2: Smart Co-location
  └─► Memory swapping between phases
  └─► Better utilization

Level 3: Disaggregated
  └─► Separate clusters
  └─► Network weight transfer

Level 4: Hybrid Engine
  └─► Shared memory, mode switching
  └─► Minimal overhead

Level 5: Async Hybrid
  └─► Overlapped generation and training
  └─► Maximum throughput
```

## What's Next?

Congratulations! You've completed the ML Systems Tutorial. You now understand:
- Distributed training primitives
- Parallelism strategies
- LLM inference systems
- RLHF architecture

For continued learning:
- Study verl, OpenRLHF, or trl source code
- Implement a simple RLHF system
- Contribute to open-source ML systems projects

## Further Reading

- [verl Paper](https://arxiv.org/abs/2409.19256)
- [OpenRLHF Repository](https://github.com/OpenRLHF/OpenRLHF)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
