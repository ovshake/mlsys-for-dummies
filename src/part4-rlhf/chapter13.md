# Chapter 13: RLHF Computation Flow

> *"Four models, one update. Orchestrating RLHF is like conducting a symphony of neural networks."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Name the four models in RLHF and their roles
- Trace the data flow through one RLHF training step
- Explain why we need a reference model
- Calculate memory requirements for RLHF training

## Prerequisites

- Completed [Chapter 12 (RL Fundamentals)](./chapter12.md)
- Understanding of PPO and advantage estimation
- Familiarity with model architecture (transformers)

## Concept Overview

### The Four Models of RLHF

| Model | Role | Updates? | Size |
|-------|------|----------|------|
| **Actor (Policy)** | Generates responses | Yes | Full LLM |
| **Critic (Value)** | Predicts expected reward | Yes | Full LLM or smaller |
| **Reward** | Scores responses | No | Trained separately |
| **Reference** | Prevents reward hacking | No | Copy of initial actor |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RLHF MODEL ORCHESTRA                               │
│                                                                         │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      │
│   │  Actor  │      │ Critic  │      │ Reward  │      │Reference│      │
│   │(Policy) │      │(Value)  │      │ Model   │      │ Policy  │      │
│   └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘      │
│        │                │                │                │            │
│        │                │                │                │            │
│   Generates         Estimates        Evaluates        Anchors         │
│   responses         future reward    quality         updates         │
│        │                │                │                │            │
│        └────────────────┴────────────────┴────────────────┘            │
│                              │                                          │
│                         PPO Update                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The RLHF Training Loop

One step of RLHF training:

```
1. SAMPLE PROMPTS
   └─► Get batch of prompts from dataset

2. GENERATE RESPONSES (Actor)
   └─► Actor generates responses for each prompt
   └─► Save token probabilities

3. SCORE RESPONSES (Reward Model)
   └─► Reward model scores each response
   └─► This is the "human feedback" signal

4. COMPUTE KL PENALTY (Reference)
   └─► Compare actor probabilities to reference
   └─► Penalize divergence (prevent reward hacking)

5. COMPUTE ADVANTAGES (Critic + GAE)
   └─► Critic estimates values
   └─► GAE computes advantages

6. PPO UPDATE (Actor + Critic)
   └─► Update actor using PPO objective
   └─► Update critic to predict rewards better
```

### Detailed Data Flow

```
                         Prompt
                           │
                           ▼
            ┌──────────────────────────┐
            │         ACTOR            │
            │  Generate response       │
            │  Output: tokens, logits  │
            └───────────┬──────────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ REWARD   │  │REFERENCE │  │  CRITIC  │
    │ MODEL    │  │          │  │          │
    │Score: 0.8│  │ logits   │  │ values   │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         └─────────────┴─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ COMPUTE REWARD  │
              │ R = R_rm - β*KL │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  COMPUTE GAE    │
              │  advantages     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   PPO UPDATE    │
              │  actor, critic  │
              └─────────────────┘
```

### The Reward Calculation

The reward for each response combines:

```
R_total = R_reward_model - β * KL(π_actor || π_reference)
```

**R_reward_model**: Score from reward model (trained on human preferences)

**KL penalty**: Prevents "reward hacking"

Without KL penalty, the model might find degenerate solutions:
- Repeating phrases that game the reward model
- Producing unnatural but high-scoring outputs
- Catastrophic forgetting of language capabilities

### Why Reference Model?

The reference model is a frozen copy of the initial policy. It serves as an anchor:

```
Without reference:
  Actor → "AMAZING! INCREDIBLE! BEST EVER!" (reward hacks)

With reference:
  Actor → Natural response similar to reference
  If too different → KL penalty reduces total reward
```

KL divergence measures how different the actor's distribution is from the reference:
```
KL(π_actor || π_ref) = Σ π_actor(token) * log(π_actor(token) / π_ref(token))
```

### Per-Token vs Per-Response Rewards

In practice, rewards can be assigned:

**Per-response** (most common):
- Reward model scores complete response
- Reward assigned to last token
- Other tokens get 0 reward
- GAE propagates signal backwards

**Per-token** (process reward):
- Each token gets a score
- More fine-grained signal
- Harder to obtain labels

### Memory Requirements

For a 7B parameter model with RLHF:

| Component | Memory (FP16) |
|-----------|--------------|
| Actor | 14 GB |
| Critic | 14 GB |
| Reward Model | 14 GB |
| Reference | 14 GB |
| Optimizer states | 56 GB |
| Activations | ~20 GB |
| **Total** | **~130 GB** |

For 70B: multiply by 10 → **~1.3 TB**!

This is why RLHF needs careful system design.

## Code Walkthrough

### Script 1: rlhf_loop_pseudo.py

Pseudocode implementation of the RLHF loop:
- Shows exact data flow
- Demonstrates each computation
- Explains intermediate values

### Script 2: reward_calculator.py

Implements reward calculation:
- Reward model scoring
- KL divergence computation
- Total reward with penalty

## Common Questions

### Q: Why not just fine-tune on high-reward responses?

Supervised fine-tuning on selected responses (rejection sampling) works, but:
- Wastes low-reward samples
- No gradient signal about "how bad" something is
- PPO makes more efficient use of data

### Q: Can the critic share weights with the actor?

Yes! Common approaches:
- **Separate critic**: Full model, independent
- **Shared backbone**: Same transformer, different heads
- **Value head**: Small MLP on top of actor's hidden states

Shared approaches save memory but may have optimization conflicts.

### Q: How is the reward model trained?

Before RLHF:
1. Collect comparison data: "Response A is better than B"
2. Train reward model with ranking loss
3. Reward model learns human preferences

The reward model is then frozen during RLHF.

## Try It Yourself

### Exercise 1: Trace Data Flow

For a batch of 4 prompts with max response length 100:
1. What are the tensor shapes at each stage?
2. How many forward passes per training step?
3. What's the communication pattern?

### Exercise 2: KL Penalty Tuning

The KL coefficient β controls the penalty:
- β too low: reward hacking
- β too high: no learning

Experiment (conceptually):
1. What happens if β = 0?
2. What happens if β = 10?
3. How would you find the right β?

### Exercise 3: Memory Optimization

You have 8× 80GB GPUs and want to train a 70B model with RLHF.
1. What parallelism strategies would you use?
2. Can you fit all 4 models?
3. What trade-offs would you make?

## Key Takeaways

1. **Four models, one loop** - Actor, Critic, Reward, Reference
2. **KL penalty is crucial** - Prevents reward hacking
3. **GAE for credit assignment** - Propagates reward signal
4. **Memory is the bottleneck** - 4× model weights minimum
5. **Reference stays frozen** - Anchors the learning

## The RLHF Equation

The complete PPO-RLHF objective:

```
L = E[
    L^PPO(actor_params)           # Policy improvement
  - c₁ * L^VF(critic_params)      # Value function loss
  + c₂ * Entropy(actor)           # Exploration bonus
]

Where:
  L^PPO = min(ratio * A, clip(ratio) * A)
  L^VF = (V_predicted - R_observed)²
  A = GAE(rewards, values)
  rewards = R_reward_model - β * KL
```

## What's Next?

In [Chapter 14](./chapter14.md), we'll explore **RLHF System Architecture**—how to efficiently orchestrate these models across GPUs with co-location, disaggregation, and hybrid approaches.

## Further Reading

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [Training language models to follow instructions](https://openai.com/research/instruction-following)
