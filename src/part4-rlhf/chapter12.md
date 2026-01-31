# Chapter 12: RL Fundamentals for LLMs

> *"Before you can teach a model with human feedback, you need to speak the language of reinforcement learning."*

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the core RL concepts: states, actions, rewards, policies
- Understand value functions and the Bellman equation
- Implement policy gradients and the REINFORCE algorithm
- Explain PPO (Proximal Policy Optimization) and why it's used for LLMs

## Prerequisites

- Completed Part III (LLM Inference)
- Basic calculus (derivatives)
- Familiarity with neural network training

## Concept Overview

### RL in 60 Seconds

**Supervised Learning**: Given input X, predict label Y (teacher provides answer)

**Reinforcement Learning**: Given state S, take action A, observe reward R (learn from trial and error)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL FRAMEWORK                                  │
│                                                                     │
│    ┌─────────┐         action a         ┌─────────────┐           │
│    │  Agent  │ ─────────────────────────► Environment  │           │
│    │ (Policy)│ ◄───────────────────────── (World)     │           │
│    └─────────┘    state s, reward r     └─────────────┘           │
│                                                                     │
│    Goal: Learn policy π(a|s) that maximizes cumulative reward      │
└─────────────────────────────────────────────────────────────────────┘
```

### The LLM as an RL Agent

| RL Concept | LLM Interpretation |
|------------|-------------------|
| **State** | Prompt + generated tokens so far |
| **Action** | Next token to generate |
| **Policy** | The LLM itself (token probabilities) |
| **Reward** | Human preference score (or reward model) |
| **Episode** | One complete generation |

### Value Functions: Predicting Future Rewards

**State Value V(s)**: Expected total reward starting from state s
```
V(s) = E[R₀ + γR₁ + γ²R₂ + ... | S₀ = s]
```

**Action Value Q(s,a)**: Expected total reward after taking action a in state s
```
Q(s,a) = E[R₀ + γR₁ + γ²R₂ + ... | S₀ = s, A₀ = a]
```

**γ (gamma)**: Discount factor (0-1). Lower γ = short-sighted, higher γ = long-term thinking.

For LLMs, we typically use γ ≈ 1 (care equally about all future rewards).

### The Bellman Equation

The fundamental equation of RL:
```
V(s) = E[R + γV(s') | S = s]
     = Σₐ π(a|s) [R(s,a) + γ Σₛ' P(s'|s,a) V(s')]
```

"The value of a state is the immediate reward plus the discounted value of the next state."

This recursive structure enables dynamic programming solutions.

### Policy Gradients: Learning by Gradient Ascent

Instead of computing values, directly optimize the policy!

**Objective**: Maximize expected reward
```
J(θ) = E[Σₜ R(sₜ, aₜ)]
```

**Policy Gradient Theorem**:
```
∇J(θ) = E[Σₜ ∇log π_θ(aₜ|sₜ) · Gₜ]
```

Where Gₜ = total future reward from time t.

**Intuition**:
- If action led to high reward: increase its probability (positive gradient)
- If action led to low reward: decrease its probability (negative gradient)

### REINFORCE Algorithm

The simplest policy gradient algorithm:

```python
for episode in episodes:
    # Collect trajectory
    states, actions, rewards = collect_episode(policy)

    # Compute returns
    returns = compute_returns(rewards, gamma)

    # Update policy
    for t, (s, a, G) in enumerate(zip(states, actions, returns)):
        loss = -log_prob(policy(s), a) * G
        loss.backward()

    optimizer.step()
```

**Problem**: High variance! Returns can vary wildly between episodes.

### Variance Reduction: Baselines

Subtract a baseline from returns to reduce variance:
```
∇J(θ) = E[Σₜ ∇log π_θ(aₜ|sₜ) · (Gₜ - b(sₜ))]
```

Common baseline: **Value function V(s)** — learn to predict expected return.

This gives us the **Advantage**:
```
A(s,a) = Q(s,a) - V(s)
       ≈ R + γV(s') - V(s)  (TD error)
```

"How much better is this action compared to the average?"

### Actor-Critic: Best of Both Worlds

**Actor**: Policy network π_θ(a|s)
**Critic**: Value network V_φ(s)

```python
# Actor update (policy gradient with advantage)
advantage = reward + gamma * V(next_state) - V(state)
actor_loss = -log_prob(action) * advantage.detach()

# Critic update (value regression)
critic_loss = (V(state) - (reward + gamma * V(next_state).detach()))²
```

### Generalized Advantage Estimation (GAE)

GAE smoothly interpolates between:
- Low bias, high variance (full returns)
- High bias, low variance (TD error)

```
A^GAE_t = Σₖ (γλ)^k δₜ₊ₖ

Where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)  (TD error)
```

λ controls the tradeoff:
- λ = 0: Just TD error (high bias, low variance)
- λ = 1: Full returns (low bias, high variance)

Typical: λ = 0.95

### PPO: The Industry Standard

PPO (Proximal Policy Optimization) adds **trust region** constraints:

"Don't change the policy too much in one update."

**PPO-Clip objective**:
```
L^CLIP(θ) = E[min(rₜ(θ)Aₜ, clip(rₜ(θ), 1-ε, 1+ε)Aₜ)]

Where rₜ(θ) = π_θ(aₜ|sₜ) / π_θold(aₜ|sₜ)  (probability ratio)
```

**Intuition**:
- If advantage is positive and ratio is high: clip to prevent too much increase
- If advantage is negative and ratio is low: clip to prevent too much decrease
- Keeps policy changes bounded

### Why PPO for LLMs?

1. **Stable training**: Trust region prevents catastrophic forgetting
2. **Sample efficient**: Reuses samples within trust region
3. **Proven at scale**: Used by OpenAI, Anthropic, DeepMind
4. **Simple to implement**: No second-order optimization

## Code Walkthrough

### Script 1: ppo_cartpole.py

A minimal PPO implementation on CartPole:
- Actor-Critic networks
- GAE advantage computation
- PPO-Clip objective

This isn't for LLMs but shows PPO mechanics clearly.

### Script 2: gae_visualizer.py

Visualizes how GAE works:
- Shows TD errors over trajectory
- Compares different λ values
- Demonstrates bias-variance tradeoff

## The RLHF Connection

In RLHF:
- **State**: Prompt + partial response
- **Action**: Next token
- **Reward**: Comes from reward model (trained on human preferences)
- **Episode**: Complete response generation

The PPO objective becomes:
```
max E[R_reward_model(response) - β * KL(π || π_ref)]

Where:
- R_reward_model: Score from reward model
- KL term: Penalty for diverging from reference policy
- β: KL coefficient (prevents reward hacking)
```

## Try It Yourself

### Exercise 1: Implement REINFORCE

Implement REINFORCE for a simple environment:
1. Collect episodes
2. Compute returns
3. Update policy
4. Track learning curves

### Exercise 2: Add a Baseline

Modify your REINFORCE to use a learned baseline:
1. Add a value network
2. Compute advantages
3. Compare variance with/without baseline

### Exercise 3: Understand PPO Clipping

For different advantage signs and probability ratios:
1. Compute clipped and unclipped objectives
2. Determine which is used
3. Explain why clipping helps stability

## Key Takeaways

1. **RL learns from rewards, not labels** - Trial and error, not supervision
2. **Value functions predict future rewards** - Enables credit assignment
3. **Policy gradients directly optimize the policy** - No need to estimate values
4. **Baselines reduce variance** - Critical for practical training
5. **PPO is stable and scalable** - The go-to algorithm for RLHF

## The RL Hierarchy

```
Simple ────────────────────────────────────► Complex

REINFORCE → Actor-Critic → A2C → PPO → RLHF with PPO
  ↓              ↓           ↓      ↓           ↓
High        Value as    Parallel  Trust    Multi-model
variance    baseline    training  region   orchestration
```

## What's Next?

In [Chapter 13](./chapter13.md), we'll dive into **RLHF Computation Flow**—how the Actor, Critic, Reward, and Reference models work together during training.

## Further Reading

- [Policy Gradient Methods (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
