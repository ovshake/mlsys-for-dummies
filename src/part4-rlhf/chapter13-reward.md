# reward_calculator.py

> Understand reward calculation with KL penalty

This script demonstrates how the total reward in RLHF is computed from the reward model score and KL penalty.

## What It Does

1. Shows raw reward model scores
2. Computes KL divergence between actor and reference
3. Applies the KL penalty with different β values
4. Demonstrates why the penalty prevents reward hacking

## Run It

```bash
python tutorial/part4-rlhf/chapter13-rlhf-flow/scripts/reward_calculator.py
```

## Example Output

```
=== RLHF Reward Calculator ===

Response: "This is a great product! I highly recommend it!"

Reward Model Score: 0.85 (high quality response)

KL Divergence Calculation:
  Actor log prob for each token:
    "This": -2.3,  "is": -1.1,  "a": -0.8,  ...
  Reference log prob for each token:
    "This": -2.1,  "is": -1.0,  "a": -0.9,  ...

  KL per token = actor_logp - ref_logp
    "This": -0.2,  "is": -0.1,  "a": +0.1,  ...

  Total KL: 0.45 (actor has diverged from reference)

Total Reward with Different β:
  β = 0.0: R = 0.85 - 0.0 * 0.45 = 0.85
  β = 0.1: R = 0.85 - 0.1 * 0.45 = 0.805
  β = 0.5: R = 0.85 - 0.5 * 0.45 = 0.625
  β = 1.0: R = 0.85 - 1.0 * 0.45 = 0.40

Observation: Higher β penalizes divergence more heavily.
```

## Why KL Penalty Matters

```
Without penalty (β=0):
  Actor learns to say "AMAZING! INCREDIBLE!" for everything
  Reward model gives high scores
  But output is unnatural

With penalty (β=0.1):
  Actor stays close to reference
  Must improve while remaining natural
  Better quality outputs
```

## The Formula

```python
def compute_reward(response, actor, reference, reward_model, beta):
    # Get reward model score
    rm_score = reward_model(response)

    # Compute KL divergence
    actor_logp = actor.log_prob(response)
    ref_logp = reference.log_prob(response)
    kl = (actor_logp - ref_logp).sum()

    # Total reward with penalty
    total_reward = rm_score - beta * kl

    return total_reward
```

## Source Code

```python
{{#include ../../tutorial/part4-rlhf/chapter13-rlhf-flow/scripts/reward_calculator.py}}
```
