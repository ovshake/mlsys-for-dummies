# rlhf_loop_pseudo.py

> The complete RLHF training loop in pseudocode

This script shows the exact computation flow of one RLHF training step, making it easy to understand what happens and when.

## What It Does

1. Simulates all four models (Actor, Critic, Reward, Reference)
2. Walks through each step of the training loop
3. Shows tensor shapes and intermediate values
4. Demonstrates the complete PPO update

## Run It

```bash
python tutorial/part4-rlhf/chapter13-rlhf-flow/scripts/rlhf_loop_pseudo.py
```

## Example Output

```
=== RLHF Training Loop Demo ===

Step 1: Sample prompts
  Batch size: 4
  Prompt shapes: (4, 64) tokens

Step 2: Generate responses (Actor)
  Actor forward pass...
  Generated tokens: (4, 128)
  Actor logits: (4, 128, 50257)
  Old log probs: (4, 128)

Step 3: Score responses (Reward Model)
  Reward model forward pass...
  Scores: [0.73, 0.45, 0.91, 0.62]

Step 4: Compute KL penalty (Reference)
  Reference forward pass...
  Reference log probs: (4, 128)
  KL divergence per token: (4, 128)
  Mean KL: 0.23

Step 5: Compute total rewards
  reward = reward_model_score - β * KL
  Total rewards: [0.50, 0.28, 0.75, 0.41]

Step 6: Compute advantages (Critic + GAE)
  Critic forward pass...
  Values: (4, 128)
  GAE advantages: (4, 128)

Step 7: PPO update
  Ratio = exp(new_log_prob - old_log_prob)
  Clipped ratio: clip(ratio, 0.8, 1.2)
  Actor loss: -0.042
  Critic loss: 0.156

  Update complete!
```

## The Core Loop

```python
for batch in dataloader:
    # 1. Generate
    responses, old_logprobs = actor.generate(batch.prompts)

    # 2. Score
    rewards = reward_model(batch.prompts, responses)

    # 3. KL penalty
    ref_logprobs = reference(batch.prompts, responses)
    kl = old_logprobs - ref_logprobs
    rewards = rewards - beta * kl

    # 4. Advantages
    values = critic(batch.prompts, responses)
    advantages = gae(rewards, values)

    # 5. PPO update
    new_logprobs = actor(batch.prompts, responses)
    ratio = (new_logprobs - old_logprobs).exp()
    actor_loss = -torch.min(ratio * advantages,
                           ratio.clamp(0.8, 1.2) * advantages)
    critic_loss = (values - rewards) ** 2

    # 6. Backprop
    (actor_loss + critic_loss).backward()
    optimizer.step()
```

## Source Code

```python
{{#include ../../tutorial/part4-rlhf/chapter13-rlhf-flow/scripts/rlhf_loop_pseudo.py}}
```
