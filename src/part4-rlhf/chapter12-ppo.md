# ppo_cartpole.py

> Learn PPO mechanics on a simple game before applying to LLMs

This script implements PPO (Proximal Policy Optimization) on the classic CartPole environment. It's simpler than LLM training but demonstrates all the same concepts.

## What It Does

1. Creates Actor (policy) and Critic (value) networks
2. Collects episodes using the current policy
3. Computes advantages using GAE
4. Updates policy with PPO-Clip objective
5. Tracks learning progress

## Run It

```bash
pip install gymnasium  # Install gym environment
python tutorial/part4-rlhf/chapter12-rl-fundamentals/scripts/ppo_cartpole.py
```

## Expected Output

```
=== PPO on CartPole ===

Episode 10: Average Reward = 21.5
Episode 20: Average Reward = 45.3
Episode 30: Average Reward = 98.7
Episode 40: Average Reward = 187.2
Episode 50: Average Reward = 312.5
Episode 60: Average Reward = 500.0 (solved!)

Training complete! CartPole balanced for 500 steps.
```

## Key Components

**Actor Network:**
```python
class Actor(nn.Module):
    def forward(self, state):
        # Returns action probabilities
        return F.softmax(self.net(state), dim=-1)
```

**Critic Network:**
```python
class Critic(nn.Module):
    def forward(self, state):
        # Returns state value
        return self.net(state)
```

**PPO-Clip Loss:**
```python
ratio = new_prob / old_prob
clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
loss = -torch.min(ratio * advantage, clipped * advantage).mean()
```

## Why CartPole?

CartPole is the "Hello World" of RL:
- Simple (2D state, 2 actions)
- Fast feedback (episodes complete quickly)
- Clear success metric (balance for 500 steps)

The same PPO algorithm scales to LLMs with minimal changes!

## Source Code

```python
{{#include ../../tutorial/part4-rlhf/chapter12-rl-fundamentals/scripts/ppo_cartpole.py}}
```
