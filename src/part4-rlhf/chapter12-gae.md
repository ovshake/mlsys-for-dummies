# gae_visualizer.py

> Visualize Generalized Advantage Estimation (GAE)

This script helps you understand how GAE works by visualizing the advantage computation for different λ values.

## What It Does

1. Creates a sample trajectory with rewards and values
2. Computes advantages with different λ values
3. Visualizes how λ affects the bias-variance tradeoff
4. Shows why λ=0.95 is common

## Run It

```bash
python tutorial/part4-rlhf/chapter12-rl-fundamentals/scripts/gae_visualizer.py
```

## Example Output

```
=== GAE Visualizer ===

Sample trajectory (10 steps):
  Rewards: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  Values:  [0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TD Errors (δ):
  Step 0: δ = 0 + 0.99*0.6 - 0.5 = 0.094
  Step 1: δ = 0 + 0.99*0.7 - 0.6 = 0.093
  ...

Advantages by λ:

λ = 0.0 (TD error only, high bias):
  A = [0.09, 0.09, 0.11, 0.42, -0.36, 0.10, 0.11, 0.12, 0.31, -0.89]
  Variance: 0.15

λ = 0.5 (balanced):
  A = [0.21, 0.19, 0.22, 0.38, -0.15, 0.16, 0.18, 0.18, 0.24, -0.89]
  Variance: 0.12

λ = 0.95 (common choice):
  A = [0.45, 0.38, 0.35, 0.32, -0.02, 0.23, 0.22, 0.20, 0.18, -0.89]
  Variance: 0.14

λ = 1.0 (full returns, low bias):
  A = [0.52, 0.44, 0.40, 0.35, 0.01, 0.26, 0.24, 0.21, 0.18, -0.89]
  Variance: 0.16
```

## The GAE Formula

```
A^GAE_t = δt + (γλ)δt+1 + (γλ)²δt+2 + ...
        = Σ (γλ)^k δt+k

Where δt = rt + γV(st+1) - V(st)
```

## Why λ = 0.95?

- **λ = 0**: Only considers immediate TD error (high bias, low variance)
- **λ = 1**: Full Monte Carlo returns (low bias, high variance)
- **λ = 0.95**: Good balance - mostly looks ahead, slight smoothing

## Source Code

```python
{{#include ../../tutorial/part4-rlhf/chapter12-rl-fundamentals/scripts/gae_visualizer.py}}
```
