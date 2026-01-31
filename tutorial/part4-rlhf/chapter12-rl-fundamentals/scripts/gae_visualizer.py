#!/usr/bin/env python3
"""
GAE (Generalized Advantage Estimation) Visualizer

This script demonstrates how GAE works and its effect on advantage estimation.

Usage:
    python gae_visualizer.py
"""

import argparse
from typing import List, Tuple


def generate_trajectory(length: int = 10) -> Tuple[List[float], List[float]]:
    """
    Generate a sample trajectory with rewards and values.

    Returns:
        rewards: List of rewards at each step
        values: List of value estimates at each step
    """
    # Simulated trajectory: mostly small rewards, occasional large
    rewards = [
        0.1, 0.1, 0.2, 0.1, 0.5,  # Early exploration
        0.1, 0.3, 0.1, 0.1, 1.0,  # Some success at end
    ][:length]

    # Value estimates (what the critic predicts)
    values = [
        0.8, 0.7, 0.7, 0.6, 0.5,  # Decreasing as end approaches
        0.4, 0.4, 0.3, 0.2, 0.1,
    ][:length]

    return rewards, values


def compute_td_errors(rewards: List[float], values: List[float],
                       gamma: float = 0.99) -> List[float]:
    """
    Compute TD (Temporal Difference) errors.

    TD error = r_t + γV(s_{t+1}) - V(s_t)

    This is the "surprise" - how much better/worse than expected.
    """
    td_errors = []
    n = len(rewards)

    for t in range(n):
        r = rewards[t]
        v_t = values[t]
        v_next = values[t + 1] if t + 1 < n else 0  # Terminal state has 0 value

        delta = r + gamma * v_next - v_t
        td_errors.append(delta)

    return td_errors


def compute_gae(td_errors: List[float], gamma: float = 0.99,
                lam: float = 0.95) -> List[float]:
    """
    Compute GAE advantages.

    A^GAE_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}

    λ controls bias-variance tradeoff:
    - λ=0: Just TD error (high bias, low variance)
    - λ=1: Full returns minus baseline (low bias, high variance)
    """
    advantages = []
    gae = 0
    n = len(td_errors)

    # Compute backwards
    for t in reversed(range(n)):
        gae = td_errors[t] + gamma * lam * gae
        advantages.insert(0, gae)

    return advantages


def compute_monte_carlo_returns(rewards: List[float], values: List[float],
                                 gamma: float = 0.99) -> List[float]:
    """
    Compute Monte Carlo returns (full returns minus baseline).

    This is GAE with λ=1.
    """
    n = len(rewards)
    returns = [0.0] * n
    G = 0

    for t in reversed(range(n)):
        G = rewards[t] + gamma * G
        returns[t] = G - values[t]  # Advantage = return - baseline

    return returns


def visualize_advantages(rewards: List[float], values: List[float],
                          td_errors: List[float],
                          advantages_by_lambda: dict):
    """Visualize how different λ values affect advantage estimation."""
    n = len(rewards)

    print("\n" + "=" * 80)
    print(" TRAJECTORY DATA")
    print("=" * 80)

    print(f"\n{'Step':<6} {'Reward':<10} {'Value':<10} {'TD Error':<12}")
    print("-" * 40)
    for t in range(n):
        print(f"{t:<6} {rewards[t]:<10.2f} {values[t]:<10.2f} {td_errors[t]:<12.4f}")

    print("\n" + "=" * 80)
    print(" GAE WITH DIFFERENT λ VALUES")
    print("=" * 80)

    header = f"{'Step':<6}"
    for lam in sorted(advantages_by_lambda.keys()):
        header += f"{'λ=' + str(lam):<12}"
    print(f"\n{header}")
    print("-" * (6 + 12 * len(advantages_by_lambda)))

    for t in range(n):
        row = f"{t:<6}"
        for lam in sorted(advantages_by_lambda.keys()):
            row += f"{advantages_by_lambda[lam][t]:<12.4f}"
        print(row)


def analyze_bias_variance():
    """Analyze the bias-variance tradeoff in GAE."""
    print("\n" + "=" * 80)
    print(" BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("=" * 80)
    print("""
GAE with different λ values:

┌────────────────────────────────────────────────────────────────────┐
│                     λ = 0 (TD Error Only)                          │
│                                                                    │
│  A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)                           │
│                                                                    │
│  Properties:                                                        │
│    - HIGH BIAS: Only looks one step ahead                          │
│    - LOW VARIANCE: Single reward, single value estimate            │
│    - Fast to adapt, but might miss long-term patterns             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     λ = 1 (Monte Carlo)                            │
│                                                                    │
│  A_t = G_t - V(s_t) = Σ γ^k r_{t+k} - V(s_t)                      │
│                                                                    │
│  Properties:                                                        │
│    - LOW BIAS: Uses all future rewards                             │
│    - HIGH VARIANCE: Accumulates noise from many rewards            │
│    - Accurate but slow to learn                                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     λ = 0.95 (Typical)                             │
│                                                                    │
│  A_t = Σ (γλ)^k δ_{t+k}                                           │
│                                                                    │
│  Properties:                                                        │
│    - BALANCED: Weights earlier steps more                          │
│    - PRACTICAL: Good empirical performance                         │
│    - Exponential decay of TD errors                                │
└────────────────────────────────────────────────────────────────────┘

The weighting scheme (λ = 0.95, γ = 0.99):

  Step t:   weight = 1.00
  Step t+1: weight = 0.94 (γλ = 0.99 × 0.95)
  Step t+2: weight = 0.88 (γλ)²
  Step t+3: weight = 0.83 (γλ)³
  ...
  Step t+10: weight = 0.53

Far future TD errors contribute less, reducing variance while
maintaining enough signal for learning.
""")


def demonstrate_numerical_example():
    """Show a concrete numerical example of GAE computation."""
    print("\n" + "=" * 80)
    print(" NUMERICAL EXAMPLE: GAE COMPUTATION")
    print("=" * 80)

    # Simple 3-step trajectory
    rewards = [0.1, 0.2, 1.0]  # Big reward at end
    values = [0.5, 0.4, 0.2]   # Decreasing values
    gamma = 0.99
    lam = 0.95

    print(f"""
Trajectory:
  Step 0: r=0.1, V=0.5
  Step 1: r=0.2, V=0.4
  Step 2: r=1.0, V=0.2 (terminal)

TD Errors (δ_t = r_t + γV_{t+1} - V_t):
  δ_0 = 0.1 + 0.99×0.4 - 0.5 = {0.1 + 0.99*0.4 - 0.5:.4f}
  δ_1 = 0.2 + 0.99×0.2 - 0.4 = {0.2 + 0.99*0.2 - 0.4:.4f}
  δ_2 = 1.0 + 0.99×0.0 - 0.2 = {1.0 + 0.99*0.0 - 0.2:.4f}

GAE Computation (working backwards, λ={lam}):
  A_2 = δ_2 = {1.0 + 0.99*0.0 - 0.2:.4f}
  A_1 = δ_1 + γλ×A_2 = {0.2 + 0.99*0.2 - 0.4:.4f} + {gamma*lam}×{1.0 + 0.99*0.0 - 0.2:.4f}
      = {(0.2 + 0.99*0.2 - 0.4) + gamma*lam*(1.0 + 0.99*0.0 - 0.2):.4f}
  A_0 = δ_0 + γλ×A_1
      = {(0.1 + 0.99*0.4 - 0.5) + gamma*lam*((0.2 + 0.99*0.2 - 0.4) + gamma*lam*(1.0 + 0.99*0.0 - 0.2)):.4f}

Notice: Step 0's advantage includes discounted information about the
big reward at step 2, but that information is attenuated by (γλ)².
""")


def main():
    parser = argparse.ArgumentParser(description="GAE Visualizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--trajectory-length", type=int, default=10,
                        help="Length of trajectory")
    args = parser.parse_args()

    print("╔" + "═" * 78 + "╗")
    print("║" + " GENERALIZED ADVANTAGE ESTIMATION (GAE) VISUALIZER".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Generate trajectory
    rewards, values = generate_trajectory(args.trajectory_length)

    # Compute TD errors
    td_errors = compute_td_errors(rewards, values, args.gamma)

    # Compute GAE for different lambda values
    lambda_values = [0.0, 0.5, 0.9, 0.95, 1.0]
    advantages_by_lambda = {}

    for lam in lambda_values:
        advantages_by_lambda[lam] = compute_gae(td_errors, args.gamma, lam)

    # Visualize
    visualize_advantages(rewards, values, td_errors, advantages_by_lambda)

    # Analysis
    analyze_bias_variance()

    # Numerical example
    demonstrate_numerical_example()

    # Key insights
    print("\n" + "=" * 80)
    print(" KEY INSIGHTS FOR RLHF")
    print("=" * 80)
    print("""
In RLHF training:

1. TRAJECTORY = One response generation
   - States: prompt + partial response
   - Actions: generated tokens
   - Reward: typically only at the end (from reward model)

2. GAE HELPS WITH CREDIT ASSIGNMENT
   - Which tokens contributed to the final reward?
   - GAE propagates reward signal backwards through the response
   - λ controls how far back the signal reaches

3. TYPICAL RLHF SETTINGS
   - γ = 0.99 or 1.0 (we care about all tokens)
   - λ = 0.95 (good balance)
   - Sparse reward (only at end of generation)

4. VALUE FUNCTION IN RLHF
   - Critic network predicts expected reward
   - Helps reduce variance in policy gradient
   - Often shares layers with the policy (actor-critic)

5. PPO USES GAE ADVANTAGES
   - Compute GAE for each token in response
   - Update policy using PPO-Clip objective
   - Bounded updates prevent catastrophic forgetting
""")


if __name__ == "__main__":
    main()
