#!/usr/bin/env python3
"""
RLHF Reward Calculator

This script demonstrates how rewards are computed in RLHF:
- Reward model scoring
- KL divergence penalty
- Combined reward signal

Usage:
    python reward_calculator.py
"""

import argparse
import math
from typing import List, Tuple


def compute_kl_divergence(actor_log_probs: List[float],
                          ref_log_probs: List[float]) -> List[float]:
    """
    Compute per-token KL divergence.

    KL(actor || ref) = Σ p_actor * log(p_actor / p_ref)
                     = Σ p_actor * (log p_actor - log p_ref)

    Since we have log probs, this simplifies to computing the difference
    and then exponentiating to get actual KL.
    """
    kl_per_token = []
    for actor_lp, ref_lp in zip(actor_log_probs, ref_log_probs):
        # Approximate KL using log prob difference
        # Full KL would be: exp(actor_lp) * (actor_lp - ref_lp)
        # Common approximation: just the difference (works well in practice)
        kl = actor_lp - ref_lp
        kl_per_token.append(kl)
    return kl_per_token


def compute_rewards(
    reward_model_score: float,
    actor_log_probs: List[float],
    ref_log_probs: List[float],
    kl_coef: float = 0.02,
    reward_at_end_only: bool = True,
) -> Tuple[List[float], dict]:
    """
    Compute per-token rewards with KL penalty.

    Args:
        reward_model_score: Score from reward model (typically for full response)
        actor_log_probs: Log probabilities from actor for each token
        ref_log_probs: Log probabilities from reference for each token
        kl_coef: Coefficient for KL penalty (β in papers)
        reward_at_end_only: If True, RM score only at last token

    Returns:
        List of rewards for each token
        Dictionary with stats
    """
    num_tokens = len(actor_log_probs)

    # Compute KL divergence
    kl_per_token = compute_kl_divergence(actor_log_probs, ref_log_probs)

    # Compute rewards
    rewards = []
    for t in range(num_tokens):
        kl_penalty = kl_coef * kl_per_token[t]

        if reward_at_end_only:
            # RM score only at last token
            if t == num_tokens - 1:
                r = reward_model_score - kl_penalty
            else:
                r = -kl_penalty  # Only penalty
        else:
            # RM score distributed across tokens
            r = reward_model_score / num_tokens - kl_penalty

        rewards.append(r)

    stats = {
        'total_kl': sum(kl_per_token),
        'avg_kl': sum(kl_per_token) / len(kl_per_token),
        'total_kl_penalty': sum(kl_coef * kl for kl in kl_per_token),
        'reward_model_score': reward_model_score,
        'total_reward': sum(rewards),
    }

    return rewards, stats


def visualize_rewards(rewards: List[float], kl_per_token: List[float],
                       kl_coef: float, rm_score: float):
    """Visualize reward distribution across tokens."""
    print("\n" + "=" * 70)
    print(" TOKEN-LEVEL REWARD BREAKDOWN")
    print("=" * 70)

    print(f"\n{'Token':<8} {'KL':<12} {'KL Penalty':<12} {'RM Contrib':<12} {'Reward':<12}")
    print("-" * 60)

    num_tokens = len(rewards)
    for t in range(num_tokens):
        kl = kl_per_token[t]
        penalty = kl_coef * kl
        rm_contrib = rm_score if t == num_tokens - 1 else 0

        print(f"{t:<8} {kl:>+.4f}     {-penalty:>+.4f}      {rm_contrib:>+.4f}      {rewards[t]:>+.4f}")

    print("-" * 60)
    print(f"{'Total':<8} {sum(kl_per_token):>+.4f}     {-kl_coef*sum(kl_per_token):>+.4f}      "
          f"{rm_score:>+.4f}      {sum(rewards):>+.4f}")


def demonstrate_kl_penalty_effect():
    """Show how KL coefficient affects learning."""
    print("\n" + "=" * 70)
    print(" KL COEFFICIENT EFFECT")
    print("=" * 70)

    # Simulated response with moderate divergence
    actor_lps = [-1.0, -1.2, -0.8, -1.5, -0.9]  # Actor log probs
    ref_lps = [-1.1, -1.0, -1.0, -1.2, -1.0]    # Reference log probs
    rm_score = 0.5  # Positive reward from RM

    print("\nScenario: Response with RM score = 0.5")
    print("Actor is somewhat divergent from reference\n")

    kl_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5]

    print(f"{'KL Coef':<10} {'Total KL Penalty':<18} {'Net Reward':<12} {'Effect':<20}")
    print("-" * 60)

    for kl_coef in kl_values:
        rewards, stats = compute_rewards(rm_score, actor_lps, ref_lps, kl_coef)
        net_reward = stats['total_reward']

        if net_reward > rm_score * 0.8:
            effect = "Weak penalty"
        elif net_reward > 0:
            effect = "Moderate penalty"
        elif net_reward > -0.5:
            effect = "Strong penalty"
        else:
            effect = "Overwhelming penalty"

        print(f"{kl_coef:<10} {stats['total_kl_penalty']:>+.4f}            {net_reward:>+.4f}       {effect}")

    print("""
Interpretation:
  β ≈ 0.00: No penalty, risk of reward hacking
  β ≈ 0.02: Typical value, balanced
  β ≈ 0.10: Strong regularization, slower learning
  β ≈ 0.50: KL dominates, almost no RM signal
""")


def demonstrate_kl_scenarios():
    """Show different KL divergence scenarios."""
    print("\n" + "=" * 70)
    print(" KL DIVERGENCE SCENARIOS")
    print("=" * 70)

    kl_coef = 0.02

    scenarios = [
        ("Low divergence (similar to reference)", [-1.0, -1.1, -0.9], [-1.0, -1.0, -1.0]),
        ("High divergence (very different)", [-0.5, -2.0, -0.3], [-1.5, -0.8, -1.2]),
        ("More confident than reference", [-0.2, -0.3, -0.2], [-1.0, -1.0, -1.0]),
        ("Less confident than reference", [-2.0, -2.5, -2.0], [-1.0, -1.0, -1.0]),
    ]

    rm_score = 0.5

    for name, actor_lps, ref_lps in scenarios:
        kl_per_token = compute_kl_divergence(actor_lps, ref_lps)
        rewards, stats = compute_rewards(rm_score, actor_lps, ref_lps, kl_coef)

        print(f"\n{name}:")
        print(f"  Actor log probs: {actor_lps}")
        print(f"  Ref log probs:   {ref_lps}")
        print(f"  Per-token KL:    {[f'{k:.2f}' for k in kl_per_token]}")
        print(f"  Total KL:        {stats['total_kl']:.4f}")
        print(f"  KL penalty:      {stats['total_kl_penalty']:.4f}")
        print(f"  Net reward:      {stats['total_reward']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RLHF Reward Calculator")
    parser.add_argument("--kl-coef", "-k", type=float, default=0.02,
                        help="KL penalty coefficient")
    parser.add_argument("--rm-score", "-r", type=float, default=0.5,
                        help="Reward model score")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " RLHF REWARD CALCULATOR".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Example response
    actor_log_probs = [-1.2, -0.8, -1.5, -0.9, -1.1, -1.0, -0.7, -1.3]
    ref_log_probs = [-1.0, -1.0, -1.2, -1.0, -1.0, -1.1, -1.0, -1.0]

    print(f"\nConfiguration:")
    print(f"  KL coefficient (β): {args.kl_coef}")
    print(f"  Reward model score: {args.rm_score}")
    print(f"  Response length: {len(actor_log_probs)} tokens")

    # Compute rewards
    rewards, stats = compute_rewards(
        args.rm_score, actor_log_probs, ref_log_probs, args.kl_coef
    )

    # Compute KL for visualization
    kl_per_token = compute_kl_divergence(actor_log_probs, ref_log_probs)

    # Visualize
    visualize_rewards(rewards, kl_per_token, args.kl_coef, args.rm_score)

    # Show stats
    print("\n" + "=" * 70)
    print(" SUMMARY STATISTICS")
    print("=" * 70)
    print(f"""
Reward Model Score:    {stats['reward_model_score']:>+.4f}
Total KL Divergence:   {stats['total_kl']:>+.4f}
Total KL Penalty:      {stats['total_kl_penalty']:>+.4f}
Net Total Reward:      {stats['total_reward']:>+.4f}

Reward Composition:
  RM contribution:     {stats['reward_model_score']:>+.4f} (at last token)
  KL penalty:          {-stats['total_kl_penalty']:>+.4f} (distributed)
  ────────────────────────────
  Net reward:          {stats['total_reward']:>+.4f}
""")

    # Demonstrate KL effects
    demonstrate_kl_penalty_effect()
    demonstrate_kl_scenarios()

    # Key insights
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. REWARD = RM_SCORE - β × KL
   The KL penalty prevents the actor from diverging too far from
   the reference model, avoiding "reward hacking".

2. KL IS COMPUTED PER-TOKEN
   Each token's probability is compared to the reference.
   This gives fine-grained control over divergence.

3. RM SCORE IS TYPICALLY END-ONLY
   The reward model scores the complete response.
   This score appears only at the last token.
   GAE propagates it backwards during training.

4. β IS A CRITICAL HYPERPARAMETER
   Too low: Reward hacking, degenerate solutions
   Too high: Learning is too slow, policy doesn't change
   Typical values: 0.01 - 0.05

5. NEGATIVE REWARDS ARE OK
   The policy gradient cares about relative advantages,
   not absolute reward values.
""")


if __name__ == "__main__":
    main()
