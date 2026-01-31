#!/usr/bin/env python3
"""
Speculative Decoding Demonstration

This script demonstrates how speculative decoding works:
- Draft model generates multiple candidate tokens
- Target model verifies them in parallel
- Acceptance/rejection based on probability ratio

Usage:
    python speculative_demo.py
    python speculative_demo.py --draft-length 5 --acceptance-rate 0.8
"""

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Token:
    """Represents a generated token."""
    id: int
    text: str
    draft_prob: float
    target_prob: float


def simulate_draft_model(prompt: str, num_tokens: int,
                         vocab: List[str]) -> List[Token]:
    """
    Simulate a draft model generating tokens.

    In reality, this would be a small LLM like LLaMA-7B.
    """
    tokens = []
    for _ in range(num_tokens):
        # Random token selection (simulated)
        token_id = random.randint(0, len(vocab) - 1)
        token_text = vocab[token_id]

        # Random probabilities (simulated)
        # Draft model is less confident
        draft_prob = random.uniform(0.3, 0.9)

        tokens.append(Token(
            id=token_id,
            text=token_text,
            draft_prob=draft_prob,
            target_prob=0  # Set by target model
        ))

    return tokens


def simulate_target_verification(draft_tokens: List[Token],
                                  base_acceptance_rate: float) -> List[Token]:
    """
    Simulate target model verification of draft tokens.

    In reality, this would run the large model (e.g., LLaMA-70B)
    on all draft tokens in parallel.
    """
    for token in draft_tokens:
        # Target model's probability (simulated)
        # Higher acceptance rate = closer to draft distribution
        if random.random() < base_acceptance_rate:
            # Target agrees or is more confident
            token.target_prob = token.draft_prob * random.uniform(0.9, 1.5)
        else:
            # Target disagrees
            token.target_prob = token.draft_prob * random.uniform(0.1, 0.8)

        # Clamp to valid probability
        token.target_prob = min(1.0, token.target_prob)

    return draft_tokens


def speculative_acceptance(tokens: List[Token]) -> Tuple[List[Token], Optional[Token]]:
    """
    Apply speculative decoding acceptance criterion.

    For each token:
    - If p_target >= p_draft: ACCEPT
    - Else: ACCEPT with probability p_target / p_draft
    - On first rejection: sample from adjusted distribution, stop
    """
    accepted = []
    correction_token = None

    for i, token in enumerate(tokens):
        if token.target_prob >= token.draft_prob:
            # Definitely accept
            accepted.append(token)
        else:
            # Probabilistic acceptance
            acceptance_prob = token.target_prob / token.draft_prob
            if random.random() < acceptance_prob:
                accepted.append(token)
            else:
                # Reject: sample from (target - draft) distribution
                # Simulated as a new random token
                correction_token = Token(
                    id=random.randint(0, 99),
                    text=f"[corrected_{i}]",
                    draft_prob=0,
                    target_prob=token.target_prob
                )
                break  # Stop accepting after first rejection

    return accepted, correction_token


def run_speculative_decoding(prompt: str, target_length: int,
                              draft_length: int, acceptance_rate: float,
                              vocab: List[str]) -> Tuple[List[str], dict]:
    """
    Run speculative decoding simulation.

    Returns generated tokens and statistics.
    """
    generated = []
    stats = {
        'target_calls': 0,
        'draft_calls': 0,
        'tokens_accepted': 0,
        'tokens_rejected': 0,
        'total_tokens': 0,
    }

    while len(generated) < target_length:
        # Step 1: Draft model generates k tokens
        remaining = target_length - len(generated)
        k = min(draft_length, remaining)
        draft_tokens = simulate_draft_model(prompt, k, vocab)
        stats['draft_calls'] += 1

        # Step 2: Target model verifies all k tokens in parallel (ONE call)
        verified_tokens = simulate_target_verification(draft_tokens, acceptance_rate)
        stats['target_calls'] += 1

        # Step 3: Apply acceptance criterion
        accepted, correction = speculative_acceptance(verified_tokens)

        # Add accepted tokens
        for token in accepted:
            generated.append(token.text)
            stats['tokens_accepted'] += 1

        # Add correction token if any
        if correction:
            generated.append(correction.text)
            stats['tokens_rejected'] += 1

        stats['total_tokens'] = len(generated)

        # Update prompt for next iteration
        prompt = prompt + " " + " ".join(t.text for t in accepted)
        if correction:
            prompt += " " + correction.text

    return generated[:target_length], stats


def calculate_speedup(stats: dict, draft_length: int,
                       draft_cost_ratio: float = 0.1) -> dict:
    """
    Calculate speedup from speculative decoding.

    Args:
        stats: Statistics from run_speculative_decoding
        draft_length: Number of tokens drafted per call
        draft_cost_ratio: Cost of draft call relative to target (e.g., 0.1 = 10%)
    """
    tokens = stats['total_tokens']

    # Without speculative: one target call per token
    baseline_cost = tokens

    # With speculative: target + draft calls
    spec_cost = stats['target_calls'] + stats['draft_calls'] * draft_cost_ratio

    speedup = baseline_cost / spec_cost

    tokens_per_target_call = tokens / stats['target_calls']

    return {
        'baseline_cost': baseline_cost,
        'speculative_cost': spec_cost,
        'speedup': speedup,
        'tokens_per_target_call': tokens_per_target_call,
        'acceptance_rate': stats['tokens_accepted'] / (stats['tokens_accepted'] + stats['tokens_rejected'])
    }


def visualize_speculative_step(draft_tokens: List[Token],
                                accepted: List[Token],
                                correction: Optional[Token]):
    """Visualize a single speculative decoding step."""
    print("\nDraft tokens:")
    for i, token in enumerate(draft_tokens):
        status = "✓" if token in accepted else "✗"
        print(f"  {i}: {token.text:15} p_draft={token.draft_prob:.2f} "
              f"p_target={token.target_prob:.2f} {status}")

    if correction:
        print(f"\nCorrection token: {correction.text}")

    print(f"Accepted: {len(accepted)}/{len(draft_tokens)} tokens")


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Demo")
    parser.add_argument("--draft-length", "-k", type=int, default=5,
                        help="Number of tokens to draft")
    parser.add_argument("--target-length", "-n", type=int, default=50,
                        help="Total tokens to generate")
    parser.add_argument("--acceptance-rate", "-a", type=float, default=0.75,
                        help="Base acceptance rate (0-1)")
    parser.add_argument("--draft-cost", type=float, default=0.1,
                        help="Draft cost as fraction of target cost")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("╔" + "═" * 68 + "╗")
    print("║" + " SPECULATIVE DECODING DEMONSTRATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Simple vocabulary for demonstration
    vocab = [
        "the", "a", "is", "are", "was", "were", "has", "have", "had",
        "will", "would", "could", "should", "may", "might", "must",
        "and", "or", "but", "if", "then", "else", "when", "where",
        "who", "what", "which", "that", "this", "these", "those",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "good", "bad", "big", "small", "new", "old", "first", "last",
        "time", "way", "year", "day", "thing", "man", "world", "life",
    ]

    print(f"\nConfiguration:")
    print(f"  Draft length (k): {args.draft_length}")
    print(f"  Target length: {args.target_length}")
    print(f"  Base acceptance rate: {args.acceptance_rate}")
    print(f"  Draft cost ratio: {args.draft_cost}")

    # Run speculative decoding
    print("\n" + "=" * 70)
    print(" RUNNING SPECULATIVE DECODING")
    print("=" * 70)

    prompt = "Once upon a time"
    generated, stats = run_speculative_decoding(
        prompt, args.target_length, args.draft_length,
        args.acceptance_rate, vocab
    )

    print(f"\nGenerated text preview: {' '.join(generated[:20])}...")

    # Calculate speedup
    speedup_stats = calculate_speedup(stats, args.draft_length, args.draft_cost)

    # Results
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print(f"\nGeneration statistics:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Target model calls: {stats['target_calls']}")
    print(f"  Draft model calls: {stats['draft_calls']}")
    print(f"  Tokens accepted: {stats['tokens_accepted']}")
    print(f"  Tokens rejected: {stats['tokens_rejected']}")

    print(f"\nPerformance:")
    print(f"  Tokens per target call: {speedup_stats['tokens_per_target_call']:.2f}")
    print(f"  Effective acceptance rate: {speedup_stats['acceptance_rate']:.2%}")
    print(f"  Speedup: {speedup_stats['speedup']:.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print(" ANALYSIS")
    print("=" * 70)
    print(f"""
How Speculative Decoding Works:

1. DRAFT PHASE (fast, cheap)
   Small model generates {args.draft_length} candidate tokens quickly
   Cost: ~{args.draft_cost*100:.0f}% of target model

2. VERIFY PHASE (one parallel call)
   Large model processes ALL draft tokens in ONE forward pass
   Each token gets target model probability

3. ACCEPT/REJECT
   Token accepted if: p_target >= p_draft (always)
                  or: random < p_target/p_draft (probabilistic)
   First rejection triggers resampling and stops

4. GUARANTEE
   Output distribution is IDENTICAL to running target model alone
   No quality degradation!

Why It Works:
   - Verification is parallel (1 call for k tokens)
   - High acceptance rate ({speedup_stats['acceptance_rate']:.0%}) means few rejections
   - Draft model cost is negligible ({args.draft_cost*100:.0f}%)

When It Helps Most:
   - High acceptance rate (similar draft/target distributions)
   - Long generations (amortize setup cost)
   - Memory-bound systems (decode phase)

When It Helps Less:
   - Low acceptance rate (very different distributions)
   - Short generations (overhead not amortized)
   - Compute-bound systems (prefill phase)
""")


if __name__ == "__main__":
    main()
