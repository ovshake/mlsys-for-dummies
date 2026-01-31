#!/usr/bin/env python3
"""
Prefix Sharing Demonstration

This script demonstrates how prefix sharing (RadixCache) saves memory
by reusing KV cache for common prompt prefixes.

Usage:
    python prefix_sharing_demo.py
    python prefix_sharing_demo.py --num-requests 100
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict


@dataclass
class RadixNode:
    """A node in the radix tree."""
    token: Optional[int] = None
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)
    kv_index: Optional[int] = None  # Index into KV cache
    ref_count: int = 0  # Number of requests using this node


class RadixTree:
    """
    Simplified RadixCache for demonstration.

    Real implementations are more complex with:
    - Compression (multiple tokens per node)
    - LRU eviction
    - Reference counting for safe deletion
    """

    def __init__(self):
        self.root = RadixNode()
        self.next_kv_index = 0
        self.total_nodes = 0
        self.shared_nodes = 0

    def insert(self, tokens: List[int]) -> List[int]:
        """
        Insert a sequence and return KV indices.

        Returns list of KV cache indices for each token.
        Reuses existing indices where prefixes match.
        """
        kv_indices = []
        node = self.root

        for token in tokens:
            if token not in node.children:
                # Create new node
                new_node = RadixNode(token=token)
                new_node.kv_index = self.next_kv_index
                self.next_kv_index += 1
                node.children[token] = new_node
                self.total_nodes += 1
            else:
                # Reuse existing node (prefix sharing!)
                self.shared_nodes += 1

            node = node.children[token]
            node.ref_count += 1
            kv_indices.append(node.kv_index)

        return kv_indices

    def get_stats(self) -> Dict:
        """Get statistics about the tree."""
        return {
            'total_nodes': self.total_nodes,
            'shared_accesses': self.shared_nodes,
            'unique_kv_entries': self.next_kv_index,
        }


def visualize_tree(node: RadixNode, prefix: str = "", is_last: bool = True,
                   depth: int = 0, max_depth: int = 5) -> List[str]:
    """Visualize the radix tree structure."""
    lines = []

    if depth > max_depth:
        return lines

    connector = "└── " if is_last else "├── "
    token_str = f"[{node.token}]" if node.token is not None else "[root]"
    ref_str = f" (refs: {node.ref_count})" if node.ref_count > 0 else ""
    lines.append(f"{prefix}{connector}{token_str}{ref_str}")

    children = list(node.children.values())
    for i, child in enumerate(children):
        extension = "    " if is_last else "│   "
        child_is_last = (i == len(children) - 1)
        lines.extend(visualize_tree(child, prefix + extension, child_is_last,
                                   depth + 1, max_depth))

    return lines


def demo_prefix_sharing():
    """Demonstrate prefix sharing with example requests."""
    print("=" * 70)
    print(" RADIX CACHE PREFIX SHARING DEMO")
    print("=" * 70)

    # Simulate a tokenizer (just use word indices)
    def tokenize(text: str) -> List[int]:
        words = text.lower().split()
        return [hash(w) % 1000 for w in words]

    # Example requests with shared prefixes
    requests = [
        "You are a helpful assistant. What is the capital of France?",
        "You are a helpful assistant. What is the capital of Germany?",
        "You are a helpful assistant. What is the largest planet?",
        "You are a helpful assistant. Tell me a joke.",
        "You are a coding assistant. Write a Python function.",
        "You are a coding assistant. Explain recursion.",
    ]

    tree = RadixTree()
    total_tokens = 0
    request_indices = []

    print("\nProcessing requests:\n")

    for i, request in enumerate(requests):
        tokens = tokenize(request)
        total_tokens += len(tokens)

        kv_indices = tree.insert(tokens)
        request_indices.append(kv_indices)

        print(f"Request {i + 1}: {request[:50]}...")
        print(f"  Tokens: {len(tokens)}, KV indices assigned: {len(set(kv_indices))} unique")

    # Statistics
    stats = tree.get_stats()

    print("\n" + "-" * 70)
    print(" MEMORY ANALYSIS")
    print("-" * 70)

    print(f"\nWithout prefix sharing:")
    print(f"  Total tokens across all requests: {total_tokens}")
    print(f"  KV cache entries needed: {total_tokens}")

    print(f"\nWith prefix sharing (RadixCache):")
    print(f"  Unique KV cache entries: {stats['unique_kv_entries']}")
    print(f"  Shared prefix accesses: {stats['shared_accesses']}")

    savings = (1 - stats['unique_kv_entries'] / total_tokens) * 100
    print(f"\nMemory savings: {savings:.1f}%")

    # Visualize tree (simplified)
    print("\n" + "-" * 70)
    print(" RADIX TREE STRUCTURE (first 5 levels)")
    print("-" * 70)
    print("\n".join(visualize_tree(tree.root)))


def analyze_system_prompt_sharing(num_requests: int, system_prompt_len: int,
                                   user_prompt_len: int, kv_bytes_per_token: int):
    """Analyze memory savings from system prompt sharing."""
    print("\n" + "=" * 70)
    print(" SYSTEM PROMPT SHARING ANALYSIS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Number of requests: {num_requests}")
    print(f"  System prompt length: {system_prompt_len} tokens")
    print(f"  User prompt length: {user_prompt_len} tokens (average)")
    print(f"  KV bytes per token: {kv_bytes_per_token}")

    total_tokens = num_requests * (system_prompt_len + user_prompt_len)
    without_sharing = total_tokens * kv_bytes_per_token

    # With sharing: system prompt cached once, user prompts unique
    with_sharing = (system_prompt_len + num_requests * user_prompt_len) * kv_bytes_per_token

    savings = without_sharing - with_sharing
    savings_pct = (savings / without_sharing) * 100

    print(f"\nMemory usage:")
    print(f"  Without sharing: {without_sharing / 1e9:.2f} GB")
    print(f"  With sharing: {with_sharing / 1e9:.2f} GB")
    print(f"  Saved: {savings / 1e9:.2f} GB ({savings_pct:.1f}%)")

    # Break down by component
    system_memory = system_prompt_len * kv_bytes_per_token
    user_memory = num_requests * user_prompt_len * kv_bytes_per_token

    print(f"\nWith sharing breakdown:")
    print(f"  System prompt (shared): {system_memory / 1e6:.2f} MB (cached once)")
    print(f"  User prompts (unique): {user_memory / 1e9:.2f} GB")


def analyze_few_shot_sharing(num_requests: int, num_examples: int,
                              example_len: int, query_len: int,
                              kv_bytes_per_token: int):
    """Analyze memory savings from few-shot example sharing."""
    print("\n" + "=" * 70)
    print(" FEW-SHOT EXAMPLE SHARING ANALYSIS")
    print("=" * 70)

    few_shot_len = num_examples * example_len

    print(f"\nConfiguration:")
    print(f"  Number of requests: {num_requests}")
    print(f"  Few-shot examples: {num_examples} × {example_len} = {few_shot_len} tokens")
    print(f"  Query length: {query_len} tokens (average)")

    total_tokens = num_requests * (few_shot_len + query_len)
    without_sharing = total_tokens * kv_bytes_per_token

    with_sharing = (few_shot_len + num_requests * query_len) * kv_bytes_per_token

    savings = without_sharing - with_sharing
    savings_pct = (savings / without_sharing) * 100

    print(f"\nMemory usage:")
    print(f"  Without sharing: {without_sharing / 1e9:.2f} GB")
    print(f"  With sharing: {with_sharing / 1e9:.2f} GB")
    print(f"  Saved: {savings / 1e9:.2f} GB ({savings_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prefix Sharing Demo")
    parser.add_argument("--num-requests", "-n", type=int, default=100,
                        help="Number of requests for analysis")
    parser.add_argument("--system-prompt-len", type=int, default=500,
                        help="System prompt length in tokens")
    parser.add_argument("--user-prompt-len", type=int, default=100,
                        help="Average user prompt length")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " PREFIX SHARING (RADIXCACHE) DEMONSTRATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Basic demo
    demo_prefix_sharing()

    # LLaMA-70B KV bytes per token (with GQA)
    kv_bytes = 2 * 80 * 8 * 128 * 2  # 327,680 bytes

    # System prompt sharing analysis
    analyze_system_prompt_sharing(
        num_requests=args.num_requests,
        system_prompt_len=args.system_prompt_len,
        user_prompt_len=args.user_prompt_len,
        kv_bytes_per_token=kv_bytes
    )

    # Few-shot sharing analysis
    analyze_few_shot_sharing(
        num_requests=args.num_requests,
        num_examples=5,
        example_len=200,
        query_len=50,
        kv_bytes_per_token=kv_bytes
    )

    # Key insights
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. SYSTEM PROMPTS ARE FREE (almost)
   - First request pays the cost
   - Subsequent requests share the KV cache
   - Especially valuable for long system prompts

2. FEW-SHOT EXAMPLES BENEFIT HUGELY
   - 5 examples × 200 tokens = 1000 tokens shared
   - With 100 requests: 99% memory reduction for examples!

3. RADIXCACHE IS AUTOMATIC
   - No manual prefix specification needed
   - Tree structure detects sharing automatically
   - Works for any common prefix

4. LIMITATIONS:
   - Only exact prefix matches benefit
   - Different orderings = different prefixes
   - Token-level sharing (not semantic)

5. REAL-WORLD IMPACT:
   - APIs with shared system prompts: massive savings
   - Batch inference with templates: huge efficiency
   - Speculative decoding: shared draft prefixes
""")


if __name__ == "__main__":
    main()
