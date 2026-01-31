#!/usr/bin/env python3
"""
JSON Constraint Decoding Demonstration

This script demonstrates how constraint decoding ensures valid JSON output
by masking invalid tokens at each generation step.

Usage:
    python json_constraint_demo.py
"""

import argparse
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Set, Optional


class JsonState(Enum):
    """States in simplified JSON grammar."""
    START = auto()           # Expecting { or [
    OBJECT_START = auto()    # Just saw {, expecting key or }
    OBJECT_KEY = auto()      # Expecting string key
    OBJECT_COLON = auto()    # Expecting :
    OBJECT_VALUE = auto()    # Expecting value
    OBJECT_COMMA = auto()    # Expecting , or }
    ARRAY_START = auto()     # Just saw [, expecting value or ]
    ARRAY_VALUE = auto()     # Expecting value
    ARRAY_COMMA = auto()     # Expecting , or ]
    STRING = auto()          # Inside a string
    NUMBER = auto()          # Inside a number
    DONE = auto()           # Finished


@dataclass
class Token:
    """Represents a vocabulary token."""
    id: int
    text: str
    is_valid: bool = True


class SimplifiedJsonGrammar:
    """
    Simplified JSON grammar for demonstration.

    Real implementations use proper grammar parsing (e.g., lark, interegular).
    """

    def __init__(self):
        self.state = JsonState.START
        self.stack = []  # Track nested structures

        # Simplified vocabulary
        self.vocab = {
            0: "{",
            1: "}",
            2: "[",
            3: "]",
            4: ":",
            5: ",",
            6: '"name"',
            7: '"value"',
            8: '"id"',
            9: '"type"',
            10: '"hello"',
            11: '"world"',
            12: "123",
            13: "456",
            14: "true",
            15: "false",
            16: "null",
        }

    def get_valid_tokens(self) -> Set[int]:
        """Return set of valid next token IDs given current state."""
        valid = set()

        if self.state == JsonState.START:
            valid = {0, 2}  # { or [

        elif self.state == JsonState.OBJECT_START:
            valid = {1, 6, 7, 8, 9}  # } or string keys

        elif self.state == JsonState.OBJECT_KEY:
            valid = {6, 7, 8, 9}  # String keys

        elif self.state == JsonState.OBJECT_COLON:
            valid = {4}  # :

        elif self.state == JsonState.OBJECT_VALUE:
            valid = {0, 2, 6, 7, 10, 11, 12, 13, 14, 15, 16}  # Any value

        elif self.state == JsonState.OBJECT_COMMA:
            if self.stack and self.stack[-1] == "object":
                valid = {1, 5}  # } or ,
            else:
                valid = {1}

        elif self.state == JsonState.ARRAY_START:
            valid = {0, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16}  # ] or values

        elif self.state == JsonState.ARRAY_VALUE:
            valid = {0, 2, 6, 7, 10, 11, 12, 13, 14, 15, 16}  # Any value

        elif self.state == JsonState.ARRAY_COMMA:
            valid = {3, 5}  # ] or ,

        return valid

    def advance(self, token_id: int):
        """Advance grammar state based on token."""
        token = self.vocab[token_id]

        if token == "{":
            self.stack.append("object")
            self.state = JsonState.OBJECT_START

        elif token == "}":
            if self.stack and self.stack[-1] == "object":
                self.stack.pop()
            if not self.stack:
                self.state = JsonState.DONE
            else:
                self.state = JsonState.OBJECT_COMMA if self.stack[-1] == "object" else JsonState.ARRAY_COMMA

        elif token == "[":
            self.stack.append("array")
            self.state = JsonState.ARRAY_START

        elif token == "]":
            if self.stack and self.stack[-1] == "array":
                self.stack.pop()
            if not self.stack:
                self.state = JsonState.DONE
            else:
                self.state = JsonState.OBJECT_COMMA if self.stack[-1] == "object" else JsonState.ARRAY_COMMA

        elif token == ":":
            self.state = JsonState.OBJECT_VALUE

        elif token == ",":
            if self.stack[-1] == "object":
                self.state = JsonState.OBJECT_KEY
            else:
                self.state = JsonState.ARRAY_VALUE

        elif token.startswith('"') and self.state in [JsonState.OBJECT_START, JsonState.OBJECT_KEY]:
            self.state = JsonState.OBJECT_COLON

        elif token.startswith('"') or token in ["true", "false", "null"] or token.isdigit():
            if self.state == JsonState.OBJECT_VALUE:
                self.state = JsonState.OBJECT_COMMA
            elif self.state in [JsonState.ARRAY_START, JsonState.ARRAY_VALUE]:
                self.state = JsonState.ARRAY_COMMA


def constrained_generation(grammar: SimplifiedJsonGrammar,
                           max_tokens: int = 20) -> List[str]:
    """
    Generate tokens with grammar constraints.

    At each step:
    1. Get valid tokens from grammar
    2. "Sample" from valid tokens (random for demo)
    3. Advance grammar state
    """
    generated = []

    for _ in range(max_tokens):
        if grammar.state == JsonState.DONE:
            break

        valid_tokens = grammar.get_valid_tokens()

        if not valid_tokens:
            print("Warning: No valid tokens!")
            break

        # In real implementation, this would be:
        # 1. Get logits from model
        # 2. Mask invalid tokens (set to -inf)
        # 3. Sample from softmax

        # Here we just pick randomly from valid tokens
        token_id = random.choice(list(valid_tokens))
        token_text = grammar.vocab[token_id]

        generated.append(token_text)
        grammar.advance(token_id)

    return generated


def demonstrate_token_masking():
    """Show how token masking works at each step."""
    print("=" * 70)
    print(" TOKEN MASKING DEMONSTRATION")
    print("=" * 70)

    grammar = SimplifiedJsonGrammar()

    steps = []
    generated = []

    for i in range(10):
        if grammar.state == JsonState.DONE:
            break

        valid_tokens = grammar.get_valid_tokens()
        all_tokens = set(grammar.vocab.keys())
        invalid_tokens = all_tokens - valid_tokens

        # Pick a random valid token
        token_id = random.choice(list(valid_tokens))
        token_text = grammar.vocab[token_id]

        steps.append({
            'step': i,
            'state': grammar.state.name,
            'valid': [grammar.vocab[t] for t in valid_tokens],
            'invalid_count': len(invalid_tokens),
            'chosen': token_text,
        })

        generated.append(token_text)
        grammar.advance(token_id)

    print("\nStep-by-step token masking:\n")

    for step in steps:
        print(f"Step {step['step']}:")
        print(f"  State: {step['state']}")
        print(f"  Valid tokens: {step['valid']}")
        print(f"  Masked (invalid): {step['invalid_count']} tokens")
        print(f"  Chosen: {step['chosen']}")
        print()

    result = "".join(generated)
    print(f"Generated JSON: {result}")


def compare_constrained_unconstrained():
    """Compare constrained vs unconstrained generation."""
    print("\n" + "=" * 70)
    print(" CONSTRAINED vs UNCONSTRAINED COMPARISON")
    print("=" * 70)

    vocab = list(SimplifiedJsonGrammar().vocab.values())

    print("\nUNCONSTRAINED (random tokens):")
    unconstrained = [random.choice(vocab) for _ in range(10)]
    result = "".join(unconstrained)
    print(f"  Generated: {result}")
    print(f"  Valid JSON: {is_valid_json_like(result)}")

    print("\nCONSTRAINED (grammar-guided):")
    random.seed(42)  # For reproducibility
    grammar = SimplifiedJsonGrammar()
    constrained = constrained_generation(grammar, max_tokens=15)
    result = "".join(constrained)
    print(f"  Generated: {result}")
    print(f"  Valid JSON: {is_valid_json_like(result)}")


def is_valid_json_like(s: str) -> bool:
    """Simple check if string looks like valid JSON structure."""
    s = s.strip()
    if not s:
        return False

    # Check balanced brackets
    stack = []
    for char in s:
        if char in "{[":
            stack.append(char)
        elif char == "}":
            if not stack or stack[-1] != "{":
                return False
            stack.pop()
        elif char == "]":
            if not stack or stack[-1] != "[":
                return False
            stack.pop()

    return len(stack) == 0 and (s.startswith("{") or s.startswith("["))


def show_real_world_usage():
    """Show real-world constraint decoding scenarios."""
    print("\n" + "=" * 70)
    print(" REAL-WORLD CONSTRAINT DECODING SCENARIOS")
    print("=" * 70)
    print("""
1. JSON SCHEMA CONSTRAINTS
   Force model output to match a specific schema:

   Schema: {"name": string, "age": number, "active": boolean}

   At each step, only allow tokens that can lead to valid schema:
   - After {"name": only allow ": and string tokens
   - After "name": "...", only allow , or }
   - etc.

2. SQL QUERY CONSTRAINTS
   Ensure valid SQL syntax:

   Grammar: SELECT columns FROM table WHERE condition

   Mask tokens that would break syntax:
   - After SELECT: only column names or *
   - After FROM: only table names
   - etc.

3. FUNCTION CALL CONSTRAINTS
   Match function signature:

   def greet(name: str, times: int = 1)

   Force output like: greet("Alice", 3)
   - First token must be function name
   - Then (
   - Then valid arguments matching types
   - etc.

4. REGEX PATTERN CONSTRAINTS
   Match patterns like email, URL, phone number:

   Email: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

   Each token must keep the partial output matching the pattern.

5. PROGRAMMING LANGUAGE CONSTRAINTS
   Generate syntactically valid code:

   Python grammar ensures:
   - Proper indentation
   - Balanced parentheses
   - Valid keywords

IMPLEMENTATION NOTE:
   Real systems use libraries like:
   - outlines (https://github.com/outlines-dev/outlines)
   - guidance (https://github.com/guidance-ai/guidance)
   - lmql (https://lmql.ai/)
""")


def main():
    parser = argparse.ArgumentParser(description="JSON Constraint Decoding Demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("╔" + "═" * 68 + "╗")
    print("║" + " JSON CONSTRAINT DECODING DEMONSTRATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    demonstrate_token_masking()
    compare_constrained_unconstrained()
    show_real_world_usage()

    # Summary
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. CONSTRAINT DECODING GUARANTEES VALIDITY
   Every generated token is checked against grammar
   Invalid tokens are masked (probability = 0)
   Output is always syntactically correct

2. MINIMAL QUALITY IMPACT
   Model still chooses among valid tokens
   Only invalid options are removed
   Semantic quality preserved

3. SLIGHT LATENCY INCREASE
   Grammar state must be tracked
   Valid token computation at each step
   Usually <10% overhead

4. COMPOSABLE WITH OTHER TECHNIQUES
   Works with speculative decoding
   Works with beam search
   Works with any sampling strategy

5. LIBRARY SUPPORT
   Use production libraries (outlines, guidance, lmql)
   They handle complex grammars efficiently
   Pre-compiled finite automata for speed
""")


if __name__ == "__main__":
    main()
