# speculative_demo.py

> Simulate speculative decoding to understand the speedup

This script demonstrates how speculative decoding works by simulating the draft-verify process.

## What It Does

1. Simulates a draft model generating k tokens
2. Simulates a target model verifying them
3. Shows acceptance/rejection for each token
4. Calculates effective speedup

## Run It

```bash
python tutorial/part3-inference/chapter11-spec-constraint/scripts/speculative_demo.py
```

## Example Output

```
=== Speculative Decoding Demo ===

Settings:
  Draft length (k): 5
  Acceptance rate (γ): 0.80

Simulation (10 generations):

Generation 1:
  Draft tokens:  ["The", "quick", "brown", "fox", "jumps"]
  Target verify: [✓ accept] [✓ accept] [✓ accept] [✗ reject] [- skip]
  Tokens accepted: 3 (with 1 target forward pass)

Generation 2:
  Draft tokens:  ["over", "the", "lazy", "dog", "."]
  Target verify: [✓ accept] [✓ accept] [✓ accept] [✓ accept] [✓ accept]
  Tokens accepted: 5 (with 1 target forward pass)

...

Summary:
  Total tokens generated: 38
  Total target forward passes: 10
  Tokens per pass: 3.8 (vs 1.0 without speculation)
  Theoretical speedup: 3.8x

Cost breakdown:
  Target passes: 10 × 100ms = 1000ms
  Draft passes: 50 × 5ms = 250ms
  Total time: 1250ms
  Time without speculation: 3800ms
  Actual speedup: 3.04x
```

## The Math

Expected tokens per target pass:
```
E[tokens] = Σ(i=0 to k) γⁱ = (1 - γ^(k+1)) / (1 - γ)
```

For γ=0.8, k=5: E[tokens] = 3.36

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter11-spec-constraint/scripts/speculative_demo.py}}
```
