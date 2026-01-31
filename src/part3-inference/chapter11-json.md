# json_constraint_demo.py

> Generate valid JSON using constrained decoding

This script shows how to ensure LLM output follows a specific format using grammar-based constraints.

## What It Does

1. Defines a simple JSON grammar
2. At each step, identifies valid next tokens
3. Masks invalid tokens
4. Generates guaranteed-valid JSON

## Run It

```bash
python tutorial/part3-inference/chapter11-spec-constraint/scripts/json_constraint_demo.py
```

## Example Output

```
=== JSON Constraint Decoding Demo ===

Target schema:
{
  "name": <string>,
  "age": <number>,
  "active": <boolean>
}

Generation trace:

Step 1: State = START_OBJECT
  Valid tokens: ['{']
  Sampled: '{'

Step 2: State = EXPECT_KEY
  Valid tokens: ['"name"', '"age"', '"active"']
  Sampled: '"name"'

Step 3: State = EXPECT_COLON
  Valid tokens: [':']
  Sampled: ':'

Step 4: State = EXPECT_STRING
  Valid tokens: ['"', 'a'-'z', 'A'-'Z', ...]
  Sampled: '"Alice"'

...

Final output (guaranteed valid JSON):
{
  "name": "Alice",
  "age": 30,
  "active": true
}
```

## The Technique

```python
def constrained_generate(model, grammar):
    state = grammar.initial_state()
    output = []

    while not state.is_finished():
        # Get model's preferences
        logits = model.get_logits(output)

        # Mask invalid tokens
        valid_tokens = state.get_valid_tokens()
        for i in range(vocab_size):
            if i not in valid_tokens:
                logits[i] = float('-inf')

        # Sample from valid tokens only
        token = sample(logits)
        output.append(token)
        state = state.advance(token)

    return output
```

## Why This Matters

Without constraints:
- Model might output invalid JSON
- Need retry logic
- Unpredictable latency

With constraints:
- Always valid output
- Single generation attempt
- Predictable behavior

## Source Code

```python
{{#include ../../tutorial/part3-inference/chapter11-spec-constraint/scripts/json_constraint_demo.py}}
```
