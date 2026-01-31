# ML Systems for Dummies

> From distributed primitives to production RLHF: A hands-on journey through ML infrastructure

**[Read the Tutorial Online](https://abhishekmaiti.com/mlsys-for-dummies/)** | [View Source](./tutorial/)

---

## What is This?

A comprehensive tutorial teaching ML infrastructure systems—from the basics of distributed computing to production RLHF architectures. If you're comfortable with PyTorch and transformers but wonder "how do people actually train GPT-4?", this is for you.

## Who Should Read This?

- **Strong ML background**: You know PyTorch, can train models, understand attention
- **New to systems**: You haven't done distributed training, don't know NCCL from TCP
- **Curious about scale**: You want to understand how 1000-GPU training actually works

## What You'll Learn

| Part | Topic | Chapters |
|------|-------|----------|
| **I** | Distributed Computing Foundations | rank/world_size, send/recv, collectives, NCCL |
| **II** | Parallelism Strategies | DDP, ZeRO/FSDP, tensor parallel, pipeline parallel, MoE |
| **III** | LLM Inference Systems | KV cache, PagedAttention, CUDA graphs, speculative decoding |
| **IV** | RLHF Systems | PPO, reward models, multi-model orchestration |

## Quick Start

### Run the Scripts

Every chapter includes runnable Python scripts:

```bash
# Verify your setup
python tutorial/part1-distributed/chapter01-first-program/scripts/verify_setup.py

# Your first distributed program
python tutorial/part1-distributed/chapter01-first-program/scripts/hello_distributed.py
```

**Requirements:**
```bash
pip install torch
pip install gymnasium  # Optional, for RL chapter
```

No GPU required—all scripts have CPU fallback with the `gloo` backend.

### Read the Tutorial

**Option 1: Online** (recommended)

Visit [https://abhishekmaiti.github.io/mlsys-for-dummies/](https://abhishekmaiti.github.io/mlsys-for-dummies/)

**Option 2: Locally**

```bash
# Install mdBook
brew install mdbook  # macOS
# or: cargo install mdbook

# Serve locally
mdbook serve
# Opens http://localhost:3000
```

## Repository Structure

```
mlsys-for-dummies/
├── tutorial/                    # Original content and scripts
│   ├── part1-distributed/       # Chapters 1-4
│   ├── part2-parallelism/       # Chapters 5-7
│   ├── part3-inference/         # Chapters 8-11
│   └── part4-rlhf/              # Chapters 12-14
├── src/                         # mdBook website source
├── book.toml                    # mdBook configuration
├── theme/                       # Custom CSS
└── .github/workflows/           # Auto-deploy to GitHub Pages
```

## Contributing

Found an error? Have a suggestion? PRs welcome!

- **Content fixes**: Edit files in `tutorial/` (README.md or scripts)
- **Website changes**: Edit files in `src/`
- **Run locally**: Use `mdbook serve` to preview changes

## License

MIT

---

*"The best way to understand distributed systems is to build one. The second best way is this tutorial."*
