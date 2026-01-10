# Task: Baseline GPT (Paper Architecture)

This task implements a standard 2-layer Transformer architecture, serving as a baseline for performance and accuracy comparisons.

## Overview

The "Paper" model configuration is a classic 2-layer GPT-style Transformer with an embedding dimension of 256. It is used to verify the core implementation of self-attention and feed-forward networks in TensorLogic (TL).

## Configuration

- **Architecture**: 2-Layer GPT.
- **Embedding Dimension**: 256.
- **Vocabulary Size**: 13.
- **Sequence Length**: 12 tokens.
- **Learning Rate**: 0.001.

## Usage

To train the baseline model:
```bash
cargo run --release --bin tl -- run examples/tasks/paper/train_paper.tl
```

The model weights will be saved to `model_paper.safetensors`. This model is often used as a starting point for more complex experiments.
