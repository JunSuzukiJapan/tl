# Task: Sequence Reversal

A sequence modeling task where the model must learn to reverse an input sequence of digits.

## Overview

In this task, the model is given a sequence of digits followed by a separator and is expected to output the same digits in reverse order:
`Input: [d1, d2, d3, d4, SEP, d4, d3, d2, d1]`

This task specifically tests the model's understanding of **Position Embeddings** and its ability to utilize **Multi-Head Attention** to map relative positions.

## Implementation Details

- **Model**: Transformer with 2 Blocks.
- **Attention**: 4-head Multi-Head Attention (explicitly reshaped and managed in TL).
- **Embeddings**: Learned Position Embeddings.
- **Hidden Dimension**: 192.

## Usage

To train the reversal model:
```bash
cargo run --release --bin tl -- examples/tasks/reverse/reverse_train.tl
```

Weights are saved to `reverse_model.safetensors`.
