# Task: Heavy Model Training

This example demonstrates a larger Transformer architecture and the usage of memory-optimization features in TensorLogic (TL).

## Overview

The "Heavy" model is a 4-layer GPT with an embedding dimension of 384. Training such a model on consumer GPUs or CPUs requires efficient memory management, which this implementation demonstrates via two key TL features:

1.  **Activation Checkpointing**: The `checkpoint()` function is used for forward passes of individual blocks. This saves memory by not storing intermediate activations, recomputing them during the backward pass instead.
2.  **Explicit Gradient Clearing**: The `tl_clear_grads()` function is called frequently to free up memory used by gradients that are no longer needed.

## Configuration

- **Model**: 4-Layer GPT.
- **Hidden Dimension**: 384.
- **Projected Hidden Dimension**: 1536 (Hidden * 4).
- **Techniques**: Xavier-ish initialization, `checkpoint`, `tl_clear_grads`.

## Usage

To train the heavy model:
```bash
cargo run --release --bin tl -- examples/tasks/heavy/train_heavy.tl
```

The model weights will be saved to `model_heavy.safetensors` every 5 epochs.
