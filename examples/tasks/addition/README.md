# Task: 2-Digit Addition

This task trains a Transformer model to perform 2-digit addition (e.g., `12 + 34 = 46`).

## Overview

The primary goal of this example is to demonstrate how to implement complex neural network modules directly in TensorLogic (TL) without relying on high-level built-in functions. 

The implementation features custom-coded versions of:
- **One-Hot Embedding**: Manually iterating over indices to create embedding representations.
- **Causal Masking**: Generating a lower-triangular matrix using nested `for` loops.
- **Cross Entropy Loss**: Manually computing log-probabilities and the mean loss value.

## Implementation Details

- **Scripts**:
    - `train_add.tl`: Training loop for the 100-epoch addition task.
    - `infer_add.tl`: Script to load weights and perform inference.
    - `train_verify_2digit.tl`: Stability test script.
- **Model**: 3-Layer GPT-style Transformer.
- **Vocabulary Size**: 13 (Digits 0-9, `+`, `=`, and `PAD`).
- **Embedding Size**: 128.

## Usage

To train the model:
```bash
cargo run --release --bin tl -- run examples/tasks/addition/train_add.tl
```

The model weights will be saved as `model_add.safetensors`.
