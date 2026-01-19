# Task: MNIST Digit Classification

A classic machine learning example implementing hand-written digit classification using TensorLogic (TL).

## Overview

This task trains a simple Linear Layer (Logistic Regression) to classify 28x28 grayscale images of digits (0-9). It demonstrates:
- Loading binary data in the `idx` format.
- Batch processing using the `slice()` method.
- Standard training metrics (Average Loss per Epoch).

## Dataset

The dataset is located in `examples/tasks/mnist/data/` and consists of:
- `train-images-idx3-ubyte`: 60,000 training images.
- `train-labels-idx1-ubyte`: 60,000 training labels.
- `t10k-images-idx3-ubyte`: 10,000 test images.
- `t10k-labels-idx1-ubyte`: 10,000 test labels.

Detailed data loading logic is implemented in `mnist_common.tl`.

## Model

- **Architecture**: Single Linear layer (Input: 784, Output: 10).
- **Optimizer**: Simple SGD (implemented via `model.step(lr)`).
- **Loss**: Cross Entropy.

## Usage

To train the model:
```bash
cargo run --release --bin tl -- examples/tasks/mnist/train.tl
```

To run a single inference (predicting from a sample PNG):
```bash
cargo run --release --bin tl -- examples/tasks/mnist/infer.tl
```
