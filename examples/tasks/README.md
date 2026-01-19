# TensorLogic (TL) Examples: Tasks

This directory contains various AI and Machine Learning tasks implemented in TensorLogic (TL). These examples serve as both tests for the compiler and educational templates for building neural networks from scratch.

## Available Tasks

| Task | Description | Core Components |
|---|---|---|
| [**addition**](./addition) | 2-digit addition task. Demonstrates manual implementation of core layers. | GPT, Custom Embedding/Attention |
| [**heavy**](./heavy) | A larger GPT model for stress-testing. Features memory-saving techniques. | 4-layer GPT, Activation Checkpointing |
| [**mnist**](./mnist) | Classic hand-written digit classification. | Logistic Regression (Linear Layer) |
| [**paper**](./paper) | Baseline 2-layer GPT model referred to in documentation. | GPT, Standard Transformer Architecture |
| [**recall**](./recall) | Associative Recall task. Tests retrieval capabilities of attention. | Causal Self-Attention, K-V retrieval |
| [**reverse**](./reverse) | Sequence Reversal task. Features multi-head attention. | Transformer, Multi-Head Attention |

## Getting Started

Each directory contains:
- `train_xxx.tl`: The training script.
- `infer_xxx.tl`: The inference/validation script (if applicable).
- A `README.md` with specific details for that task.

To run a task (e.g., MNIST training):
```bash
cargo run --release --bin tl -- examples/tasks/mnist/train.tl
```
