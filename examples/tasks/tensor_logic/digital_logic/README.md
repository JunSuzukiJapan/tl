# Differentiable Digital Logic (XOR Learning)

This example demonstrates learning a neural network that implements the XOR Boolean function, showcasing differentiable approximation of digital logic.

## Concept

- **XOR Truth Table**:
    - (0, 0) → 0
    - (0, 1) → 1
    - (1, 0) → 1
    - (1, 1) → 0
- **Network Architecture**: 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid)
- **Training**: Gradient descent to minimize MSE between predictions and truth table.

## Implementation Details

- **Differentiable Gates**: Sigmoid approximates step function (0/1), enabling gradient flow.
- **2-Layer Network**: XOR requires at least one hidden layer (not linearly separable).
- **Learning Rate**: 1.0 (relatively high for this simple problem).

## Results

After 1000 iterations:
- Loss converges from ~1.1 to ~0.005
- Predictions accurately approximate XOR:
    - (0, 0) → ~0.03
    - (0, 1) → ~0.97
    - (1, 0) → ~0.97
    - (1, 1) → ~0.03

## Usage

```bash
tl run examples/tasks/tensor_logic/digital_logic/logic.tl
```
