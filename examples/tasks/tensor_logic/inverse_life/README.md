# Inverse Game of Life

This example demonstrates how to use differentiable programming to solve the "inverse" problem of Conway's Game of Life: given a target final state, find an initial state that evolves into it.

## Concept

- **Forward Problem**: Given an initial state $S_0$, compute the state $S_T$ after $T$ steps of the Game of Life rules.
- **Inverse Problem**: Given a target state $S_T$, find an initial state $S_0$ such that $\text{Forward}(S_0, T) \approx S_T$.

## Implementation Details

- **Grid**: `[1, 1, N, N]` tensor (NCHW format for conv2d).
- **Neighbor Counting**: Uses `conv2d` with a 3x3 kernel of ones (minus the center cell).
- **Differentiable Update Rule**:
    - Standard GoL: Birth on 3 neighbors, Survive on 2 or 3 neighbors.
    - Approximation: $\exp(-(N - 3)^2)$ for "Birth" and $\exp(-(N - 2)^2)$ for "Survive".
- **Optimization**: Gradient Descent on the initial state logits to minimize MSE between evolved state and target.

## Usage

```bash
tl run examples/tasks/tensor_logic/inverse_life/inverse_life.tl
```

## Notes

- The solver uses a soft approximation of GoL rules, which enables gradient flow but may not perfectly match the discrete dynamics.
- A custom Metal-compatible `sigmoid` implementation was added to the runtime (`1 / (1 + exp(-x))`).
