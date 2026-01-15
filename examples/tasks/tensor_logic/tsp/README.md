# Differentiable TSP Solver

This example demonstrates how to solve the Traveling Salesperson Problem (TSP) using a differentiable programming approach in Tensor Logic.

## Concept

The goal is to find the shortest possible route that visits each city exactly once and returns to the origin city.
We learn a permutation matrix `P` of shape `[N, N]`, where `P[i, j]` represents the probability that the `i`-th step of the tour visits city `j`.

## Implementation Details

- **Vectorized Distance Calculation**: Computes the pairwise squared distance matrix between all cities using broadcasting and matrix multiplication.
- **Permutation Learning**: Uses a `Softmax` over the city dimension to approximate a permutation matrix.
- **Loss Function**:
    - **Tour Length**: Minimizes the total squared distance of the path.
    - **Path Constraints**: Penalizes visiting the same city multiple times (column sums of `P` should be 1).
- **Shift Matrix**: A cyclic shift matrix `S` is constructed to compute the "next" position in the tour for loss calculation.

## Usage

```bash
tl run examples/tasks/tensor_logic/tsp/tsp.tl
```
