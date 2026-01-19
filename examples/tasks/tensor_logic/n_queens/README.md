# Differentiable N-Queens Solver

This example demonstrates how to solve the N-Queens problem using **Tensor Logic** and differentiable programming. Instead of using backtracking, we treat the board as a probabilistic distribution and minimize a loss function that penalizes invalid configurations.

## Problem Description
Place $N$ queens on an $N \times N$ board such that no two queens attack each other.
Queens attack horizontally, vertically, and diagonally.

## Method
1.  **Board Representation**: The board is represented as an $N \times N$ tensor of learnable logits.
2.  **Soft Constraint (Softmax)**: We apply `softmax` row-wise to enforce a conceptual "One Queen Per Row" constraint. This transforms logits into probabilities where each row sums to 1.
3.  **Loss Function**:
    *   **Column Loss**: Penalizes if the sum of probabilities in any column deviates from 1.0.
    *   **Diagonal Loss**: Penalizes if the sum of probabilities along any diagonal exceeds 1.0 (using `relu(sum - 1.0)^2`).
4.  **Optimization**: We perform gradient descent to minimize the total loss.

## Code Highlights
*   **Tensor Comprehensions**: Used to efficiently extract diagonal sums without manual indexing loops.
    ```rust
    let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
    ```
*   **Automatic Differentiation**: The solver automatically computes gradients for the continuous relaxation of the discrete problem.

## Usage
Run the solver:
```bash
cargo run --release --bin tl -- examples/tasks/n_queens/n_queens.tl
```
