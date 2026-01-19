# Differentiable Sudoku Solver

This example demonstrates solving Sudoku puzzles using differentiable programming and gradient descent, similar to the N-Queens solver approach.

## Problem Description
Given a 9×9 Sudoku puzzle with some cells pre-filled, find a solution where:
- Each row contains digits 1-9 exactly once
- Each column contains digits 1-9 exactly once
- Each 3×3 box contains digits 1-9 exactly once

## Method
1. **Board Representation**: The board is represented as a 9×9×9 tensor of learnable logits (one-hot encoding for 9 possible digits per cell).
2. **Soft Constraint (Softmax)**: Apply softmax along the digit dimension to get probability distributions.
3. **Loss Function**:
   - **Row Loss**: Penalizes if digit probabilities in any row don't sum to 1
   - **Column Loss**: Penalizes if digit probabilities in any column don't sum to 1
   - **Cell Loss**: Penalizes if cell probabilities don't represent a single digit
4. **Optimization**: Gradient descent minimizes the total loss.

## Command Line Usage

The puzzle is passed as a command line argument (81-character string, 0 = empty cell):

```bash
cargo run --release -- examples/tasks/tensor_logic/sudoku/sudoku.tl -- "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

If no argument is provided, a default puzzle is used.

## Code Highlights

- **Command Line Arguments**: Uses `args_count()` and `args_get(index)` functions
- **Tensor Comprehensions**: Uses softmax and sum operations for constraint formulation
- **Automatic Differentiation**: Gradients computed automatically via `.backward()`

## Note

This is a demonstration of differentiable constraint solving. The current implementation may require parameter tuning for convergence on harder puzzles.
