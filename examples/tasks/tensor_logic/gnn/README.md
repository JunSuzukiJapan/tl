# Graph Neural Network (GNN) for Shortest Path

This example demonstrates how to implement a Graph Neural Network (GNN) in Tensor Logic (`tl`) to solve a shortest path problem using Unsupervised Learning (Neural Bellman-Ford).

## Overview

The program performs the following:
1.  **Graph Generation**: Creates a random adjacency matrix `A` for `N` nodes.
2.  **Model**: Initializes a learnable distance tensor `val` for each node.
3.  **Unsupervised Training**: Optimizes `val` to satisfy the Bellman-Ford equation:
    `d[v] = min(d[u] + w(u, v))`
    It uses a "SoftMin" approximation for differentiability and backpropagation.
4.  **Constraints**: Enforces `val[source] = 0` and `val[v] >= 0`.

## Implementation Details

-   **Message Passing**: Uses tensor broadcasting and operations (transpose, expand) to efficienty compute pairwise costs `val[u] + weight`.
-   **Sparse logic**: Masks non-existent edges with large values to ignore them in SoftMin.
-   **Loss Function**: A combination of consistency loss (Bellman-Ford error), boundary condition loss, and positivity regularization.

## Usage

Run the example using the `tl` compiler:

```bash
cargo run --release -- run examples/tasks/tensor_logic/gnn/gnn.tl
```

## Key Learnings

-   **Tensor Comprehensions**: Used for generating the adjacency matrix logic.
-   **Vectorization**: The core message passing is fully vectorized without explicit node loops, leveraging Tensor Logic's broadcasting capabilities.
