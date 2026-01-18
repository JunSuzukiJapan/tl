# Shortest Path Search in TensorLogic

This example demonstrates how to solve graph optimization problems using **Tropical Matrix Logic** (min-plus algebra).

## Concept: Tropical Semiring

In standard matrix multiplication, we use the `(+, *)` semiring. Logic programming often uses the Boolean `(OR, AND)` semiring (as seen in the Family Tree example).

For **Shortest Path** problems, we use the **Tropical Semiring** `(min, +)`:
- **"Addition"** becomes `min`
- **"Multiplication"** becomes `+` (sum of weights)

A matrix multiplication in this semiring corresponds to finding the shortest path between nodes by combining edges.

## The Algorithm

We can compute the All-Pairs Shortest Path (similar to Floyd-Warshall) by repeated "multiplication" of the adjacency matrix $A$ over the tropical semiring.
Ideally, if weights are non-negative, the shortest path metrics converge after $N$ iterations (where $N$ is the number of nodes).

In TensorLogic:
```rust
// Tropical Matrix Multiplication
// C[i, j] = min_k ( A[i, k] + B[k, j] )
let C = [i, j, k | ... { A[i, k] + B[k, j] }].min(2);
```

## Usage

```bash
cargo run --release --bin tl -- run examples/tasks/logic/shortest_path/shortest_path.tl
```
