# Logic Programming in TensorLogic

This example demonstrates how to implement Prolog-like logic programming using TensorLogic's tensor operations.

## Concept: Matrix Logic

Instead of symbolic unification and backtracking (like Prolog), we use **Adjacency Matrices** and **Matrix Multiplication** to represent facts and infer relationships. This allows us to leverage GPU acceleration for large-scale logical inference.

## Example: Family Tree

We define a simple family tree:
- Alice -> Bob
- Bob -> Charlie
- Charlie -> Diana

And infer:
- **GrandParent relationship**: Calculated via `Parent x Parent` matrix multiplication.
- **Ancestor relationship**: Calculated via transitive closure (iterative matrix multiplication).

## Usage

```bash
cargo run --release --bin tl -- run examples/tasks/logic/family_tree/family_tree.tl
```
