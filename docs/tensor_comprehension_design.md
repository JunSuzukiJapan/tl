## Design Goals of Tensor Comprehensions

The original motivation for designing tensor comprehensions was to express **tensor contraction** concisely.
The goal was to contract tensors with multiple dimensions along specific indices.

### Motivation: Matrix Multiplication Example

Consider matrix multiplication $C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$.

Traditional loop-based representation:
```python
for i in range(M):
    for k in range(P):
        for j in range(N):
            C[i, k] += A[i, j] * B[j, k]
```

What we essentially want to express is:
- **Indices to keep**: `i`, `k` (dimensions of the output tensor)
- **Indices to contract**: `j` (summed over and eliminated)

### The Tensor Comprehension Concept

We designed a notation to directly specify which indices to keep:

```
[ i, k | { A[i, j] * B[j, k] } ]
```

In this notation:
- `[ i, k |` : The output tensor has dimensions `i` and `k`
- `A[i, j] * B[j, k]` : Product of tensors
- **Implicit contraction**: `j` doesn't appear in the output, so it's automatically summed over

The core idea is: "declare only the indices to keep, and the rest are automatically contracted."

### Relationship with Einstein Summation Convention

This is the same concept as Einstein's summation convention:

| Operation | Einstein Notation | Tensor Comprehension |
|-----------|-------------------|---------------------|
| Matrix Multiply | $C_{ik} = A_{ij}B_{jk}$ | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Trace | $\text{tr}(A) = A_{ii}$ | `[ \| { A[i,i] } ]` (scalar) |
| Outer Product | $C_{ij} = a_i b_j$ | `[ i, j \| { a[i] * b[j] } ]` |

### Current Implementation

In the current TL language, **generators can be omitted** and index ranges are **automatically inferred from tensor shapes**:

```rust
let A = [i | i <- 0..5 { i }];  // Explicit generator

let C = [i | { A[i] + 1 }];     // Implicit range inference (from A's shape)
```

In the `C` example above, the range of `i` is automatically inferred from the access pattern of tensor `A`.
This achieves the original vision of "declare only the indices to keep."

---

## Influence from Haskell List Comprehensions

### Origin of the Syntax

The syntax of tensor comprehensions is influenced by **Haskell's list comprehensions**:

```haskell
-- Haskell: List comprehension
[ x * 2 | x <- [1..5] ]          -- [2, 4, 6, 8, 10]
[ (x, y) | x <- [1..3], y <- [1..3], x /= y ]  -- List of pairs
```

```rust
// TL: Tensor comprehension
[ i | i <- 0..5 { i * 2 } ]      // Tensor [0, 2, 4, 6, 8]
```

Both share the common structure `[ output | generator, filter ]`.

### Introduction of Generators

Haskell's list comprehensions have the ability to **generate lists**.
From this insight, we thought "if we extend tensor comprehensions, we could generate tensors too," and **generators** were implemented.

The evolution of tensor comprehensions followed this path:

1. **Initial concept**: Contraction of existing tensors (`[ i, k | { A[i,j] * B[j,k] } ]`)
2. **Influence from Haskell**: Tensor generation via generators (`[ i | i <- 0..N { expr } ]`)
3. **Current form**: Supports both use cases

Through this evolution, tensor comprehensions grew from a simple contraction notation into **a unified expression for both tensor generation and transformation**.
