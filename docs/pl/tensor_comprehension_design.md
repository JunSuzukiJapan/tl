# Design Goals of Tensor Comprehensions

## What We Wanted to Achieve with Tensor Comprehensions

The original motivation for tensor comprehensions was the desire to perform tensor contractions. We wanted to contract tensors with certain dimensions along specific variables.

### Motivation: Matrix Multiplication Example

Consider matrix multiplication $C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$.

Traditional loop-based representation:
```python
for i in range(M):
    for k in range(P):
        for j in range(N):
            C[i, k] += A[i, j] * B[j, k]
```

What we fundamentally want to know is:
- **Indices to keep**: `i`, `k` (dimensions of the output tensor)
- **Indices to contract**: `j` (dimension summed over and eliminated)

### The Tensor Comprehension Idea

So we devised a way to directly describe "which indices to keep":

```
[ i, k | { A[i, j] * B[j, k] } ]
```

In this notation:
- `[ i, k |` : The output tensor has dimensions `i` and `k`
- `A[i, j] * B[j, k]` : Product of tensors
- **Implicit contraction**: `j` doesn't appear in the output, so it is automatically summed over

In other words, the original intent was that by "declaring only the indices to keep," the rest would be automatically contracted.

### Relationship to Einstein Summation Convention

This is the same idea as Einstein's summation convention:

| Operation | Einstein Notation | Tensor Comprehension |
|------|---------------------|------------------|
| Matrix multiplication | $C_{ik} = A_{ij}B_{jk}$ | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Trace | $\text{tr}(A) = A_{ii}$ | `[ \| { A[i,i] } ]` (scalar) |
| Outer product | $C_{ij} = a_i b_j$ | `[ i, j \| { a[i] * b[j] } ]` |

### Current Implementation

In the current TL language, you can **omit generators** and have index ranges **automatically inferred from tensor shapes**:

```rust
let A = [i | i <- 0..5 { i }];  // Explicit generator

let C = [i | { A[i] + 1 }];     // Implicit range inference (from A's shape)
```

In the `C` example above, the range of `i` is automatically inferred from tensor `A`'s access pattern.
This realizes the original ideal of "just declare the indices to keep" in a simple notation.

---

## Influence from Haskell List Comprehensions

### Origin of the Notation

The syntax of tensor comprehensions was influenced by **Haskell's list comprehensions**:

```haskell
-- Haskell: List comprehension
[ x * 2 | x <- [1..5] ]          -- [2, 4, 6, 8, 10]
[ (x, y) | x <- [1..3], y <- [1..3], x /= y ]  -- List of pairs
```

```rust
// TL: Tensor comprehension
[ i | i <- 0..5 { i * 2 } ]      // Tensor [0, 2, 4, 6, 8]
```

Both share the common structure `[ output | generators, filters ]`.

### Introduction of Generators

Haskell's list comprehensions have the ability to **generate lists**.
From this idea, we thought "if we extend tensor comprehensions, we can also generate tensors," and **generators** were implemented.

In summary, the evolution of tensor comprehensions followed this path:

1. **Original idea**: Contraction of existing tensors (`[ i, k | { A[i,j] * B[j,k] } ]`)
2. **Haskell influence**: Tensor generation via generators (`[ i | i <- 0..N { expr } ]`)
3. **Current form**: Supports both use cases

This transformed tensor comprehensions from a mere contraction notation into **a unified expression for tensor generation and transformation**.
