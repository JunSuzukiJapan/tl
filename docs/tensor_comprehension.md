# Tensor Comprehension Guide

TL provides a powerful and expressive tensor comprehension syntax inspired by Haskell's list comprehensions. This allows for concise tensor creation, element-wise transformations, and reductions.

## Syntax Overview

The general syntax is:

```rust
[ <indices> | <clauses> { <body> } ]
```

*   **`<indices>`**: A comma-separated list of variables representing the dimensions of the output tensor.
*   **`<clauses>`**: A comma-separated list of *generators* or *conditions*.
*   **`{ <body> }`**: A block expression that computes the value for each element. If omitted, it may be inferred or default to 0.0.

---

## 1. Generators

Generators define the range of values for the index variables.

### Explicit Generators
You can explicitly define the range for an index using `<-`.

```rust
// Create a 1D tensor of size 5: [0, 1, 2, 3, 4]
let A = [i | i <- 0..5 { i }];
```

### Implicit Inference
If a generator is omitted for an index used in the output `<indices>`, TL attempts to infer the range from tensors used in the `<body>`.

```rust
// 'i' is inferred from the shape of A
let B = [i | { A[i] * 2 }];
```

---

## 2. Conditions (Filtering)

You can include boolean expressions in the `<clauses>` section to act as filters.

*   If the condition is **true**, the computation proceeds for that iteration.
*   If the condition is **false**, the iteration is skipped (value remains 0.0 or does not contribute to reduction).

```rust
// Create a filtered tensor (sparse-like behavior with 0.0)
// Only even indices will have a value; others will be 0.0.
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
// Result: [0, 0, 2, 0, 4]
```

Multiple conditions are combined with logical AND.

```rust
let C = [i | i <- 0..10, i > 2, i < 8 { i }];
```

---

## 3. Reductions

Variables defined in `<clauses>` but **not** present in the output `<indices>` are treated as **reduction variables**. They are summed over.

```rust
// Scalar product (Dot product)
// 'k' is a reduction variable because it is not in the output [].
// 'i' is a dummy index for a scalar output (size 1).
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

### Conditional Reduction
You can use conditions to sum only specific elements.

```rust
// Sum of all even numbers from 0 to 9
let SumEvens = [i | i <- 0..1, k <- 0..10, (k % 2) == 0 { k }];
// Result: [20] (0 + 2 + 4 + 6 + 8)
```

---

## 4. More Examples

### Matrix Multiplication
```rust
// C[i, j] = Sum_k ( A[i, k] * B[k, j] )
// 'k' corresponds to the shared dimension and is reduced.
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

### Masking / ReLU-like operation
```rust
// If val > 0 return val, else 0
let ReLU = [i | i <- 0..N, A[i] > 0 { A[i] }];
```

### Diagonal Extraction
```rust
// Extract diagonal elements where row index equals column index
let Diag = [i | i <- 0..N, j <- 0..N, i == j { Matrix[i, j] }];

---

## 5. Optional Body (Implicit Default)

If the **body block** `{ ... }` is omitted, the element value defaults to the **first index variable**.

```rust
// A = [0, 1, 2, 3, 4]
let A = [i | i <- 0..5];
```

For higher rank tensors, it uses the first index:
```rust
// T[i, j] = i
// [[0, 0], [1, 1], [2, 2]]
let T = [i, j | i <- 0..3, j <- 0..2];
```
```
