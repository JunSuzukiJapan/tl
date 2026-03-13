# Tensor Comprehension Guide

The TL language provides a powerful and expressive tensor comprehension syntax, inspired by Haskell's list comprehensions. This allows you to concisely describe tensor creation, element-wise transformations, and reductions.

## Syntax Overview

The basic syntax is as follows:

```rust
[ <indices> | <clauses> { <body> } ]
```

*   **`<indices>`**: A comma-separated list of variables representing the dimensions of the output tensor.
*   **`<clauses>`**: A comma-separated list of generators (range specifications) or conditions (filters).
*   **`{ <body> }`**: A block expression that computes the value of each element. If omitted, it is inferred from context or defaults to 0.0.

---

## 1. Generators

Generators define the range of values that index variables can take.

### Explicit Range Specification
Use `<-` to explicitly specify the range of an index.

```rust
// Create a 1D tensor of size 5: [0, 1, 2, 3, 4]
let A = [i | i <- 0..5 { i }];
```

### Implicit Inference
If a generator for a variable used in the output `<indices>` is omitted, the system attempts to infer the range from the shapes of tensors used in `<body>`.

```rust
// The range of 'i' is inferred from A's shape
let B = [i | { A[i] * 2 }];
```

---

## 2. Conditions (Filtering)

By including boolean expressions in the `<clauses>` section, you can filter computations.

*   If the condition is **true**, the computation for that iteration is executed.
*   If the condition is **false**, the iteration is skipped (the value remains 0.0 or is not added to the reduction).

```rust
// Tensor with values only at even indices (others are 0.0)
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
// Result: [0, 0, 2, 0, 4]
```

Multiple conditions can be written, and they are treated as logical AND.

```rust
let C = [i | i <- 0..10, i > 2, i < 8 { i }];
```

---

## 3. Reductions

Variables that are defined in `<clauses>` but **not included** in the output `<indices>` are treated as **reduction variables**. The sum is computed over these variables. This is implemented as **implicit reduction analysis (Einstein summation convention)**, which analyzes the body expression to automatically detect reduction variables.

```rust
// Dot product
// 'k' is not in the output [], so it becomes a reduction variable.
// 'i' is a dummy index for scalar output (size 1).
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

### Conditional Reductions
You can use conditions to sum only specific elements.

```rust
// Sum of even numbers from 0 to 9
let SumEvens = [i | i <- 0..1, k <- 0..10, (k % 2) == 0 { k }];
// Result: [20] (0 + 2 + 4 + 6 + 8)
```

---

## 4. More Examples

### Matrix Multiplication
```rust
// C[i, j] = Sum_k ( A[i, k] * B[k, j] )
// 'k' corresponds to the shared dimension and is aggregated.
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

### Masking / ReLU-like Operations
```rust
// Keep value if positive, otherwise 0
let ReLU = [i | i <- 0..N, A[i] > 0 { A[i] }];
```

### Diagonal Extraction
```rust
// Extract elements where row index equals column index
let Diag = [i | i <- 0..N, j <- 0..N, i == j { Matrix[i, j] }];
```

---

## 5. Optional Body

When the **body block** `{ ... }` is omitted, the following **"smart implicit body"** is generated:

1.  **Single index:** The index variable itself is used as the value.
    ```rust
    // A = Tensor[[5], f32]: [0, 1, 2, 3, 4]
    let A = [i | i <- 0..5];
    ```

2.  **Multiple indices:** A **vector (coordinate)** composed of the index variables is generated. This increases the tensor's rank by one.
    ```rust
    // T = Tensor[[N, M, 2], f32]
    // At each point of the Shape(N, M) grid, a vector [i, j] is placed.
    let grid = [i, j | i <- 0..N, j <- 0..M];
    
    // Concrete example:
    // [i, j | i <- 0..2, j <- 0..2]
    // Result:
    // [
    //   [[0, 0], [0, 1]], 
    //   [[1, 0], [1, 1]]
    // ]
    // (Tensor of Shape(2, 2, 2))
    ```

This enables intuitive operations on vector fields like `grid - center`.
This behavior contrasts with Haskell's list comprehensions, which return a flat list `[(0,0), (0,1)...]` (Shape(N*M)). TL prioritizes preserving the multi-dimensional structure (topology) of tensors.
