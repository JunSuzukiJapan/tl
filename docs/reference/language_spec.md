# TensorLogic Language Specification

TensorLogic (TL) is a statically-typed programming language designed for high-performance tensor operations and logic programming, JIT-compiled to LLVM.

## 1. Comments

```rust
// This is a single-line comment
/* This is a 
   multi-line comment */
```

## 2. Data Types

### Primitive Types
*   **`f32`**: 32-bit floating point number.
*   **`f64`**: 64-bit floating point number.
*   **`i64`**: 64-bit signed integer.
*   **`bool`**: Boolean value (`true` or `false`).
*   **`String`**: UTF-8 string.

### Tensor
The core data structure. Tensors are multidimensional arrays.
```rust
let t: Tensor = Tensor::zeros([2, 2], true);
```

### Structs
User-defined compound types.
```rust
struct Point {
    x: f32,
    y: f32,
}
```

### Enums
Tagged unions for defining variants.
```rust
enum Option {
    Some { value: i64 },
    None,
}
```

## 3. Variables

Variables are immutable by default in the sense of binding, but objects like Tensors can be mutated if methods allow. Shadowing is permitted.

```rust
let x = 10;
let x = x + 1; // Shadowing
```

## 4. Functions

Functions are first-class residents.

```rust
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

## 5. Control Flow

### If Expression
```rust
let result = if x > 0 { 1 } else { 0 };
```

### While Loop
```rust
while i < 10 {
    print(i);
    i = i + 1;
}
```

### For Loop
Ranges usage:
```rust
for i in 0..10 {
    print(i);
}
```

## 6. Tensor Comprehension

A powerful syntax for creating tensors.

```rust
// Syntax: [indices | clauses { body }]
let A = [i, j | i <- 0..5, j <- 0..5 { i + j }];
```

## 7. Classes & Methods

Methods can be attached to Structs or Enums via `impl`.

```rust
impl Point {
    fn distance(self) -> f32 {
        (self.x.pow(2.0) + self.y.pow(2.0)).sqrt()
    }
}
```
