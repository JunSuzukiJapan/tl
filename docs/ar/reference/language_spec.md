# TensorLogic Language Specification

TensorLogic (TL) is a statically typed programming language designed for high-performance tensor computation and logic programming, JIT-compiled to LLVM. It features Rust-like syntax.

## 1. Comments

```rust
// This is a single-line comment
/* This is a
   multi-line comment */
```

## 2. Data Types

### Primitive Types
| Type | Description |
|---|---|
| `f32` | 32-bit floating point |
| `f64` | 64-bit floating point |
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `usize` | Pointer-sized unsigned integer |
| `bool` | Boolean (`true` / `false`) |
| `String` | UTF-8 string |
| `Char` | Single character (`'a'`) |

### Tensor
The core data structure representing multi-dimensional arrays.
```rust
let t: Tensor = Tensor::zeros([2, 2], true);
```

### Structs
User-defined composite types.
```rust
struct Point {
    x: f32,
    y: f32,
}
```

### Enums
Tagged unions. Supports Unit, Tuple, and Struct variants.
```rust
enum Shape {
    Circle(f32),           // Tuple variant
    Rectangle(f32, f32),   // Tuple variant
    Point,                 // Unit variant
}
```

### Tuple Types
```rust
let pair: (i64, f32) = (42, 3.14);
let x = pair.0;  // Tuple access
```

### Fixed-Size Arrays
```rust
let arr: [i64; 3] = [1, 2, 3];
```

### Generic Types

TL supports Rust-style generics.

```rust
struct Pair<A, B> {
    first: A,
    second: B,
}
```

Built-in generic types:
- `Vec<T>` — Dynamic array
- `HashMap<K, V>` — Hash map
- `Option<T>` — `Some(T)` or `None`
- `Result<T, E>` — `Ok(T)` or `Err(E)`

## 3. Variables

### Immutable Variables (Default)
```rust
let x = 10;
let x = x + 1; // Shadowing is allowed
```

### Mutable Variables
```rust
let mut count = 0;
count = count + 1;  // Reassignment is allowed
```

### Compound Assignment Operators
```rust
count += 1;
count -= 1;
count *= 2;
count /= 2;
count %= 3;
```

## 4. Functions

```rust
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

### Generic Functions
```rust
fn identity<T>(x: T) -> T {
    x
}
```

### Public Visibility
```rust
pub fn public_function() { }
pub struct PublicStruct { pub field: i64 }
```

## 5. Control Flow

### If Expressions
```rust
let result = if x > 0 { 1 } else { 0 };
```

### While Loops
```rust
while i < 10 {
    i += 1;
}
```

### For Loops
Can be used with any type that supports the iterator protocol (`len` + `get` methods).
```rust
// Range
for i in 0..10 { print(i); }

// Vec
let v = Vec::new();
v.push(1); v.push(2);
for item in v { print(item); }
```

### Loop (Infinite Loop)
```rust
loop {
    if done { break; }
}
```

### Loop Control
```rust
for i in 0..10 {
    if i == 5 { continue; }
    if i == 8 { break; }
}
```

### Match Expressions
```rust
match value {
    Option::Some(x) => println("got: {}", x),
    Option::None => println("none"),
}
```

### If Let Expressions
```rust
if let Option::Some(x) = maybe_value {
    println("value is: {}", x);
} else {
    println("no value");
}
```

## 6. Operators

### Arithmetic Operators
`+`, `-`, `*`, `/`, `%`

### Comparison Operators
`==`, `!=`, `<`, `>`, `<=`, `>=`

### Logical Operators
`&&`, `||`, `!`

### Bitwise Operators
`&` (AND), `|` (OR), `^` (XOR)

### Type Casting
```rust
let x = 42 as f32;
```

### Try Operator (`?`)
Can only be used with `Result` types. Returns early on `Err`.
```rust
fn read_file() -> Result<String, String> {
    let content = File::read("path.txt")?;
    Result::Ok(content)
}
```

## 7. Structs and impl Blocks

```rust
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        Point { x: x, y: y }
    }

    fn distance(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
```

## 8. Traits

Traits provide an interface mechanism similar to Rust.
```rust
trait Display {
    fn display(self) -> String;
}

impl Display for Point {
    fn display(self) -> String {
        println("{}, {}", self.x, self.y);
        ""
    }
}
```

### Standard Traits
- `Index<Idx, Output>` — `[]` read access
- `IndexMut<Idx, Value>` — `[]` write access
- `Iterable<T>` — Iterator protocol for `for` loops (`len` + `get`)

## 9. Tensor Comprehensions

```rust
// Syntax: [indices | clauses { body }]
let A = [i, j | i <- 0..5, j <- 0..5 { i + j }];
```

Implicit reduction analysis automatically detects indices not present in the LHS as reduction indices (Einstein summation convention).

## 10. Logic Programming

TL integrates Datalog-style logic programming.

```rust
// Relation declaration
relation parent(entity, entity);

// Fact definition
parent("Alice", "Bob");
parent("Bob", "Charlie");

// Rule definition
ancestor(X, Y) :- parent(X, Y);
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y);

// Query
?- ancestor("Alice", Who);
```

## 11. Module System

```rust
use math::{sin, cos};
use utils::*;           // Glob import
use parser as p;        // Alias
```
