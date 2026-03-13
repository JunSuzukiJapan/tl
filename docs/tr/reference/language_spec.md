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
The core data structure representing multi-dimensional arrays. Types distinguish whether gradients are tracked.
```rust
let t: Tensor = Tensor::zeros([2, 2], false);       // No gradient (inference)
let g: GradTensor = Tensor::zeros([2, 2], true);     // With gradient (training)
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

### Fn Type (Function / Closure Type)
Represents the type of functions and closures.
```rust
fn apply(f: Fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}
```

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

## 5. Closures (Anonymous Functions)

TL supports Rust-style closures that can capture variables from the enclosing scope.

### Basic Syntax

```rust
// No arguments
let greet = || println("Hello!");
greet();

// Single expression closure
let double = |x: i64| x * 2;
println("{}", double(5)); // 10

// Block body
let complex = |x: i64, y: i64| -> i64 {
    let sum = x + y;
    sum * 2
};
```

### Passing to Higher-Order Functions

```rust
let numbers = Vec::new();
numbers.push(1);
numbers.push(2);
numbers.push(3);

// map: apply a function to each element
let doubled = numbers.map(|x: i64| -> i64 { x * 2 });

// filter: keep only elements satisfying a condition
let evens = numbers.filter(|x: i64| -> bool { x % 2 == 0 });
```

### Variable Capture

```rust
let factor = 3;
let multiply = |x: i64| -> i64 { x * factor };  // captures factor
println("{}", multiply(5)); // 15
```

### Type Annotations

Argument type annotations are optional (resolved by type inference). Return types can also be omitted.

```rust
// With type annotations
let add = |x: i64, y: i64| -> i64 { x + y };

// Without type annotations (type inference)
let add = |x, y| x + y;
```

## 6. Control Flow

### If Expression
```rust
let result = if x > 0 { 1 } else { 0 };
```

### While Loop
```rust
while i < 10 {
    i += 1;
}
```

### For Loop
Can be used with any type supporting the iterator protocol (`len` + `get` methods).
```rust
// Range
for i in 0..10 { print(i); }

// Half-open range (start..)
for i in 0..5 { print(i); }

// Vec
let v = Vec::new();
v.push(1); v.push(2);
for item in v { print(item); }

// Tensor
let t = Tensor::zeros([5], false);
for val in t { print(val); }
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

### Match Expression
```rust
match value {
    Option::Some(x) => println("got: {}", x),
    Option::None => println("none"),
}
```

### If Let Expression
```rust
if let Option::Some(x) = maybe_value {
    println("value is: {}", x);
} else {
    println("no value");
}
```

## 7. Operators

### Arithmetic Operators
`+`, `-`, `*`, `/`, `%`

### Comparison Operators
`==`, `!=`, `<`, `>`, `<=`, `>=`

### Logical Operators
`&&`, `||`, `!`

### Bitwise Operators
`&` (AND), `|` (OR), `^` (XOR)

### Range Operator
`..` — Range (`0..10`, `0..`, `..10`)

### Type Cast
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

## 8. Structs and impl Blocks

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

## 9. Traits

Traits provide an interface mechanism similar to Rust. Default methods are also supported.
```rust
trait Display {
    fn display(self) -> String;
}

trait Greetable {
    fn name(self) -> String;
    // Default method
    fn greet(self) -> String {
        let n = self.name();
        n.concat(", hello!")
    }
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

## 10. Tensor Comprehensions

```rust
// Syntax: [indices | clauses { body }]
let A = [i, j | i <- 0..5, j <- 0..5 { i + j }];
```

Implicit reduction analysis automatically detects indices not present in the LHS as reduction indices (Einstein summation convention).

## 11. Logic Programming

TL integrates Datalog-style logic programming.

```rust
// Relation definition
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

## 12. Module System

```rust
use math::{sin, cos};
use utils::*;           // Glob import
use parser as p;        // Alias
```
