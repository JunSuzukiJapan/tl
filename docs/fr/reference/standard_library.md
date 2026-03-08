# TensorLogic Standard Library Overview

This document provides an overview of the TL language standard library.
For detailed API reference, see [functions_and_types.md](functions_and_types.md).

---

## 1. Global Functions

### IO & System
- **`print(value, ...) -> void`** — Print without newline (`{}` formatting supported)
- **`println(value, ...) -> void`** — Print with newline (`{}` formatting supported)
- **`read_line(prompt: String) -> String`** — Read a line from stdin
- **`panic(message: String) -> Never`** — Print error message and terminate

### Command-Line Arguments
- **`args_count() -> i64`** — Number of command-line arguments
- **`args_get(index: i64) -> String`** — Get a command-line argument

---

## 2. Built-in Generic Types

### Vec\<T\> — Dynamic Array
```rust
let mut v: Vec<i64> = Vec::new();
v.push(1);
v.push(2);
let first = v.get(0).unwrap(); // 1
let popped = v.pop();          // Option::Some(2)
```

| Method | Description |
|---|---|
| `Vec::new() -> Vec<T>` | Create empty Vec |
| `Vec::with_capacity(cap) -> Vec<T>` | Create Vec with capacity |
| `len() -> i64` | Number of elements |
| `capacity() -> i64` | Capacity |
| `is_empty() -> bool` | Whether empty |
| `push(item: T)` | Append to end |
| `pop() -> Option<T>` | Remove from end |
| `get(index) -> Option<T>` | Get by index |
| `set(index, item: T)` | Update by index |

### HashMap\<K, V\> — Hash Map
```rust
let mut m: HashMap<String, i64> = HashMap::new();
m.insert("key", 42);
let val = m.get("key").unwrap(); // 42
```

| Method | Description |
|---|---|
| `HashMap::new() -> HashMap<K, V>` | Create empty HashMap |
| `len() -> i64` | Number of entries |
| `is_empty() -> bool` | Whether empty |
| `insert(key, value)` | Insert |
| `get(key) -> Option<V>` | Look up |

### Option\<T\> — Optional Value
```rust
let some_val: Option<i64> = Option::Some(42);
let none_val: Option<i64> = Option::None;

match some_val {
    Option::Some(x) => println("got: {}", x),
    Option::None => println("none"),
}
```

| Method | Description |
|---|---|
| `is_some() -> bool` | Whether `Some` |
| `is_none() -> bool` | Whether `None` |
| `unwrap() -> T` | Extract value (panics on `None`) |
| `unwrap_or(default: T) -> T` | Extract with default |

### Result\<T, E\> — Success/Failure
```rust
fn divide(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Result::Err("division by zero");
    }
    Result::Ok(a / b)
}
```

| Method | Description |
|---|---|
| `is_ok() -> bool` | Whether `Ok` |
| `is_err() -> bool` | Whether `Err` |
| `unwrap() -> T` | Extract value (panics on `Err`) |
| `unwrap_err() -> E` | Extract error |

`?` operator: Can be used with `Result` types; returns early on `Err`.

---

## 3. Standard Traits

| Trait | Methods | Description |
|---|---|---|
| `Index<Idx, Output>` | `index(self, idx) -> Output` | `[]` read access |
| `IndexMut<Idx, Value>` | `set(self, idx, value)` | `[]` write access |
| `Iterable<T>` | `len(self) -> i64`, `get(self, index) -> T` | Iterator for `for` loops |

---

## 4. Standard Classes (Static Methods)

### Tensor
- `Tensor::zeros(shape, requires_grad) -> Tensor`
- `Tensor::randn(shape, requires_grad) -> Tensor`
- `Tensor::ones(shape, requires_grad) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Param (Parameter Management)
- `Param::save_all(path) -> void` — Save all parameters
- `Param::load_all(path) -> void` — Load all parameters
- `Param::register(t: Tensor) -> Tensor` — Register tensor as parameter
- `Param::update_all(lr: f32) -> void` — Update all parameters
- `Param::register_modules(root: Struct) -> void` — Register modules
- `Param::checkpoint(method, input) -> Tensor` — Activation checkpoint
- `Param::set_device(device) -> void` — Set compute device

### File
- `File::open(path, mode) -> File`
- `File::exists(path) -> bool`
- `File::read(path) -> String`
- `File::write(path, content) -> bool`
- `File::download(url, dest) -> bool`

### Path
- `Path::new(path) -> Path`

### System
- `System::time() -> f32`
- `System::sleep(seconds) -> void`
- `System::memory_mb() -> i64`

### Env
- `Env::get(key) -> String`
- `Env::set(key, value) -> void`

### Http
- `Http::get(url) -> String`
- `Http::download(url, dest) -> bool`

### Image
- `Image::load_grayscale(path) -> Tensor`

---

## 5. Tensor Instance Methods (Excerpt)

#### Math (Element-wise)
`abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
`sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`, `pow(exp)`

#### Shape & Operations
- `reshape(...dims)`, `transpose(dim1, dim2)`, `slice(start, len)`
- `len()`, `dim()`, `item()`, `item_i64()`

#### Reductions & Statistics
`sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
`softmax(dim)`, `log_softmax(dim)`

#### Linear Algebra
- `matmul(other)`, `tril(diagonal)`, `embedding(weights)`
- `cross_entropy(targets)`, `conv2d(weight, padding, stride)`
- `rms_norm(weight, eps)`, `apply_rope(cos, sin, dim)`

#### Autograd
- `backward()`, `grad()`, `enable_grad()`, `detach()`, `clone()`

#### Device
- `cuda()`, `cpu()`

---

## 6. Scalar Type Methods

### String
- `len()`, `contains(other)`, `concat(other)`, `char_at(index)`, `to_i64()`

### F32 / F64
Math functions: `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round`, etc. (31 functions total)

### I64 / I32
- `abs()`, `signum()`, `is_positive()`, `is_negative()`
- `div_euclid(x)`, `rem_euclid(x)`, `pow(x)`
