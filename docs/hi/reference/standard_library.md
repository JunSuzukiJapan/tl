# TensorLogic Standard Library Overview

This document provides an overview of the TL language standard library.
For detailed API reference, see [functions_and_types.md](functions_and_types.md).

---

## 1. Global Functions

### IO & System
- **`print(value, ...) -> void`** — Print without newline (`{}` format supported)
- **`println(value, ...) -> void`** — Print with newline (`{}` format supported)
- **`read_line(prompt: String) -> String`** — Read one line from stdin
- **`panic(message: String) -> Never`** — Print error and terminate

### Command-Line Arguments
- **`args_count() -> i64`** — Number of command-line arguments
- **`args_get(index: i64) -> String`** — Get command-line argument

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
| `len() -> i64` | Element count |
| `capacity() -> i64` | Capacity |
| `is_empty() -> bool` | Is empty |
| `push(item: T)` | Append to end |
| `pop() -> Option<T>` | Remove from end |
| `get(index) -> Option<T>` | Get by index |
| `set(index, item: T)` | Set by index |
| `map(f: Fn(T) -> U) -> Vec<U>` | Apply function to each element |
| `filter(f: Fn(T) -> bool) -> Vec<T>` | Keep elements matching condition |

### HashMap\<K, V\> — Hash Map
```rust
let mut m: HashMap<String, i64> = HashMap::new();
m.insert("key", 42);
let val = m.get("key").unwrap(); // 42
```

| Method | Description |
|---|---|
| `HashMap::new() -> HashMap<K, V>` | Create empty HashMap |
| `len() -> i64` | Entry count |
| `is_empty() -> bool` | Is empty |
| `insert(key, value)` | Insert |
| `get(key) -> Option<V>` | Lookup |
| `remove(key)` | Remove |
| `contains_key(key) -> bool` | Check key existence |
| `keys() -> Vec<K>` | List of keys |
| `values() -> Vec<V>` | List of values |

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
| `is_some() -> bool` | Is `Some` |
| `is_none() -> bool` | Is `None` |
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
| `is_ok() -> bool` | Is `Ok` |
| `is_err() -> bool` | Is `Err` |
| `unwrap() -> T` | Extract value (panics on `Err`) |
| `unwrap_err() -> E` | Extract error |

`?` operator: Can be used with `Result` for early return on `Err`.

---

## 3. Standard Traits

| Trait | Methods | Description |
|---|---|---|
| `Index<Idx, Output>` | `index(self, idx) -> Output` | `[]` read access |
| `IndexMut<Idx, Value>` | `set(self, idx, value)` | `[]` write access |
| `Iterable<T>` | `len(self) -> i64`, `get(self, index) -> T` | Iterator for `for` loops |

---

## 4. Closures (Anonymous Functions)

TL supports Rust-style closures.

```rust
// Single expression
let double = |x: i64| x * 2;

// Block body
let add = |x: i64, y: i64| -> i64 { x + y };

// Variable capture
let factor = 3;
let mul = |x: i64| -> i64 { x * factor };

// Passing to higher-order functions
let nums = Vec::new();
nums.push(1); nums.push(2); nums.push(3);
let doubled = nums.map(|x: i64| -> i64 { x * 2 });
```

Type notation: `Fn(i64, i64) -> i64`

---

## 5. Standard Classes (Static Methods)

### Tensor
- `Tensor::zeros(shape, requires_grad) -> Tensor`
- `Tensor::randn(shape, requires_grad) -> Tensor`
- `Tensor::ones(shape, requires_grad) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Param (Parameter Management)
- `Param::save_all(path)`, `Param::load_all(path)`
- `Param::register(t) -> Tensor`, `Param::update_all(lr)`
- `Param::register_modules(root)`, `Param::checkpoint(method, input)`
- `Param::set_device(device)`

### File
- `File::open(path, mode)`, `File::exists(path)`, `File::read(path)`, `File::write(path, content)`, `File::download(url, dest)`

### Path, System, Env, Http, Image
See [functions_and_types.md](functions_and_types.md) for full details.

---

## 6. Tensor Instance Methods (Excerpt)

#### Math (Element-wise)
`abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
`sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`, `pow(exp)`,
`leaky_relu(slope)`, `elu(alpha)`, `mish()`, `hardswish()`, `hardsigmoid()`

#### Shape & Operations
- `reshape(shape)`, `view(shape)`, `transpose(d1, d2)`, `permute(dims)`
- `flatten()`, `squeeze(dim)`, `unsqueeze(dim)`, `contiguous()`
- `slice(start, len)`, `narrow(dim, start, len)`, `chunk(n, dim)`, `split(size, dim, idx)`
- `cat(tensors, dim)`, `broadcast_to(shape)`, `expand(shape)`

#### Reductions & Statistics
`sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
`softmax(dim)`, `log_softmax(dim)`, `prod()`, `var()`, `std()`, `cumsum(dim)`, `norm(p)`, `topk(k, dim)`

#### Neural Networks
- `matmul(other)`, `linear(weight, bias?)`
- `conv1d(w, b?, stride, padding)`, `conv2d(w, padding, stride)`, `conv_transpose2d(...)`
- `max_pool2d(k, s)`, `avg_pool2d(k, s)`, `adaptive_avg_pool2d(oh, ow)`
- `interpolate(oh, ow, mode)`, `pad(left, right, value)`
- `dropout(p, training)`, `dropout2d(p, training)`
- `batch_norm(...)`, `layer_norm(...)`, `group_norm(...)`, `instance_norm(...)`, `rms_norm(...)`
- `embedding(weights)`, `cross_entropy(targets)`, `dot(other)`, `fill_(value)`

#### Linear Algebra
- `inverse()`, `det()`, `solve(b)`, `svd_u()`, `svd_s()`, `svd_v()`, `eig_values()`, `eig_vectors()`

#### Autograd
- `backward()`, `grad()`, `enable_grad()`, `detach()`, `clone()`
- `freeze()`, `unfreeze()`, `clip_grad_norm(max)`, `clip_grad_value(val)`

#### Device
- `cuda()`, `cpu()`, `to(device)`

---

## 7. Scalar Type Methods

### String
- `len()`, `is_empty()`, `contains(s)`, `starts_with(s)`, `ends_with(s)`
- `split(sep)`, `trim()`, `replace(from, to)`, `substring(start, len)`
- `concat(other)`, `index_of(s)`, `char_at(i)`
- `to_uppercase()`, `to_lowercase()`
- `to_i64()`, `to_f32()`, `to_f64()`

### F32 / F64
Math: `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round`, etc. (31 functions)
Conversion: `to_i64()`, `to_f32()`, `to_f64()`, `to_string()`
Other: `min(x)`, `max(x)`, `clamp(min, max)`

### I64 / I32
- `abs()`, `signum()`, `is_positive()`, `is_negative()`
- `div_euclid(x)`, `rem_euclid(x)`, `pow(x)`
- `min(x)`, `max(x)`, `clamp(min, max)`
- `to_f32()`, `to_f64()`, `to_string()`
