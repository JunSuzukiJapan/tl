# TensorLogic API Reference

This document lists the currently supported **global functions**, **types**, and their **methods**. These are based on the compiler/runtime implementation at the time of writing.

Notes:
- Signatures are written in TL notation.
- `Tensor<T, R>` means a tensor with element type `T` and rank `R` (rank may be dynamic).
- Many numeric and tensor methods are also available via operators (`+`, `-`, `*`, `/`, `%`).

---

## 1. Global Functions

### I/O
- `print(value, ...) -> void`  
  Prints without newline. When using `{}` formatting, the first argument must be a string literal.
- `println(value, ...) -> void`  
  Prints with newline. When using `{}` formatting, the first argument must be a string literal.
- `read_line(prompt: String) -> String`  
  Displays a prompt and reads a line from stdin.

### Args
- `args_count() -> i64`  
  Returns the number of command-line arguments.
- `args_get(index: i64) -> String`  
  Returns the command-line argument at the given index.

### System
- `panic(message: String) -> Never`  
  Prints an error message and terminates the program.

---

## 2. Standard Types (Static Methods)

### Tensor (Static)
- `Tensor::zeros(shape, requires_grad?: bool) -> Tensor`
- `Tensor::randn(shape, requires_grad?: bool) -> Tensor`
- `Tensor::ones(shape, requires_grad?: bool) -> Tensor`
- `Tensor::load(path: String) -> Tensor`
- `Tensor::from_vec_u8(data: Vec<u8>, shape: Vec<i64>) -> Tensor` — Creates a Tensor from a byte Vec
- `Tensor::clear_grads() -> void` — Clears the global gradient store

### Vec\<T\> (Static)
- `Vec<T>::new() -> Vec<T>` — Creates an empty Vec
- `Vec<T>::with_capacity(cap: i64) -> Vec<T>` — Creates a Vec with specified capacity

### HashMap\<K, V\> (Static)
- `HashMap<K, V>::new() -> HashMap<K, V>` — Creates an empty HashMap

### Option\<T\> (Enum)
- `Option::Some(value: T)` — Variant holding a value
- `Option::None` — Variant with no value

### Result\<T, E\> (Enum)
- `Result::Ok(value: T)` — Variant holding a success value
- `Result::Err(error: E)` — Variant holding an error value

### Param (Parameter Management)
- `Param::save_all(path: String, format?: String) -> void`
- `Param::load_all(path: String, format?: String) -> void`
- `Param::save(target, path: String) -> void`
- `Param::load(path: String) -> Tensor`
- `Param::load(target, path: String) -> void`
- `Param::add(name: String, t: Tensor) -> void`
- `Param::register(t: Tensor) -> Tensor`
- `Param::update_all(lr: f32) -> void`
- `Param::register_modules(root: Struct) -> void`
- `Param::checkpoint(method_ref, input) -> Tensor`
- `Param::set_device(device: Device) -> void`

### VarBuilder (Static)
- `VarBuilder::get(name: String, ...dims) -> Tensor`
- `VarBuilder::grad(t: Tensor) -> Tensor`

### File (Static)
- `File::open(path: String, mode: String) -> File`
- `File::exists(path: String) -> bool`
- `File::read(path: String) -> String`
- `File::write(path: String, content: String) -> bool`
- `File::download(url: String, dest: String) -> bool`

### Path (Static)
- `Path::new(path: String) -> Path`

### Tokenizer (Static)
- `Tokenizer::new(path: String) -> Tokenizer`

### KVCache (Static)
- `KVCache::new(max_len: i64) -> KVCache`

### Map (Static)
- `Map::load(path: String) -> Map`

### System (Static)
- `System::time() -> f32`
- `System::sleep(seconds: f32) -> void`
- `System::memory_mb() -> i64`
- `System::pool_count() -> i64`
- `System::refcount_count() -> i64`
- `System::scope_depth() -> i64`
- `System::metal_pool_bytes() -> i64`
- `System::metal_pool_mb() -> i64`
- `System::metal_pool_count() -> i64`
- `System::metal_sync() -> void`

### Env (Static)
- `Env::get(key: String) -> String`
- `Env::set(key: String, value: String) -> void`

### Http (Static)
- `Http::get(url: String) -> String`
- `Http::download(url: String, dest: String) -> bool`

### Image (Static)
- `Image::load_grayscale(path: String) -> Tensor`
- `Image::width() -> i64`
- `Image::height() -> i64`

---

## 3. Instance Methods

### Tensor (Instance)

#### Shape and Indexing
- `reshape(shape) -> Tensor`
- `narrow(dim: i64, start: i64, len: i64) -> Tensor`
- `slice(dim: i64, start: i64, len: i64, stride: i64) -> Tensor`
- `transpose(dim1: i64, dim2: i64) -> Tensor`
- `transpose_2d() -> Tensor`
- `squeeze(dim: i64) -> Tensor` — Removes a dimension of size 1
- `unsqueeze(dim: i64) -> Tensor` — Adds a dimension of size 1
- `flatten(dim: i64) -> Tensor` — Flattens from the given dimension
- `permute(dims: Vec<i64>) -> Tensor` — Reorders dimensions
- `contiguous() -> Tensor` — Ensures contiguous memory layout
- `cat(other: Tensor) -> Tensor` — Concatenates tensors
- `gather(indices: Tensor) -> Tensor` — Gathers elements along an axis
- `len() -> i64`
- `dim(d: i64) -> i64` — Size of a specific dimension
- `ndim() -> i64` — Number of dimensions (rank)
- `shape() -> Vec<i64>` — Returns shape as a Vec
- `get_shape() -> Vec<i64>` — Same as `shape()`
- `get(i64...) -> f32`
- `set(i64..., value: f32) -> void`

#### Reductions
- `sum(dim?) -> Tensor`
- `sum_dim(dim: i64, keepdim: bool) -> Tensor`
- `mean(dim?) -> Tensor`
- `max(dim?) -> Tensor`
- `min(dim?) -> Tensor`
- `argmax(dim: i64) -> Tensor`
- `argmin(dim: i64) -> Tensor`

#### Element-wise Operations / Activation Functions
- `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`
- `exp()`, `log()`, `sqrt()`, `abs()`, `neg()`
- `sin()`, `cos()`, `tan()`
- `pow(exp)`, `pow(exp_tensor)`  
- `clamp(min, max) -> Tensor`
- `scale(s: f32) -> Tensor`

#### Linear Algebra and Neural Network Operations
- `matmul(other) -> Tensor`
- `matmul_4d(other) -> Tensor`
- `add_4d(other) -> Tensor`
- `cat_4d(other) -> Tensor`
- `embedding(weights) -> Tensor`
- `cross_entropy(targets) -> Tensor`
- `conv2d(weight, padding: i64, stride: i64) -> Tensor`
- `tril(diagonal: i64) -> Tensor`
- `softmax(dim: i64) -> Tensor`
- `log_softmax(dim: i64) -> Tensor`
- `rms_norm(weight, eps: f32) -> Tensor`
- `apply_rope(cos, sin, dim: i64) -> Tensor`
- `repeat_interleave(dim: i64, repeats: i64) -> Tensor`
- `sample() -> i64`

#### Arithmetic (Method Form)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)`, `mod(other)`
- `add_assign(other)`, `sub_assign(other)`, `mul_assign(other)`, `div_assign(other)`, `mod_assign(other)`

#### Comparison (Operator Form)
These return a Tensor of `0.0` / `1.0` values:
- `==` (`eq`), `!=` (`neq`), `<` (`lt`), `<=` (`le`), `>` (`gt`), `>=` (`ge`)

#### Autograd
- `backward() -> void`
- `grad() -> Tensor`
- `detach(requires_grad?: bool) -> Tensor`
- `enable_grad() -> Tensor`
- `clone() -> Tensor`
- `shallow_clone() -> Tensor` — Clones without deep-copying data

#### Device
- `cuda() -> Tensor`
- `cpu() -> Tensor`
- `to(device: String) -> Tensor` — Move tensor to specified device

#### Debug / I/O
- `print()`, `display()`
- `item() -> f32`
- `item_i64() -> i64`
- `save(path: String) -> void` — Save tensor to file

#### Type Conversion
- `to_i64() -> Tensor<I64>` — Converts element type to i64

#### Regularization
- `dropout(p: f32, training: bool) -> Tensor` — Applies dropout

#### Sampling
- `sample(temperature: f32, top_p: f32) -> Tensor<I64>` — Samples from logits with temperature and top-p

#### Quantization / Other
- `matmul_quantized(weight) -> Tensor`
- `cat_i64(other, dim: i64) -> Tensor`
- `sumall() -> Tensor` — Reduces all elements to a scalar tensor

---

### Vec\<T\> (Instance)
- `len() -> i64` — Returns the number of elements
- `capacity() -> i64` — Returns the capacity
- `is_empty() -> bool` — Whether the vec is empty
- `push(item: T) -> void` — Appends an element to the end
- `pop() -> Option<T>` — Removes and returns the last element
- `get(index: i64) -> Option<T>` — Gets an element by index
- `set(index: i64, item: T) -> void` — Updates an element by index

---

### HashMap\<K, V\> (Instance)
- `len() -> i64` — Returns the number of entries
- `is_empty() -> bool` — Whether the map is empty
- `insert(key: K, value: V) -> void` — Inserts a key-value pair
- `get(key: K) -> Option<V>` — Looks up a value by key
- `remove(key: K) -> void` — Removes an entry by key (not yet implemented)

---

### Option\<T\> (Instance)
- `is_some() -> bool` — Whether it is `Some`
- `is_none() -> bool` — Whether it is `None`
- `unwrap() -> T` — Extracts the value (panics on `None`)
- `unwrap_or(default: T) -> T` — Extracts the value (returns default on `None`)

---

### Result\<T, E\> (Instance)
- `is_ok() -> bool` — Whether it is `Ok`
- `is_err() -> bool` — Whether it is `Err`
- `unwrap() -> T` — Extracts the value (panics on `Err`)
- `unwrap_err() -> E` — Extracts the error (panics on `Ok`)

`?` operator: Can be used with `Result` types; returns early on `Err`.

---

### Numeric Types (F32, F64, I32, I64)

#### F32 / F64
Unary operations:
`abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `cos`, `cosh`,
`exp`, `exp2`, `exp_m1`, `floor`, `fract`, `ln`, `ln_1p`, `log`, `log10`, `log2`, `recip`, `round`,
`signum`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `to_degrees`, `to_radians`, `trunc`

Binary operations:
`atan2(x)`, `copysign(x)`, `hypot(x)`, `powf(x)`, `pow(x)`, `powi(x)`

#### I64 / I32
- `abs() -> int`
- `signum() -> int`
- `is_positive() -> bool`
- `is_negative() -> bool`
- `div_euclid(x) -> int`
- `rem_euclid(x) -> int`
- `pow(x) -> int`

---

### String
- `len() -> i64` — String length
- `contains(other: String) -> bool` — Whether it contains a substring
- `concat(other: String) -> String` — String concatenation
- `char_at(index: i64) -> Char` — Character at the given position
- `to_i64() -> i64` — Parse as integer
- `print() -> void` — Print
- `display() -> void` — Display

---

### File (Instance)
- `read_string() -> String`
- `write_string(s: String) -> void`
- `read_to_end() -> Vec<u8>`
- `write(bytes: Vec<u8>) -> void`
- `close() -> void`
- `free() -> void`

---

### Path (Instance)
- `join(part: String) -> Path`
- `exists() -> bool`
- `is_dir() -> bool`
- `is_file() -> bool`
- `to_string() -> String`
- `free() -> void`

---

### Tokenizer (Instance)
- `encode(text: String) -> Tensor<I64, 1>`
- `decode(tokens: Tensor<I64, 1>) -> String`
- `token_id(token: String) -> i64`
- `vocab_size() -> i64`
- `free() -> void`

---

### KVCache (Instance)
- `get_k(layer: i64) -> Tensor<F32, 0>`
- `get_v(layer: i64) -> Tensor<F32, 0>`
- `update(layer: i64, k: Tensor, v: Tensor) -> void`
- `free() -> void`

---

### Map (Instance)
- `get(key: String) -> Tensor<F32, 0>`
- `get_1d(key: String) -> Tensor<F32, 1>`
- `get_quantized(key: String) -> i64`
- `set(key: String, value: String) -> void`
- `free() -> void`
