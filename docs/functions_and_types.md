# TensorLogic API Reference

This document lists the currently supported **global functions**, **types**, and their
**methods** as implemented by the compiler/runtime at the time of writing.

Notes:
- Signatures are given in TL terms.
- `Tensor<T, R>` means tensor of element type `T` and rank `R` (rank may be dynamic).
- Many numeric and tensor methods are also available via operators (`+`, `-`, `*`, `/`, `%`).

---

## 1. Global Functions

### I/O
- `print(value, ...) -> void`  
  Prints without newline. First argument must be a string literal if using `{}` formatting.
- `println(value, ...) -> void`  
  Prints with newline. First argument must be a string literal if using `{}` formatting.
- `read_line(prompt: String) -> String`  
  Prints prompt and reads a line from stdin.

### Args
- `args_count() -> i64`  
  Returns the number of CLI args.
- `args_get(index: i64) -> String`  
  Returns the CLI arg at `index`.

---

## 2. Standard Types (Static Methods)

### Tensor (static)
- `Tensor::zeros(shape, requires_grad: bool) -> Tensor`
- `Tensor::randn(shape, requires_grad: bool) -> Tensor`
- `Tensor::ones(shape, requires_grad: bool) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Param (parameter management)
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

### VarBuilder (static)
- `VarBuilder::get(name: String, ...dims) -> Tensor`
- `VarBuilder::grad(t: Tensor) -> Tensor`

### File (static)
- `File::open(path: String, mode: String) -> File`
- `File::exists(path: String) -> bool`
- `File::read(path: String) -> String`
- `File::write(path: String, content: String) -> bool`
- `File::download(url: String, dest: String) -> bool`

### Path (static)
- `Path::new(path: String) -> Path`

### Tokenizer (static)
- `Tokenizer::new(path: String) -> Tokenizer`

### KVCache (static)
- `KVCache::new(max_len: i64) -> KVCache`

### Map (static)
- `Map::load(path: String) -> Map`

### System (static)
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

### Env (static)
- `Env::get(key: String) -> String`
- `Env::set(key: String, value: String) -> void`

### Http (static)
- `Http::get(url: String) -> String`
- `Http::download(url: String, dest: String) -> bool`

### Image (static)
- `Image::load_grayscale(path: String) -> Tensor`
- `Image::width() -> i64`
- `Image::height() -> i64`

---

## 3. Instance Methods

### Tensor (instance)

#### Shape & indexing
- `reshape(shape) -> Tensor`
- `narrow(dim: i64, start: i64, len: i64) -> Tensor`
- `slice(start: i64, len: i64) -> Tensor`
- `transpose(dim1: i64, dim2: i64) -> Tensor`
- `transpose_2d() -> Tensor`
- `len() -> i64`
- `dim() -> i64`
- `get_shape() -> Tensor` (shape tensor)
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

#### Elementwise / activation
- `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`
- `exp()`, `log()`, `sqrt()`, `abs()`, `neg()`
- `sin()`, `cos()`, `tan()`
- `pow(exp)`, `pow(exp_tensor)`  
- `clamp(min, max) -> Tensor`
- `scale(s: f32) -> Tensor`

#### Linear algebra & NN ops
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

#### Arithmetic (method form)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)`, `mod(other)`
- `add_assign(other)`, `sub_assign(other)`, `mul_assign(other)`, `div_assign(other)`, `mod_assign(other)`

#### Autograd
- `backward() -> void`
- `grad() -> Tensor`
- `detach() -> Tensor`
- `enable_grad() -> Tensor`
- `clone() -> Tensor`

#### Device
- `cuda() -> Tensor`
- `cpu() -> Tensor`

#### Debug / IO
- `print()`, `print_1()`, `print_2()`, `print_3()`
- `item() -> f32`
- `item_i64() -> i64`

#### Quantized / misc
- `matmul_quantized(weight) -> Tensor`
- `cat_i64(other) -> Tensor`

---

### Numeric types (F32, F64, I32, I64)

#### F32 / F64
Unary:
`abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `cos`, `cosh`,
`exp`, `exp2`, `exp_m1`, `floor`, `fract`, `ln`, `ln_1p`, `log10`, `log2`, `recip`, `round`,
`signum`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `to_degrees`, `to_radians`, `trunc`

Binary:
`atan2(x)`, `copysign(x)`, `hypot(x)`, `log(x)`, `powf(x)`, `pow(x)`

Also: `powi(int)`

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
- `len() -> i64`
- `concat(other: String) -> String`
- `char_at(index: i64) -> String`
- `print() -> void`
- `display() -> void`
- `to_i64() -> i64`

---

### File (instance)
- `read_string() -> String`
- `write_string(s: String) -> void`
- `read_to_end() -> Vec<u8>`
- `write(bytes: Vec<u8>) -> void`
- `close() -> void`
- `free() -> void`

---

### Path (instance)
- `join(part: String) -> Path`
- `exists() -> bool`
- `is_dir() -> bool`
- `is_file() -> bool`
- `to_string() -> String`
- `free() -> void`

---

### Vec<T> (limited runtime support)
- `len() -> i64`
- `read_i32_be(index: i64) -> i64` (Vec<u8> only)
- `free() -> void`

---

### Tokenizer (instance)
- `encode(text: String) -> Tensor<I64,1>`
- `decode(tokens: Tensor<I64,1>) -> String`
- `token_id(token: String) -> i64`
- `vocab_size() -> i64`
- `free() -> void`

---

### KVCache (instance)
- `get_k(layer: i64) -> Tensor<F32,0>`
- `get_v(layer: i64) -> Tensor<F32,0>`
- `update(layer: i64, k: Tensor, v: Tensor) -> void`
- `free() -> void`

---

### Map (instance)
- `get(key: String) -> Tensor<F32,0>`
- `get_1d(key: String) -> Tensor<F32,1>`
- `get_quantized(key: String) -> i64`
- `set(key: String, value: String) -> void`
- `free() -> void`

