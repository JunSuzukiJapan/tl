# TensorLogic API Reference

This document lists the currently supported **global functions**, **types**, and their **methods**. These are based on the compiler/runtime implementation at the time of writing.

Notes:
- Signatures are written in TL notation.
- `Tensor<T, R>` means a tensor with element type `T` and rank `R` (rank may be dynamic).
- `GradTensor<T, R>` is a gradient-tracking tensor (for training).
- Many numeric and tensor methods are also available via operators (`+`, `-`, `*`, `/`, `%`).

---

## 1. Global Functions

### I/O
- `print(value, ...) -> void`  
  Prints without newline. When using `{}` format, the first argument must be a string literal.
- `println(value, ...) -> void`  
  Prints with newline. When using `{}` format, the first argument must be a string literal.
- `read_line(prompt: String) -> String`  
  Displays a prompt and reads one line from stdin.

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
- `Tensor::zeros(shape, requires_grad: bool) -> Tensor`
- `Tensor::randn(shape, requires_grad: bool) -> Tensor`
- `Tensor::ones(shape, requires_grad: bool) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Vec\<T\> (Static)
- `Vec<T>::new() -> Vec<T>` — Create an empty Vec
- `Vec<T>::with_capacity(cap: i64) -> Vec<T>` — Create a Vec with specified capacity

### HashMap\<K, V\> (Static)
- `HashMap<K, V>::new() -> HashMap<K, V>` — Create an empty HashMap

### Option\<T\> (Enum)
- `Option::Some(value: T)` — Variant holding a value
- `Option::None` — Variant without a value

### Result\<T, E\> (Enum)
- `Result::Ok(value: T)` — Success variant
- `Result::Err(error: E)` — Error variant

### Param (Parameter Management)
- `Param::save_all(path: String, format?: String) -> void`
- `Param::load_all(path: String, format?: String) -> void`
- `Param::save(target, path: String) -> void`
- `Param::load(path: String) -> Tensor`
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
- `view(shape) -> Tensor`
- `narrow(dim: i64, start: i64, len: i64) -> Tensor`
- `slice(start: i64, len: i64) -> Tensor`
- `transpose(dim1: i64, dim2: i64) -> Tensor`
- `transpose_2d() -> Tensor`
- `permute(dims) -> Tensor`
- `contiguous() -> Tensor`
- `flatten() -> Tensor`
- `squeeze(dim: i64) -> Tensor`
- `unsqueeze(dim: i64) -> Tensor`
- `expand(shape) -> Tensor`
- `broadcast_to(shape) -> Tensor`
- `cat(tensors, dim: i64) -> Tensor`
- `chunk(chunks: i64, dim: i64) -> Tensor`
- `split(split_size: i64, dim: i64, idx: i64) -> Tensor`
- `gather(dim: i64, index: Tensor) -> Tensor`
- `len() -> i64`, `dim() -> i64`, `ndim() -> i64`
- `shape() -> Tensor`, `get_shape() -> Tensor`
- `get(i64...) -> f32`, `set(i64..., value: f32) -> void`

#### Reductions
- `sum(dim?) -> Tensor`, `sum_dim(dim, keepdim) -> Tensor`, `sumall() -> f32`
- `mean(dim?) -> Tensor`, `mean_dim(dim) -> Tensor`
- `max(dim?) -> Tensor`, `max_dim(dim) -> Tensor`
- `min(dim?) -> Tensor`, `min_dim(dim) -> Tensor`
- `argmax(dim) -> Tensor`, `argmin(dim) -> Tensor`
- `prod() -> Tensor`, `var() -> Tensor`, `std() -> Tensor`
- `cumsum(dim) -> Tensor`, `norm(p) -> Tensor`, `topk(k, dim) -> Tensor`

#### Element-wise Operations
- `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`
- `leaky_relu(negative_slope) -> Tensor`, `elu(alpha) -> Tensor`
- `mish() -> Tensor`, `hardswish() -> Tensor`, `hardsigmoid() -> Tensor`
- `exp()`, `log()`, `sqrt()`, `abs()`, `neg()`
- `sin()`, `cos()`, `tan()`, `pow(exp)`, `clamp(min, max) -> Tensor`, `scale(s) -> Tensor`

#### Logical Operations
- `logical_not() -> Tensor`

#### Neural Network Operations
- `matmul(other)`, `matmul_4d(other)`, `linear(weight, bias?)`
- `conv1d(weight, bias?, stride, padding)`, `conv2d(weight, padding, stride)`
- `conv_transpose2d(weight, bias?, stride, padding, output_padding)`
- `max_pool2d(kernel, stride)`, `avg_pool2d(kernel, stride)`
- `adaptive_avg_pool2d(output_h, output_w)`, `interpolate(output_h, output_w, mode)`
- `pad(pad_left, pad_right, value)`, `dropout(p, training)`, `dropout2d(p, training)`
- `batch_norm(w, b, rm, rv, eps)`, `layer_norm(w, b, eps)`
- `group_norm(groups, w?, b?, eps)`, `instance_norm(w?, b?, eps)`, `rms_norm(w, eps)`
- `embedding(weights)`, `cross_entropy(targets)`, `softmax(dim)`, `log_softmax(dim)`
- `tril(diagonal)`, `masked_fill(mask, value)`, `apply_rope(cos, sin, dim)`
- `repeat_interleave(dim, repeats)`, `dot(other)`, `fill_(value)`, `temperature_scale(temp)`

#### Linear Algebra
- `inverse()`, `det()`, `solve(b)`, `svd_u()`, `svd_s()`, `svd_v()`, `eig_values()`, `eig_vectors()`

#### Sampling
- `sample() -> i64`, `top_k_sample(k, temp) -> i64`, `top_p_sample(p, temp) -> i64`
- `repetition_penalty(penalty, previous_tokens) -> Tensor`

#### Arithmetic (Method form)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)`, `mod(other)`
- `add_assign(other)`, `sub_assign(other)`, `mul_assign(other)`, `div_assign(other)`

#### Autograd
- `backward()`, `grad()`, `detach()`, `enable_grad()`, `clone()`, `shallow_clone()`
- `freeze()`, `unfreeze()`, `clip_grad_norm(max_norm)`, `clip_grad_value(value)`

#### Device
- `cuda() -> Tensor`, `cpu() -> Tensor`, `to(device) -> Tensor`

#### I/O
- `print()`, `display()`, `item() -> f32`, `item_i64() -> i64`
- `to_f32() -> Tensor`, `to_i64() -> Tensor`, `save(path) -> void`

---

### Vec\<T\> (Instance)
- `len() -> i64`, `capacity() -> i64`, `is_empty() -> bool`
- `push(item: T)`, `pop() -> Option<T>`, `get(index) -> Option<T>`, `set(index, item: T)`
- `map(f: Fn(T) -> U) -> Vec<U>` — Apply a function to each element
- `filter(f: Fn(T) -> bool) -> Vec<T>` — Keep only elements satisfying a condition

---

### HashMap\<K, V\> (Instance)
- `len() -> i64`, `is_empty() -> bool`
- `insert(key, value)`, `get(key) -> Option<V>`, `remove(key)`
- `contains_key(key) -> bool`, `keys() -> Vec<K>`, `values() -> Vec<V>`

---

### Option\<T\> (Instance)
- `is_some() -> bool`, `is_none() -> bool`
- `unwrap() -> T`, `unwrap_or(default: T) -> T`

---

### Result\<T, E\> (Instance)
- `is_ok() -> bool`, `is_err() -> bool`
- `unwrap() -> T`, `unwrap_err() -> E`

`?` operator: Can be used with `Result` types for early return on `Err`.

---

### Numeric Types (F32, F64, I32, I64)

#### F32 / F64
Unary: `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `cos`, `cosh`,
`exp`, `exp2`, `exp_m1`, `floor`, `fract`, `ln`, `ln_1p`, `log`, `log10`, `log2`, `recip`, `round`,
`signum`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `to_degrees`, `to_radians`, `trunc`

Binary: `atan2(x)`, `copysign(x)`, `hypot(x)`, `powf(x)`, `pow(x)`, `powi(x)`

Conversion: `to_i64()`, `to_f32()`, `to_f64()`, `to_string()`

Other: `min(x)`, `max(x)`, `clamp(min, max)`

#### I64 / I32
- `abs()`, `signum()`, `is_positive()`, `is_negative()`
- `div_euclid(x)`, `rem_euclid(x)`, `pow(x)`
- `min(x)`, `max(x)`, `clamp(min, max)`
- `to_f32()`, `to_f64()`, `to_string()`

---

### String
- `len()`, `is_empty()`, `contains(other)`, `starts_with(prefix)`, `ends_with(suffix)`
- `split(sep) -> Vec<String>`, `trim()`, `replace(from, to)`, `substring(start, len)`
- `concat(other)`, `index_of(s) -> i64`, `char_at(index) -> Char`
- `to_uppercase()`, `to_lowercase()`
- `to_i64()`, `to_f32()`, `to_f64()`, `print()`, `display()`

---

### File (Instance)
- `read_string()`, `write_string(s)`, `read_to_end() -> Vec<u8>`, `write(bytes)`, `close()`, `free()`

### Path (Instance)
- `join(part)`, `exists()`, `is_dir()`, `is_file()`, `to_string()`, `free()`

### Tokenizer (Instance)
- `encode(text) -> Tensor<I64, 1>`, `decode(tokens) -> String`, `token_id(token) -> i64`, `vocab_size() -> i64`, `free()`

### KVCache (Instance)
- `get_k(layer)`, `get_v(layer)`, `update(layer, k, v)`, `free()`

### Map (Instance)
- `get(key)`, `get_1d(key)`, `get_quantized(key) -> i64`, `set(key, value)`, `free()`
