# TensorLogic API Reference

This document is a reference for the standard API of the TensorLogic language (global functions, standard classes, built-in types) based on the current compiler implementation.

---

## 1. Global Functions

### IO & System
*   **`print(value) -> void`**
    Prints the value to standard output (without newline).
*   **`println(value) -> void`**
    Prints the value to standard output (with newline).

---

## 2. Standard Types & Static Methods (Class Methods)

### Tensor
*   **`Tensor::zeros(shape, requires_grad) -> Tensor`**
*   **`Tensor::randn(shape, requires_grad) -> Tensor`**
*   **`Tensor::ones(shape, requires_grad) -> Tensor`**
*   **`Tensor::load(path: String) -> Tensor`** — Load tensor from file

### Param (Parameter Management)
*   **`Param::save_all(path: String) -> void`** — Save all parameters
*   **`Param::load_all(path: String) -> void`** — Load all parameters
*   **`Param::save(target, path: String) -> void`** — Save target
*   **`Param::load(path: String) -> Tensor`** — Load from file
*   **`Param::add(name: String, t: Tensor) -> void`** — Add parameter
*   **`Param::register(t: Tensor) -> Tensor`** — Register tensor as parameter
*   **`Param::update_all(lr: f32) -> void`** — Update all parameters
*   **`Param::register_modules(root: Struct) -> void`** — Register module parameters
*   **`Param::checkpoint(method, input) -> Tensor`** — Activation checkpointing
*   **`Param::set_device(device: Device) -> void`** — Set computation device

### VarBuilder
*   **`VarBuilder::get(name: String, ...dims) -> Tensor`**

### File
*   **`File::open(path: String, mode: String) -> File`**

### Path
*   **`Path::new(path: String) -> Path`**

### System
*   **`System::time() -> f32`**
*   **`System::sleep(seconds: f32) -> void`**

### Env
*   **`Env::get(key: String) -> String`**
*   **`Env::set(key: String, value: String) -> void`**

### Http
*   **`Http::get(url: String) -> String`**
*   **`Http::download(url: String, dest: String) -> bool`**

### Image
*   **`Image::load_grayscale(path: String) -> Tensor`**
*   **`Image::width() -> i64`**
*   **`Image::height() -> i64`**

---

## 3. Instance Methods

### Tensor Methods
Usage: `tensor.method(...)`

#### Math (Element-wise)
`abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
`sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`, `pow(exp)`

#### Reduction & Statistics
`sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
`softmax(dim)`, `log_softmax(dim)`

#### Shape & Manipulation
*   **`reshape(...dims) -> Tensor`** — Change shape
*   **`transpose(dim1, dim2) -> Tensor`** — Transpose dimensions
*   **`slice(start, len) -> Tensor`** — Slicing
*   **`contiguous() -> Tensor`** — Make memory contiguous
*   **`len() -> i64`** — Size of the first dimension
*   **`item() -> f32`** — Get scalar value
*   **`item_i64() -> i64`** — Get integer scalar value
*   **`to_i64() -> Tensor`** — Convert to i64 type

#### Automatic Differentiation
*   **`backward() -> void`** — Backpropagation (call from loss tensor)
*   **`grad() -> Tensor`** — Get gradient
*   **`enable_grad() -> Tensor`** — Enable gradient tracking
*   **`detach() -> Tensor`** — Detach from computation graph
*   **`clone() -> Tensor`** — Clone tensor

#### Device
*   **`cuda() -> Tensor`** — Move to CUDA
*   **`cpu() -> Tensor`** — Move to CPU

#### Linear Algebra
*   **`matmul(other) -> Tensor`** — Matrix multiplication
*   **`tril(diagonal) -> Tensor`** — Lower triangular matrix
*   **`embedding(weights) -> Tensor`** — Embedding lookup
*   **`cross_entropy(targets) -> Tensor`** — Cross entropy loss

#### Convolution
*   **`conv2d(weight, padding, stride) -> Tensor`** — 2D convolution
*   **`clamp(min, max) -> Tensor`** — Clamp values

### Scalar Methods (f32, f64)
*   **Math:** `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round`, etc.
