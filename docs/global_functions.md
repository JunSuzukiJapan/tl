# TensorLogic API リファレンス

このドキュメントは、現在のコンパイラ実装に基づくTensorLogic言語の標準API（グローバル関数、標準クラス、組み込み型）のリファレンスです。

---

## 1. グローバル関数

### IO & システム
*   **`print(value) -> void`**
    値を標準出力に出力します（改行なし）。
*   **`println(value) -> void`**
    値を標準出力に出力します（改行あり）。
*   **`tl_system_time() -> f32`**
    現在のシステム時間を秒単位で返します。
*   **`tl_get_memory_mb() -> i64`**
    現在のメモリ使用量をMB単位で返します。

### 数学 & テンソル操作
*   **`sin(t) -> Tensor`**, **`cos(t) -> Tensor`**
*   **`relu(t) -> Tensor`**, **`gelu(t) -> Tensor`**
*   **`exp(t) -> Tensor`**, **`log(t) -> Tensor`**, **`sqrt(t) -> Tensor`**
*   **`pow(base, exp) -> Tensor`**
*   **`matmul(a, b) -> Tensor`**
*   **`transpose(t, dim1, dim2) -> Tensor`**
*   **`reshape(t, ...dims) -> Tensor`**
    テンソルの形状を変更します。可変長引数で次元を指定できます。
*   **`slice(t, start, len) -> Tensor`**
*   **`len(t) -> i64`**
*   **`tril(t, diagonal) -> Tensor`**
    行列の下三角部分を返します。
*   **`embedding(indices, weights) -> Tensor`**
*   **`sum(t, dim?) -> Tensor`**
    要素の合計を計算します。ロジック: `sum(t)` (全要素), `sum(t, dim)` (次元指定)
*   **`softmax(t, dim) -> Tensor`**
*   **`cross_entropy(logits, targets) -> Tensor`**
*   **`argmax(t, dim) -> Tensor`**
*   **`item(t) -> f32/i64`**
    0次元テンソル（または単一要素テンソル）のスカラー値を返します。

### 自動微分 & 学習
*   **`grad(t) -> Tensor`**
    テンソルの勾配を返します。
*   **`backward(loss) -> void`**
    `loss`から開始して勾配を計算します。
*   **`enable_grad(t) -> Tensor`**
    テンソルの勾配追跡を有効にします。

### パラメータ管理
*   **`save_all_params(path: String) -> void`**
*   **`load_all_params(path: String) -> void`**
*   **`save_weights(target: Tensor|Struct, path: String) -> void`**
*   **`load_weights(path: String) -> Tensor|void`**
*   **`add_parameter(name: String, t: Tensor) -> void`**
*   **`parameter(t: Tensor) -> Tensor`**
    テンソルをパラメータとして登録します。
*   **`register_modules(root: Struct) -> void`**
*   **`checkpoint(method, input) -> Tensor`**
*   **`set_device(device: Device) -> void`**
    グローバルな計算デバイスを設定します (`Device::Cpu`, `Device::Cuda`, `Device::Metal`, `Device::Auto`).

---

## 2. 標準型 & 静的メソッド

### Tensor
*   **`Tensor::zeros(shape, requires_grad) -> Tensor`**
*   **`Tensor::randn(shape, requires_grad) -> Tensor`**

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

## 3. インスタンスメソッド

### Tensor メソッド
使用法: `tensor.method(...)`

*   **数学 (要素ごと):**
    `abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
    `sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`
*   **削減 & 統計:**
    `sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
    `softmax(dim)`, `log_softmax(dim)`
*   **形状 & 操作:**
    `reshape(...)`, `transpose(d1, d2)`, `slice(start, len)`, `contiguous()`,
    `len()`, `item()`, `item_i64()`, `to_i64()`
*   **自動微分:**
    `backward()`, `grad()`, `detach()`, `clone()`
*   **デバイス:**
    `cuda()`, `cpu()`
*   **線形代数:**
    `matmul(other)`

### スカラーメソッド (f32, f64)
*   **数学:** `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round` など
