# TensorLogic API リファレンス

このドキュメントは、現在サポートされている**グローバル関数**、**型**、およびそれらの**メソッド**をリストしたものです。これらは執筆時点のコンパイラ/ランタイムの実装に基づいています。

注意:
- シグネチャは TL の形式で記述されています。
- `Tensor<T, R>` は、要素型 `T`、ランク `R` のテンソルを意味します（ランクは動的な場合があります）。
- 多くの数値およびテンソルメソッドは、演算子（`+`, `-`, `*`, `/`, `%`）を介しても利用可能です。

---

## 1. グローバル関数

### I/O (入出力)
- `print(value, ...) -> void`  
  改行なしで出力します。`{}` フォーマットを使用する場合、最初の引数は文字列リテラルである必要があります。
- `println(value, ...) -> void`  
  改行付きで出力します。`{}` フォーマットを使用する場合、最初の引数は文字列リテラルである必要があります。
- `read_line(prompt: String) -> String`  
  プロンプトを表示し、標準入力から一行読み取ります。

### Args (引数)
- `args_count() -> i64`  
  コマンドライン引数の数を返します。
- `args_get(index: i64) -> String`  
  `index` 番目のコマンドライン引数を返します。

### System (制御)
- `panic(message: String) -> Never`  
  エラーメッセージを出力してプログラムを終了します。

---

## 2. 標準型（静的メソッド）

### Tensor (静的)
- `Tensor::zeros(shape, requires_grad: bool) -> Tensor`
- `Tensor::randn(shape, requires_grad: bool) -> Tensor`
- `Tensor::ones(shape, requires_grad: bool) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Vec\<T\> (静的)
- `Vec<T>::new() -> Vec<T>` — 空の Vec を作成
- `Vec<T>::with_capacity(cap: i64) -> Vec<T>` — 指定容量で Vec を作成

### HashMap\<K, V\> (静的)
- `HashMap<K, V>::new() -> HashMap<K, V>` — 空の HashMap を作成

### Option\<T\>（Enum）
- `Option::Some(value: T)` — 値を保持するバリアント
- `Option::None` — 値なしバリアント

### Result\<T, E\>（Enum）
- `Result::Ok(value: T)` — 成功値を保持するバリアント
- `Result::Err(error: E)` — エラー値を保持するバリアント

### Param (パラメータ管理)
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

### VarBuilder (静的)
- `VarBuilder::get(name: String, ...dims) -> Tensor`
- `VarBuilder::grad(t: Tensor) -> Tensor`

### File (静的)
- `File::open(path: String, mode: String) -> File`
- `File::exists(path: String) -> bool`
- `File::read(path: String) -> String`
- `File::write(path: String, content: String) -> bool`
- `File::download(url: String, dest: String) -> bool`

### Path (静的)
- `Path::new(path: String) -> Path`

### Tokenizer (静的)
- `Tokenizer::new(path: String) -> Tokenizer`

### KVCache (静的)
- `KVCache::new(max_len: i64) -> KVCache`

### Map (静的)
- `Map::load(path: String) -> Map`

### System (静的)
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

### Env (静的)
- `Env::get(key: String) -> String`
- `Env::set(key: String, value: String) -> void`

### Http (静的)
- `Http::get(url: String) -> String`
- `Http::download(url: String, dest: String) -> bool`

### Image (静的)
- `Image::load_grayscale(path: String) -> Tensor`
- `Image::width() -> i64`
- `Image::height() -> i64`

---

## 3. インスタンスメソッド

### Tensor (インスタンス)

#### 形状とインデックス
- `reshape(shape) -> Tensor`
- `narrow(dim: i64, start: i64, len: i64) -> Tensor`
- `slice(start: i64, len: i64) -> Tensor`
- `transpose(dim1: i64, dim2: i64) -> Tensor`
- `transpose_2d() -> Tensor`
- `len() -> i64`
- `dim() -> i64`
- `get_shape() -> Tensor` (形状テンソル)
- `get(i64...) -> f32`
- `set(i64..., value: f32) -> void`

#### リダクション (簡約)
- `sum(dim?) -> Tensor`
- `sum_dim(dim: i64, keepdim: bool) -> Tensor`
- `mean(dim?) -> Tensor`
- `max(dim?) -> Tensor`
- `min(dim?) -> Tensor`
- `argmax(dim: i64) -> Tensor`
- `argmin(dim: i64) -> Tensor`

#### 要素ごとの演算 / 活性化関数
- `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`
- `exp()`, `log()`, `sqrt()`, `abs()`, `neg()`
- `sin()`, `cos()`, `tan()`
- `pow(exp)`, `pow(exp_tensor)`  
- `clamp(min, max) -> Tensor`
- `scale(s: f32) -> Tensor`

#### 線形代数とニューラルネットワーク演算
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

#### 算術演算 (メソッド形式)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)`, `mod(other)`
- `add_assign(other)`, `sub_assign(other)`, `mul_assign(other)`, `div_assign(other)`, `mod_assign(other)`

#### 自動微分 (Autograd)
- `backward() -> void`
- `grad() -> Tensor`
- `detach() -> Tensor`
- `enable_grad() -> Tensor`
- `clone() -> Tensor`

#### デバイス
- `cuda() -> Tensor`
- `cpu() -> Tensor`

#### デバッグ / I/O
- `print()`, `print_1()`, `print_2()`, `print_3()`
- `item() -> f32`
- `item_i64() -> i64`

#### 量子化 / その他
- `matmul_quantized(weight) -> Tensor`
- `cat_i64(other) -> Tensor`

---

### Vec\<T\> (インスタンス)
- `len() -> i64` — 要素数を返す
- `capacity() -> i64` — 容量を返す
- `is_empty() -> bool` — 空かどうか
- `push(item: T) -> void` — 末尾に要素を追加
- `pop() -> Option<T>` — 末尾要素を取り出す
- `get(index: i64) -> Option<T>` — インデックスで要素を取得
- `set(index: i64, item: T) -> void` — インデックスで要素を更新

---

### HashMap\<K, V\> (インスタンス)
- `len() -> i64` — 要素数を返す
- `is_empty() -> bool` — 空かどうか
- `insert(key: K, value: V) -> void` — キーと値のペアを挿入
- `get(key: K) -> Option<V>` — キーで値を検索
- `remove(key: K) -> void` — キーに対応するエントリを削除（未実装）

---

### Option\<T\> (インスタンス)
- `is_some() -> bool` — `Some` かどうか
- `is_none() -> bool` — `None` かどうか
- `unwrap() -> T` — 値を取り出す（`None` の場合 panic）
- `unwrap_or(default: T) -> T` — 値を取り出す（`None` の場合デフォルト値）

---

### Result\<T, E\> (インスタンス)
- `is_ok() -> bool` — `Ok` かどうか
- `is_err() -> bool` — `Err` かどうか
- `unwrap() -> T` — 値を取り出す（`Err` の場合 panic）
- `unwrap_err() -> E` — エラーを取り出す（`Ok` の場合 panic）

`?` 演算子: `Result` 型の値に使用でき、`Err` の場合は早期リターンします。

---

### 数値型 (F32, F64, I32, I64)

#### F32 / F64
単項演算:
`abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cbrt`, `ceil`, `cos`, `cosh`,
`exp`, `exp2`, `exp_m1`, `floor`, `fract`, `ln`, `ln_1p`, `log`, `log10`, `log2`, `recip`, `round`,
`signum`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `to_degrees`, `to_radians`, `trunc`

二項演算:
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

### String (文字列)
- `len() -> i64` — 文字列長
- `contains(other: String) -> bool` — 部分文字列を含むか
- `concat(other: String) -> String` — 文字列結合
- `char_at(index: i64) -> Char` — 指定位置の文字
- `to_i64() -> i64` — 整数にパース
- `print() -> void` — 出力
- `display() -> void` — 出力

---

### File (インスタンス)
- `read_string() -> String`
- `write_string(s: String) -> void`
- `read_to_end() -> Vec<u8>`
- `write(bytes: Vec<u8>) -> void`
- `close() -> void`
- `free() -> void`

---

### Path (インスタンス)
- `join(part: String) -> Path`
- `exists() -> bool`
- `is_dir() -> bool`
- `is_file() -> bool`
- `to_string() -> String`
- `free() -> void`

---

### Tokenizer (インスタンス)
- `encode(text: String) -> Tensor<I64, 1>`
- `decode(tokens: Tensor<I64, 1>) -> String`
- `token_id(token: String) -> i64`
- `vocab_size() -> i64`
- `free() -> void`

---

### KVCache (インスタンス)
- `get_k(layer: i64) -> Tensor<F32, 0>`
- `get_v(layer: i64) -> Tensor<F32, 0>`
- `update(layer: i64, k: Tensor, v: Tensor) -> void`
- `free() -> void`

---

### Map (インスタンス)
- `get(key: String) -> Tensor<F32, 0>`
- `get_1d(key: String) -> Tensor<F32, 1>`
- `get_quantized(key: String) -> i64`
- `set(key: String, value: String) -> void`
- `free() -> void`
