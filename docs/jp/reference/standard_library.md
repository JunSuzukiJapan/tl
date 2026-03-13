# TensorLogic 標準ライブラリ概要

このドキュメントは、TL言語の標準ライブラリの概要です。
詳細な API リファレンスは [functions_and_types.md](functions_and_types.md) を参照してください。

---

## 1. グローバル関数

### IO & システム
- **`print(value, ...) -> void`** — 改行なし出力（`{}` フォーマット対応）
- **`println(value, ...) -> void`** — 改行付き出力（`{}` フォーマット対応）
- **`read_line(prompt: String) -> String`** — 標準入力から一行読み取り
- **`panic(message: String) -> Never`** — エラーメッセージを出力してプログラム終了

### コマンドライン引数
- **`args_count() -> i64`** — コマンドライン引数の数
- **`args_get(index: i64) -> String`** — コマンドライン引数を取得

---

## 2. 組み込みジェネリック型

### Vec\<T\> — 可変長配列
```rust
let mut v: Vec<i64> = Vec::new();
v.push(1);
v.push(2);
let first = v.get(0).unwrap(); // 1
let popped = v.pop();          // Option::Some(2)
```

| メソッド | 説明 |
|---|---|
| `Vec::new() -> Vec<T>` | 空の Vec を作成 |
| `Vec::with_capacity(cap) -> Vec<T>` | 指定容量で Vec を作成 |
| `len() -> i64` | 要素数 |
| `capacity() -> i64` | 容量 |
| `is_empty() -> bool` | 空か |
| `push(item: T)` | 末尾に追加 |
| `pop() -> Option<T>` | 末尾から取り出し |
| `get(index) -> Option<T>` | インデックスで取得 |
| `set(index, item: T)` | インデックスで更新 |
| `map(f: Fn(T) -> U) -> Vec<U>` | 各要素に関数を適用 |
| `filter(f: Fn(T) -> bool) -> Vec<T>` | 条件を満たす要素のみ残す |

### HashMap\<K, V\> — ハッシュマップ
```rust
let mut m: HashMap<String, i64> = HashMap::new();
m.insert("key", 42);
let val = m.get("key").unwrap(); // 42
```

| メソッド | 説明 |
|---|---|
| `HashMap::new() -> HashMap<K, V>` | 空の HashMap を作成 |
| `len() -> i64` | 要素数 |
| `is_empty() -> bool` | 空か |
| `insert(key, value)` | 挿入 |
| `get(key) -> Option<V>` | 検索 |
| `remove(key)` | 削除 |
| `contains_key(key) -> bool` | キーの存在判定 |
| `keys() -> Vec<K>` | キー一覧 |
| `values() -> Vec<V>` | 値一覧 |

### Option\<T\> — 値の有無
```rust
let some_val: Option<i64> = Option::Some(42);
let none_val: Option<i64> = Option::None;

match some_val {
    Option::Some(x) => println("got: {}", x),
    Option::None => println("none"),
}
```

| メソッド | 説明 |
|---|---|
| `is_some() -> bool` | `Some` か |
| `is_none() -> bool` | `None` か |
| `unwrap() -> T` | 値を取得（`None` で panic） |
| `unwrap_or(default: T) -> T` | デフォルト値付き取得 |

### Result\<T, E\> — 成功/失敗
```rust
fn divide(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Result::Err("division by zero");
    }
    Result::Ok(a / b)
}
```

| メソッド | 説明 |
|---|---|
| `is_ok() -> bool` | `Ok` か |
| `is_err() -> bool` | `Err` か |
| `unwrap() -> T` | 値を取得（`Err` で panic） |
| `unwrap_err() -> E` | エラーを取得 |

`?` 演算子: `Result` 型に使用でき、`Err` の場合は早期リターン。

---

## 3. 標準トレイト

| トレイト | メソッド | 説明 |
|---|---|---|
| `Index<Idx, Output>` | `index(self, idx) -> Output` | `[]` 読み取りアクセス |
| `IndexMut<Idx, Value>` | `set(self, idx, value)` | `[]` 書き込みアクセス |
| `Iterable<T>` | `len(self) -> i64`, `get(self, index) -> T` | `for` ループ用イテレータ |

---

## 4. クロージャ（無名関数）

TLはRustスタイルのクロージャをサポートします。

```rust
// 単一式
let double = |x: i64| x * 2;

// ブロック本体
let add = |x: i64, y: i64| -> i64 { x + y };

// 外部変数のキャプチャ
let factor = 3;
let mul = |x: i64| -> i64 { x * factor };

// 高階関数へ渡す
let nums = Vec::new();
nums.push(1); nums.push(2); nums.push(3);
let doubled = nums.map(|x: i64| -> i64 { x * 2 });
```

型表記: `Fn(i64, i64) -> i64`

---

## 5. 標準クラス（静的メソッド）

### Tensor
- `Tensor::zeros(shape, requires_grad) -> Tensor`
- `Tensor::randn(shape, requires_grad) -> Tensor`
- `Tensor::ones(shape, requires_grad) -> Tensor`
- `Tensor::load(path: String) -> Tensor`

### Param（パラメータ管理）
- `Param::save_all(path) -> void` — 全パラメータ保存
- `Param::load_all(path) -> void` — 全パラメータ読み込み
- `Param::register(t: Tensor) -> Tensor` — テンソルをパラメータとして登録
- `Param::update_all(lr: f32) -> void` — 全パラメータ更新
- `Param::register_modules(root: Struct) -> void` — モジュール登録
- `Param::checkpoint(method, input) -> Tensor` — 活性化チェックポイント
- `Param::set_device(device) -> void` — 計算デバイス設定

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

## 6. Tensor インスタンスメソッド（抜粋）

#### 数学 (要素ごと)
`abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
`sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`, `pow(exp)`,
`leaky_relu(slope)`, `elu(alpha)`, `mish()`, `hardswish()`, `hardsigmoid()`

#### 形状 & 操作
- `reshape(shape)`, `view(shape)`, `transpose(d1, d2)`, `permute(dims)`
- `flatten()`, `squeeze(dim)`, `unsqueeze(dim)`, `contiguous()`
- `slice(start, len)`, `narrow(dim, start, len)`, `chunk(n, dim)`, `split(size, dim, idx)`
- `cat(tensors, dim)`, `broadcast_to(shape)`, `expand(shape)`
- `len()`, `dim()`, `ndim()`, `shape()`, `item()`, `item_i64()`

#### リダクション & 統計
`sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
`softmax(dim)`, `log_softmax(dim)`, `prod()`, `var()`, `std()`, `cumsum(dim)`, `norm(p)`, `topk(k, dim)`

#### ニューラルネットワーク
- `matmul(other)`, `linear(weight, bias?)`
- `conv1d(w, b?, stride, padding)`, `conv2d(w, padding, stride)`
- `conv_transpose2d(w, b?, stride, padding, output_padding)`
- `max_pool2d(k, s)`, `avg_pool2d(k, s)`, `adaptive_avg_pool2d(oh, ow)`
- `interpolate(oh, ow, mode)`, `pad(left, right, value)`
- `dropout(p, training)`, `dropout2d(p, training)`
- `batch_norm(w, b, rm, rv, eps)`, `layer_norm(w, b, eps)`
- `group_norm(groups, w?, b?, eps)`, `instance_norm(w?, b?, eps)`
- `rms_norm(weight, eps)`, `embedding(weights)`, `cross_entropy(targets)`
- `apply_rope(cos, sin, dim)`, `dot(other)`, `fill_(value)`

#### 線形代数
- `inverse()`, `det()`, `solve(b)`, `svd_u()`, `svd_s()`, `svd_v()`, `eig_values()`, `eig_vectors()`

#### 自動微分
- `backward()`, `grad()`, `enable_grad()`, `detach()`, `clone()`
- `freeze()`, `unfreeze()`, `clip_grad_norm(max)`, `clip_grad_value(val)`

#### デバイス
- `cuda()`, `cpu()`, `to(device)`

---

## 7. スカラー型メソッド

### String
- `len()`, `is_empty()`, `contains(s)`, `starts_with(s)`, `ends_with(s)`
- `split(sep)`, `trim()`, `replace(from, to)`, `substring(start, len)`
- `concat(other)`, `index_of(s)`, `char_at(i)`
- `to_uppercase()`, `to_lowercase()`
- `to_i64()`, `to_f32()`, `to_f64()`

### F32 / F64
数学関数: `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round` 等（全31関数）
型変換: `to_i64()`, `to_f32()`, `to_f64()`, `to_string()`
その他: `min(x)`, `max(x)`, `clamp(min, max)`

### I64 / I32
- `abs()`, `signum()`, `is_positive()`, `is_negative()`
- `div_euclid(x)`, `rem_euclid(x)`, `pow(x)`
- `min(x)`, `max(x)`, `clamp(min, max)`
- `to_f32()`, `to_f64()`, `to_string()`
