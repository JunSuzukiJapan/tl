# TensorLogic API リファレンス

このドキュメントは、現在のコンパイラ実装に基づくTensorLogic言語の標準API（グローバル関数、標準クラス、組み込み型）のリファレンスです。

---

## 1. グローバル関数

### IO & システム
*   **`print(value) -> void`**
    値を標準出力に出力します（改行なし）。
*   **`println(value) -> void`**
    値を標準出力に出力します（改行あり）。

---

## 2. 標準型 & 静的メソッド（クラスメソッド）

### Tensor
*   **`Tensor::zeros(shape, requires_grad) -> Tensor`**
*   **`Tensor::randn(shape, requires_grad) -> Tensor`**
*   **`Tensor::ones(shape, requires_grad) -> Tensor`**
*   **`Tensor::load(path: String) -> Tensor`** — ファイルからテンソルを読み込み

### Param（パラメータ管理）
*   **`Param::save_all(path: String) -> void`** — 全パラメータを保存
*   **`Param::load_all(path: String) -> void`** — 全パラメータを読み込み
*   **`Param::save(target, path: String) -> void`** — 対象を保存
*   **`Param::load(path: String) -> Tensor`** — ファイルから読み込み
*   **`Param::add(name: String, t: Tensor) -> void`** — パラメータを追加
*   **`Param::register(t: Tensor) -> Tensor`** — テンソルをパラメータとして登録
*   **`Param::update_all(lr: f32) -> void`** — 全パラメータを更新
*   **`Param::register_modules(root: Struct) -> void`** — モジュールを登録
*   **`Param::checkpoint(method, input) -> Tensor`** — 活性化チェックポイント
*   **`Param::set_device(device: Device) -> void`** — 計算デバイスを設定

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

## 3. インスタンスメソッド

### Tensor メソッド
使用法: `tensor.method(...)`

#### 数学 (要素ごと)
`abs()`, `neg()`, `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()`,
`sin()`, `cos()`, `tan()`, `sqrt()`, `exp()`, `log()`, `pow(exp)`

#### 削減 & 統計
`sum(dim?)`, `mean(dim?)`, `max(dim?)`, `min(dim?)`, `argmax(dim)`, `argmin(dim)`,
`softmax(dim)`, `log_softmax(dim)`

#### 形状 & 操作
*   **`reshape(...dims) -> Tensor`** — 形状を変更
*   **`transpose(dim1, dim2) -> Tensor`** — 次元を転置
*   **`slice(start, len) -> Tensor`** — スライス
*   **`contiguous() -> Tensor`** — メモリ連続化
*   **`len() -> i64`** — 最初の次元のサイズ
*   **`item() -> f32`** — スカラー値を取得
*   **`item_i64() -> i64`** — 整数スカラー値を取得
*   **`to_i64() -> Tensor`** — i64型に変換

#### 自動微分
*   **`backward() -> void`** — 逆伝播（損失テンソルから呼び出し）
*   **`grad() -> Tensor`** — 勾配を取得
*   **`enable_grad() -> Tensor`** — 勾配追跡を有効化
*   **`detach() -> Tensor`** — 計算グラフから切り離す
*   **`clone() -> Tensor`** — テンソルを複製

#### デバイス
*   **`cuda() -> Tensor`** — CUDAに移動
*   **`cpu() -> Tensor`** — CPUに移動

#### 線形代数
*   **`matmul(other) -> Tensor`** — 行列積
*   **`tril(diagonal) -> Tensor`** — 下三角行列
*   **`embedding(weights) -> Tensor`** — 埋め込みルックアップ
*   **`cross_entropy(targets) -> Tensor`** — 交差エントロピー損失

#### 畳み込み
*   **`conv2d(weight, padding, stride) -> Tensor`** — 2D畳み込み
*   **`clamp(min, max) -> Tensor`** — 値をクランプ

### スカラーメソッド (f32, f64)
*   **数学:** `abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `powf`, `floor`, `ceil`, `round` など
