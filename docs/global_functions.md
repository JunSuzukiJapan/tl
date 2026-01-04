# TensorLogic API リファレンス

このドキュメントは、TensorLogic言語の標準API（グローバル関数および標準クラス）のリファレンスです。

---

## 1. グローバル関数

### 入出力

- **`print(value) -> void`**
  値を標準出力に出力します（改行なし）。
- **`println(value) -> void`**
  値を標準出力に出力します（改行あり）。

### テンソル生成・操作

- **`randn(shape, requires_grad) -> Tensor`**
  標準正規分布に従う乱数でテンソルを生成します。
- **`matmul(a, b) -> Tensor`**
  行列積を計算します。
- **`transpose(tensor, dim0, dim1) -> Tensor`**
  次元を入れ替えます。
- **`reshape(tensor, ...dims) -> Tensor`**
  テンソルの形状を変更します。
- **`slice(tensor, start, length) -> Tensor`**
  第1次元を切り出します。
- **`len(tensor) -> i64`**
  第1次元の要素数を返します。
- **`sum(tensor, dim?) -> Tensor`**
  総和を計算します。
- **`tril(tensor, diagonal) -> Tensor`**
  下三角行列を取り出します。
- **`embedding(indices, weights) -> Tensor`**
  埋め込み参照を行います。

### 数学関数 (Element-wise)

- **`pow(base, exp) -> Tensor`**
- **`exp(tensor) -> Tensor`**
- **`log(tensor) -> Tensor`**
- **`sqrt(tensor) -> Tensor`**
- **`sin(tensor) -> Tensor`**
- **`cos(tensor) -> Tensor`**

### ニューラルネットワーク

- **`relu(tensor) -> Tensor`**
- **`gelu(tensor) -> Tensor`**
- **`softmax(tensor, dim) -> Tensor`**
- **`cross_entropy(logits, targets) -> Tensor`**

### 自動微分

- **`backward(loss) -> void`**
- **`grad(tensor) -> Tensor`**
- **`enable_grad(tensor) -> Tensor`**

### パラメータ管理

- **`parameter(tensor) -> Tensor`**
- **`add_parameter(name, tensor) -> void`**
- **`update_all_params(lr) -> void`**
- **`save_all_params(path) -> void`**
- **`load_all_params(path) -> void`**
- **`save_weights(target, path) -> void`**
- **`load_weights(path) -> Tensor`**
- **`varbuilder_get(name, ...dims) -> Tensor`**
- **`varbuilder_grad(name) -> Tensor`**

### 制御・ユーティリティ

- **`range(start, end) -> Iterator`**
- **`tl_get_memory_mb() -> i64`**

---

## 2. 標準ライブラリクラス (名前空間)

これらは `型名::メソッド名` (静的メソッド) またはインスタンスメソッドとして呼び出します。

### File クラス
ファイル入出力を扱います。

**静的メソッド:**
- **`File::open(path: String, mode: String) -> File`**
  ファイルを指定モード（"r", "w"など）で開きます。

**インスタンスメソッド:**
- **`file.read_string() -> String`**
  ファイル全体を文字列として読み込みます。
- **`file.write_string(content: String) -> void`**
  文字列をファイルに書き込みます。
- **`file.close() -> void`**
  ファイルを閉じます。

### Path クラス
ファイルパス操作を行います。

**静的メソッド:**
- **`Path::new(path: String) -> Path`**
  新しいパスオブジェクトを作成します。

**インスタンスメソッド:**
- **`path.join(part: String) -> Path`**
  パスを結合します。
- **`path.exists() -> bool`**
- **`path.is_dir() -> bool`**
- **`path.is_file() -> bool`**
- **`path.to_string() -> String`**

### Http 名前空間
HTTPリクエストを行います。

**静的メソッド:**
- **`Http::get(url: String) -> String`**
  GETリクエストを行い、レスポンスボディを返します。
- **`Http::download(url: String, dest: String) -> bool`**
  ファイルをダウンロードします。成功時は `true` を返します。

### Env 名前空間
環境変数を扱います。

**静的メソッド:**
- **`Env::get(key: String) -> String`**
  環境変数の値を取得します。
- **`Env::set(key: String, value: String) -> void`**
  環境変数を設定します。

### System 名前空間
システム関連の機能です。

**静的メソッド:**
- **`System::time() -> f32`**
  システム時刻（秒）を取得します。
- **`System::sleep(seconds: f32) -> void`**
  指定秒数スリープします。
