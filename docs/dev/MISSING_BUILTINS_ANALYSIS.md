# TL言語 汎用言語化に向けた不足分析

現在のTL言語の組み込み型・関数を、Python / Rust / Go 等の汎用言語と比較し、不足している要素をリストアップする。

> [!NOTE]
> TL固有のドメイン（Tensor, Param, KVCache, VarBuilder, Map, Tokenizer, Image 等）は対象外。
> 汎用プログラミングに必要な基盤機能に焦点を当てる。

---

## 1. 不足している組み込み型

### 1.1 コレクション型

| 型 | 説明 | 優先度 |
|---|---|---|
| `Set<T>` / `HashSet<T>` | 一意な要素の集合。重複排除・集合演算（和・積・差）に必須 | **高** |
| `Tuple<T1, T2, ...>` | 異なる型の固定長組。多値返却に必要 | **高** |
| `Deque<T>` / `VecDeque<T>` | 両端キュー。BFS等のアルゴリズムに必要 | 中 |
| `SortedMap<K, V>` / `BTreeMap<K, V>` | 順序付きマップ。キーの順序走査が必要な場合 | 低 |

### 1.2 文字列・テキスト関連

| 型 | 説明 | 優先度 |
|---|---|---|
| `Char` | 単独の文字型（`char_at`は返すが、型としての操作が不足） | **高** |
| `StringBuilder` / `StringBuffer` | 効率的な文字列結合 | 中 |
| `Regex` | 正規表現 | 中 |

### 1.3 並行処理

| 型 | 説明 | 優先度 |
|---|---|---|
| `Thread` / `Task` | スレッド / 非同期タスク | **高** |
| `Mutex<T>` / `Lock<T>` | 排他制御 | **高** |
| `Channel<T>` | スレッド間通信 | 中 |
| `Atomic<T>` | アトミック操作 | 中 |

### 1.4 その他

| 型 | 説明 | 優先度 |
|---|---|---|
| `Duration` / `Instant` | 時間の表現（`System::time()`はf32を返すのみ） | 中 |
| `Date` / `DateTime` | 日付・時刻 | 中 |
| `Range<T>` を第一級の型として | for文で使えるが、変数に格納・関数に渡す等ができるか | 中 |
| `Error` トレイト / 基底型 | エラー型の統一的なインターフェース | 中 |

---

## 2. 不足している組み込み関数・メソッド

### 2.1 String メソッド（現状: `len`, `contains`, `concat`, `char_at`, `to_i64`, `print`, `display`）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `split(sep: String) -> Vec<String>` | 文字列分割 | **高** |
| `trim() -> String` | 前後の空白除去 | **高** |
| `starts_with(s: String) -> bool` | 前方一致 | **高** |
| `ends_with(s: String) -> bool` | 後方一致 | **高** |
| `replace(from, to) -> String` | 文字列置換 | **高** |
| `to_uppercase() -> String` | 大文字化 | 中 |
| `to_lowercase() -> String` | 小文字化 | 中 |
| `substring(start, len) -> String` | 部分文字列 | **高** |
| `index_of(s: String) -> i64` | 検索（位置を返す） | 中 |
| `to_f64() -> f64` | 浮動小数点パース | 中 |
| `repeat(n: i64) -> String` | 繰り返し | 低 |
| `is_empty() -> bool` | 空文字列判定 | **高** |
| `chars() -> Vec<Char>` | 文字列の文字配列化 | 中 |
| `format(...)` | フォーマット文字列（変数を返す形） | 中 |

### 2.2 Vec メソッド（現状: `len`, `capacity`, `is_empty`, `push`, `pop`, `get`, `set`）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `insert(index, item)` | 任意位置への挿入 | **高** |
| `remove(index) -> T` | 任意位置の削除 | **高** |
| `contains(item) -> bool` | 要素の存在チェック | **高** |
| `index_of(item) -> i64` | 要素の検索 | 中 |
| `sort()` / `sort_by(fn)` | ソート | **高** |
| `reverse()` | 逆順 | 中 |
| `clear()` | 全要素削除 | **高** |
| `extend(other: Vec<T>)` | 結合 | 中 |
| `slice(start, end) -> Vec<T>` | 部分取得 | 中 |
| `map(fn) -> Vec<U>` | 変換 | **高** |
| `filter(fn) -> Vec<T>` | フィルタリング | **高** |
| `reduce(fn) -> T` / `fold` | 畳み込み | 中 |
| `any(fn) -> bool` | いずれかが条件を満たすか | 中 |
| `all(fn) -> bool` | すべてが条件を満たすか | 中 |
| `enumerate() -> Vec<(i64, T)>` | インデックス付き走査 | 中 |
| `join(sep: String) -> String` | 文字列結合（`Vec<String>`用） | 中 |
| `flatten() -> Vec<T>` | ネスト解除 | 低 |
| `zip(other) -> Vec<(T, U)>` | 2つのVecの結合 | 低 |
| `clone() -> Vec<T>` | 複製 | 中 |

### 2.3 HashMap メソッド（現状: `len`, `is_empty`, `insert`, `get`, `remove`(未実装)）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `remove(key)` の実装完了 | ドキュメントに記載済みだが未実装 | **高** |
| `contains_key(key) -> bool` | キーの存在チェック | **高** |
| `keys() -> Vec<K>` | キー一覧 | **高** |
| `values() -> Vec<V>` | 値一覧 | **高** |
| `entries() -> Vec<(K, V)>` | エントリ一覧 | 中 |
| `clear()` | 全エントリ削除 | 中 |
| `get_or_default(key, default) -> V` | デフォルト付き取得 | 中 |

### 2.4 Option メソッド（現状: `is_some`, `is_none`, `unwrap`, `unwrap_or`）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `map(fn) -> Option<U>` | 値の変換 | **高** |
| `and_then(fn) -> Option<U>` | チェーン | 中 |
| `or(other) -> Option<T>` | 代替値 | 中 |
| `unwrap_or_else(fn) -> T` | 遅延デフォルト | 低 |
| `?` 演算子のサポート | `Result`では対応済み、`Option`でも必要 | **高** |

### 2.5 Result メソッド（現状: `is_ok`, `is_err`, `unwrap`, `unwrap_err`, `?`）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `map(fn) -> Result<U, E>` | 成功値の変換 | **高** |
| `map_err(fn) -> Result<T, F>` | エラー値の変換 | 中 |
| `and_then(fn) -> Result<U, E>` | チェーン | 中 |
| `unwrap_or(default) -> T` | デフォルト付き | 中 |
| `unwrap_or_else(fn) -> T` | 遅延デフォルト | 低 |

### 2.6 数値型メソッドの不足

| メソッド | 対象 | 説明 | 優先度 |
|---|---|---|---|
| `min(a, b)` / `max(a, b)` | グローバル or i64/i32 | 整数の最小・最大 | **高** |
| `clamp(val, min, max)` | i64/i32 | 整数のクランプ | 中 |
| `to_f32()` / `to_f64()` | i64/i32 | 型変換 | **高** |
| `to_i32()` / `to_i64()` | f32/f64 | 型変換 | **高** |
| `to_string()` | 全数値型 | 文字列変換 | **高** |
| `parse<T>(s: String)` | グローバル | 汎用パース関数 | 中 |
| `NaN`, `INFINITY` | f32/f64 | 特殊値定数 | 中 |
| `is_nan()`, `is_inf()` | f32/f64 | 特殊値判定 | 中 |
| `MAX`, `MIN` | 全数値型 | 型の最大・最小値定数 | 中 |

### 2.7 グローバル関数 / System の不足

| 関数 | 説明 | 優先度 |
|---|---|---|
| `System::exit(code: i64)` | プロセス終了（KIには記載あり、ドキュメント未記載） | **高** |
| `System::env()` / `System::platform()` | OS情報取得 | 中 |
| `System::command(cmd: String) -> String` | 外部コマンド実行 | 中 |
| `assert(cond, msg?)` | テスト・デバッグ用アサーション | **高** |
| `typeof(value) -> String` | 型情報の取得 | 中 |
| `random() -> f64` | 乱数生成（Tensor::randnはあるが、スカラーの乱数がない） | **高** |
| `random_int(min, max) -> i64` | 整数乱数 | **高** |

### 2.8 File / Path の不足

| メソッド | 説明 | 優先度 |
|---|---|---|
| `File::append(path, content)` | 追記 | 中 |
| `File::delete(path) -> bool` | ファイル削除 | 中 |
| `File::list_dir(path) -> Vec<String>` | ディレクトリ一覧 | 中 |
| `File::create_dir(path) -> bool` | ディレクトリ作成 | 中 |
| `Path::parent() -> Path` | 親ディレクトリ | 中 |
| `Path::file_name() -> String` | ファイル名取得 | 中 |
| `Path::extension() -> String` | 拡張子取得 | 低 |

---

## 3. 不足している言語機能（型・関数に関連するもの）

| 機能 | 説明 | 優先度 |
|---|---|---|
| クロージャ / ラムダ | `map`, `filter`, `sort_by`等の高階関数に必須 | **高** |
| ジェネリックなトレイト制約 | `where T: Comparable` のような制約 | **高** |
| イテレータプロトコルの拡充 | `Iterable`は`len`/`get`ベースだが、`next()`ベースの遅延イテレータが必要 | 中 |
| 型変換トレイト (`From`/`Into`) | 型間の変換を統一的に扱う | 中 |
| `Display` / `Debug` トレイト | `print`/`println`でのカスタム表示 | 中 |
| 演算子オーバーロード | ユーザ定義型での `+`, `-` 等 | 中 |
| ユーザ定義列挙型(enum) | `Option`/`Result`以外のenum定義 | **高** |

---

## 4. 優先度サマリ

### 最優先（汎用言語として最低限必要）
1. **String**: `split`, `trim`, `starts_with`, `ends_with`, `replace`, `substring`, `is_empty`
2. **Vec**: `insert`, `remove`, `contains`, `sort`, `clear`, `map`, `filter`
3. **HashMap**: `remove`の実装完了, `contains_key`, `keys`, `values`
4. **数値型変換**: `to_string()`, `to_f64()`, `to_i64()` 等
5. **グローバル**: `assert`, `random`, `random_int`, `min`/`max`
6. **型**: `Set<T>`, `Tuple`, スレッド/並行処理の基盤
7. **Option**: `map`, `?`演算子サポート
8. **Result**: `map`
9. **言語機能**: クロージャ/ラムダ, ユーザ定義enum

### 中優先（実用的なアプリケーション開発に必要）
- String: `to_uppercase`, `to_lowercase`, `index_of`, `to_f64`, `chars`
- Vec: `reverse`, `extend`, `slice`, `reduce`, `any`, `all`, `join`, `clone`
- HashMap: `entries`, `clear`, `get_or_default`
- Option/Result: `and_then`, `or`, `map_err`, `unwrap_or`
- File/Path: `delete`, `list_dir`, `create_dir`, `parent`, `file_name`
- 型: `Deque`, `Duration`, `DateTime`, `Regex`, `StringBuilder`
- 言語: イテレータプロトコル拡充, 型変換トレイト, `Display`/`Debug`

### 低優先（あれば便利）
- Vec: `flatten`, `zip`
- String: `repeat`
- Option: `unwrap_or_else`
- Path: `extension`
- 型: `SortedMap`/`BTreeMap`
