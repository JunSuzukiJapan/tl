# TL言語 汎用言語化に向けた不足分析

> [!NOTE]
> **ステータス更新 (2026-03)**: 多くの項目が実装完了済み。
> 各テーブルの「状態」列で実装状況を確認可能。
> TL固有のドメイン（Tensor, Param, KVCache, VarBuilder, Map, Tokenizer, Image 等）は対象外。

---

## 1. 不足している組み込み型

### 1.1 コレクション型

| 型 | 説明 | 状態 |
|---|---|---|
| `Set<T>` / `HashSet<T>` | 一意な要素の集合 | ❌ 未実装 |
| `Tuple<T1, T2, ...>` | 異なる型の固定長組 | ✅ 実装済み |
| `Deque<T>` / `VecDeque<T>` | 両端キュー | ❌ 未実装 |
| `SortedMap<K, V>` / `BTreeMap<K, V>` | 順序付きマップ | ❌ 未実装 |

### 1.2 文字列・テキスト関連

| 型 | 説明 | 状態 |
|---|---|---|
| `Char` | 単独の文字型 | ✅ 実装済み |
| `StringBuilder` / `StringBuffer` | 効率的な文字列結合 | ❌ 未実装（`format()` で代替可能） |
| `Regex` | 正規表現 | ❌ 未実装 |

### 1.3 並行処理

| 型 | 説明 | 状態 |
|---|---|---|
| `Thread` / `Task` | スレッド / 非同期タスク | ❌ 未実装 |
| `Mutex<T>` / `Lock<T>` | 排他制御 | ❌ 未実装 |
| `Channel<T>` | スレッド間通信 | ❌ 未実装 |
| `Atomic<T>` | アトミック操作 | ❌ 未実装 |

### 1.4 その他

| 型 | 説明 | 状態 |
|---|---|---|
| `Duration` / `Instant` | 時間の表現 | ❌ 未実装（`System::time()` はf32を返すのみ） |
| `Date` / `DateTime` | 日付・時刻 | ❌ 未実装 |
| `Range<T>` を第一級の型として | 変数に格納可能 | ✅ for文 + イテレータプロトコルで対応 |
| `Error` トレイト / 基底型 | エラー型の統一インターフェース | ❌ 未実装 |

---

## 2. 組み込み関数・メソッド

### 2.1 String メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `split(sep) -> Vec<String>` | 文字列分割 | ✅ 実装済み |
| `trim() -> String` | 前後の空白除去 | ✅ 実装済み |
| `starts_with(s) -> bool` | 前方一致 | ✅ 実装済み |
| `ends_with(s) -> bool` | 後方一致 | ✅ 実装済み |
| `replace(from, to) -> String` | 文字列置換 | ✅ 実装済み |
| `to_uppercase() -> String` | 大文字化 | ✅ 実装済み |
| `to_lowercase() -> String` | 小文字化 | ✅ 実装済み |
| `substring(start, len) -> String` | 部分文字列 | ✅ 実装済み |
| `index_of(s) -> i64` | 検索（位置を返す） | ✅ 実装済み |
| `to_f64() -> f64` | 浮動小数点パース | ✅ 実装済み |
| `repeat(n) -> String` | 繰り返し | ✅ 実装済み |
| `is_empty() -> bool` | 空文字列判定 | ✅ 実装済み |
| `chars() -> Vec<Char>` | 文字列の文字配列化 | ✅ 実装済み |
| `format(...)` | フォーマット文字列 | ✅ 実装済み（グローバル関数） |

### 2.2 Vec メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `insert(index, item)` | 任意位置への挿入 | ✅ 実装済み (vec.tl) |
| `remove(index) -> T` | 任意位置の削除 | ✅ 実装済み (vec.tl) |
| `contains(item) -> bool` | 要素の存在チェック | ✅ 実装済み (vec.tl) |
| `sort()` | ソート | ✅ 実装済み (vec.tl) |
| `clear()` | 全要素削除 | ✅ 実装済み (vec.tl) |
| `index_of(item) -> i64` | 要素の検索 | ❌ 未実装 |
| `reverse()` | 逆順 | ❌ 未実装 |
| `extend(other)` | 結合 | ❌ 未実装 |
| `slice(start, end) -> Vec<T>` | 部分取得 | ❌ 未実装 |
| `map(fn) -> Vec<U>` | 変換 | ⚠️ TODOコメント（ジェネリック特殊化要） |
| `filter(fn) -> Vec<T>` | フィルタリング | ⚠️ TODOコメント（ジェネリック特殊化要） |
| `reduce(fn) -> T` / `fold` | 畳み込み | ❌ 未実装 |
| `any(fn) -> bool` | いずれかが条件を満たすか | ❌ 未実装 |
| `all(fn) -> bool` | すべてが条件を満たすか | ❌ 未実装 |
| `enumerate()` | インデックス付き走査 | ❌ 未実装 |
| `join(sep) -> String` | 文字列結合 | ❌ 未実装 |
| `flatten()` | ネスト解除 | ❌ 未実装 |
| `zip(other)` | 2つのVecの結合 | ❌ 未実装 |
| `clone() -> Vec<T>` | 複製 | ❌ 未実装 |

### 2.3 HashMap メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `remove(key)` | キーの削除 | ✅ 実装済み (hashmap.tl) |
| `contains_key(key) -> bool` | キーの存在チェック | ✅ 実装済み (hashmap.tl) |
| `keys() -> Vec<K>` | キー一覧 | ✅ 実装済み (hashmap.tl) |
| `values() -> Vec<V>` | 値一覧 | ✅ 実装済み (hashmap.tl) |
| `entries() -> Vec<(K, V)>` | エントリ一覧 | ❌ 未実装（タプル対応が複雑） |
| `clear()` | 全エントリ削除 | ❌ 未実装 |
| `get_or_default(key, default) -> V` | デフォルト付き取得 | ❌ 未実装 |

### 2.4 Option メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `unwrap_or(default) -> T` | デフォルト付き展開 | ✅ 実装済み (option.tl) |
| `?` 演算子 | 早期リターン | ✅ 実装済み |
| `map(fn) -> Option<U>` | 値の変換 | ⚠️ TODOコメント（ジェネリック特殊化要） |
| `and_then(fn) -> Option<U>` | チェーン | ❌ 未実装 |
| `or(other) -> Option<T>` | 代替値 | ❌ 未実装 |
| `unwrap_or_else(fn) -> T` | 遅延デフォルト | ❌ 未実装 |

### 2.5 Result メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `?` 演算子 | 早期リターン | ✅ 実装済み |
| `unwrap()` / `unwrap_err()` | 展開 | ✅ 実装済み |
| `is_ok()` / `is_err()` | 判定 | ✅ 実装済み |
| `map(fn)` | 成功値の変換 | ❌ 未実装 |
| `map_err(fn)` | エラー値の変換 | ❌ 未実装 |
| `and_then(fn)` | チェーン | ❌ 未実装 |
| `unwrap_or(default)` | デフォルト付き | ❌ 未実装 |
| `unwrap_or_else(fn)` | 遅延デフォルト | ❌ 未実装 |

### 2.6 数値型メソッド

| メソッド | 対象 | 状態 |
|---|---|---|
| `min(a, b)` / `max(a, b)` | グローバル or 数値型 | ✅ 実装済み |
| `clamp(val, min, max)` | 数値型 | ✅ 実装済み |
| `to_f32()` / `to_f64()` / `to_i32()` / `to_i64()` | 数値型間変換 | ✅ 実装済み |
| `to_string()` | 全数値型 | ✅ 実装済み (`format()` 経由) |
| `abs()` / `sqrt()` / `pow()` | 数値型 | ✅ 実装済み |
| `NaN`, `INFINITY` | f32/f64 | ❌ 未実装 |
| `is_nan()`, `is_inf()` | f32/f64 | ❌ 未実装 |
| `MAX`, `MIN` | 全数値型 | ❌ 未実装 |

### 2.7 グローバル関数 / System

| 関数 | 説明 | 状態 |
|---|---|---|
| `println(...)` | コンソール出力 | ✅ 実装済み |
| `format(...)` | フォーマット文字列 | ✅ 実装済み |
| `assert(cond, msg?)` | アサーション | ❌ 未実装 |
| `typeof(value)` | 型情報取得 | ❌ 未実装 |
| `random() -> f64` | スカラー乱数 | ❌ 未実装 |
| `random_int(min, max) -> i64` | 整数乱数 | ❌ 未実装 |
| `System::exit(code)` | プロセス終了 | ❌ 未実装 (コンパイラ未登録) |
| `System::env()` / `System::platform()` | OS情報取得 | ❌ 未実装 |
| `System::command(cmd)` | 外部コマンド実行 | ❌ 未実装 |

### 2.8 File / Path

| メソッド | 説明 | 状態 |
|---|---|---|
| `File::read(path)` | ファイル読み込み | ✅ 実装済み |
| `File::write(path, content)` | ファイル書き込み | ✅ 実装済み |
| `File::append(path, content)` | 追記 | ❌ 未実装 |
| `File::delete(path)` | ファイル削除 | ❌ 未実装 |
| `File::list_dir(path)` | ディレクトリ一覧 | ❌ 未実装 |
| `File::create_dir(path)` | ディレクトリ作成 | ❌ 未実装 |
| `Path::exists(path)` | ファイル存在チェック | ✅ 実装済み |
| `Path::parent()` | 親ディレクトリ | ❌ 未実装 |
| `Path::file_name()` | ファイル名取得 | ❌ 未実装 |
| `Path::extension()` | 拡張子取得 | ❌ 未実装 |

---

## 3. 不足している言語機能（型・関数に関連するもの）

| 機能 | 説明 | 状態 |
|---|---|---|
| クロージャ / ラムダ | 高階関数に必須 | ✅ 実装済み（`Fn(T) -> U` 型） |
| ジェネリックなトレイト制約 | `where T: Comparable` | ✅ 実装済み |
| イテレータプロトコル | `len`/`get` ベースの Duck-typing | ✅ 実装済み（Range/Vec/Tensor統一） |
| 型変換トレイト (`From`/`Into`) | 統一的な型変換 | ❌ 未実装 |
| `Display` / `Debug` トレイト | カスタム表示 | ❌ 未実装 |
| 演算子オーバーロード | ユーザ定義型での `+`, `-` 等 | ❌ 未実装 |
| ユーザ定義列挙型(enum) | `Option`/`Result`以外のenum定義 | ✅ 実装済み |

---

## 4. 優先度サマリ（未実装項目）

### 次に取り組むべき項目
1. **Vec**: `map`, `filter`（ジェネリック特殊化エンジンの `Fn(T) -> U` 対応が前提）
2. **Option/Result**: `map`, `and_then`（同上）
3. **Vec**: `reverse`, `extend`, `slice`, `index_of`, `join`
4. **HashMap**: `entries`, `clear`, `get_or_default`
5. **グローバル**: `assert`, `random`, `random_int`
6. **数値型**: `NaN`, `INFINITY`, `is_nan()`, `MAX`/`MIN` 定数

### 将来の拡張
- 型: `Set<T>`, `Deque<T>`, `Duration`, `DateTime`, `Regex`
- 並行処理: `Thread`, `Mutex`, `Channel`
- 言語: `From`/`Into` トレイト, `Display`/`Debug`, 演算子オーバーロード
- File/Path: `append`, `delete`, `list_dir`, `create_dir`, `parent`, `file_name`
