# TL言語 汎用言語化に向けた不足分析

> [!NOTE]
> **ステータス更新 (2026-04 最新)**: 多くの項目（コレクション、並行処理、文字列処理など）が実装完了済みのため、ドキュメントから削除されました。
> このリストは**現在本当に未実装である項目**のみを追跡しています。

---

## 1. 不足している組み込み型

### 1.1 並行処理

| 型 | 説明 | 状態 |
|---|---|---|
| `Channel<T>` | スレッド間通信 | ❌ 未実装 |
| `Atomic<T>` | アトミック操作 | ❌ 未実装 |

### 1.2 その他

| 型 | 説明 | 状態 |
|---|---|---|
| `Duration` / `Instant` | 時間の表現 | ❌ 未実装（`System::time()` はf32を返すのみ） |
| `Date` / `DateTime` | 日付・時刻 | ❌ 未実装 |
| `Error` トレイト / 基底型 | エラー型の統一インターフェース | ❌ 未実装 |

---

## 2. 組み込み関数・メソッドの不足

### 2.1 Vec メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `index_of(item) -> i64` | 要素の検索 | ❌ 未実装 |
| `reverse()` | 逆順 | ❌ 未実装 |
| `extend(other)` | 結合 | ❌ 未実装 |
| `slice(start, end) -> Vec<T>` | 部分取得 | ❌ 未実装 |
| `reduce(fn) -> T` / `fold` | 畳み込み | ❌ 未実装 |
| `enumerate()` | インデックス付き走査 | ❌ 未実装 |
| `join(sep) -> String` | 文字列結合 | ❌ 未実装 |
| `flatten()` | ネスト解除 | ❌ 未実装 |
| `zip(other)` | 2つのVecの結合 | ❌ 未実装 |
| `clone() -> Vec<T>` | 複製 | ❌ 未実装 |

### 2.2 HashMap メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `entries() -> Vec<(K, V)>` | エントリ一覧 | ❌ 未実装（タプル対応が複雑） |
| `clear()` | 全エントリ削除 | ❌ 未実装 |
| `get_or_default(key, default) -> V` | デフォルト付き取得 | ❌ 未実装 |

### 2.3 Option メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `and_then(fn) -> Option<U>` | チェーン | ❌ 未実装 |
| `or(other) -> Option<T>` | 代替値 | ❌ 未実装 |
| `unwrap_or_else(fn) -> T` | 遅延デフォルト | ❌ 未実装 |

### 2.4 Result メソッド

| メソッド | 説明 | 状態 |
|---|---|---|
| `map(fn)` | 成功値の変換 | ❌ 未実装 |
| `map_err(fn)` | エラー値の変換 | ❌ 未実装 |
| `and_then(fn)` | チェーン | ❌ 未実装 |
| `unwrap_or(default)` | デフォルト付き | ❌ 未実装 |
| `unwrap_or_else(fn)` | 遅延デフォルト | ❌ 未実装 |

### 2.5 数値型メソッド

| メソッド | 対象 | 状態 |
|---|---|---|
| `NaN`, `INFINITY` | f32/f64 | ❌ 未実装 |
| `is_nan()`, `is_inf()` | f32/f64 | ❌ 未実装 |
| `MAX`, `MIN` | 全数値型 | ❌ 未実装 |

### 2.6 グローバル関数 / System

| 関数 | 説明 | 状態 |
|---|---|---|
| `typeof(value)` | 型情報取得 | ❌ 未実装 |
| `System::exit(code)` | プロセス終了 | ❌ 未実装 (コンパイラ未登録) |
| `System::env()` / `System::platform()` | OS情報取得 | ❌ 未実装 |
| `System::command(cmd)` | 外部コマンド実行 | ❌ 未実装 |

### 2.7 File / Path

| メソッド | 説明 | 状態 |
|---|---|---|
| `File::append(path, content)` | 追記 | ❌ 未実装 |
| `File::delete(path)` | ファイル削除 | ❌ 未実装 |
| `File::list_dir(path)` | ディレクトリ一覧 | ❌ 未実装 |
| `File::create_dir(path)` | ディレクトリ作成 | ❌ 未実装 |
| `Path::parent()` | 親ディレクトリ | ❌ 未実装 |
| `Path::file_name()` | ファイル名取得 | ❌ 未実装 |
| `Path::extension()` | 拡張子取得 | ❌ 未実装 |

---

## 3. 不足している言語機能

| 機能 | 説明 | 状態 |
|---|---|---|
| 型変換トレイト (`From`/`Into`) | 統一的な型変換 | ❌ 未実装 |
| `Display` / `Debug` トレイト | カスタム表示 | ❌ 未実装 |
| 演算子オーバーロード | ユーザ定義型での `+`, `-` 等 | ❌ 未実装 |

---

## 4. 優先度サマリ（未実装項目）

### 次に取り組むべき項目
1. **Vec**: `slice`, `extend`, `reverse`, `index_of`
2. **Option/Result**: `and_then`, `unwrap_or_else`
3. **HashMap**: `entries`, `clear`

### 将来の拡張
- 型: `Channel<T>`, `Atomic<T>`, `Duration`, `DateTime`
- 言語機能: `From`/`Into` トレイト, `Display`/`Debug`, 演算子オーバーロード
- System/IO: `System::env()`, コマンド実行, `File::list_dir` などの OS機能連携
