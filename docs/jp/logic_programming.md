# TensorLanguage (TL) における論理プログラミング

TLは、強力なProlog風の論理推論エンジンをランタイム（テンソル計算エンジン）に直接統合しています。これにより、知識ベースを定義し、論理的推論を実行し、その結果（シンボル推論）をニューラルネットワークや数値計算とシームレスに組み合わせることが可能です。

## 1. 構文の概要

TLでは、論理ステートメントが第一級市民として扱われます。**事実 (Facts)**、**ルール (Rules)**、**クエリ (Queries)** を定義できます。

### 事実 (Facts)
事実は静的な知識を宣言します。述語と引数（エンティティまたは値）で構成されます。

```rust
// 構文シュガー (推奨)
father(alice, bob).       // "alice は bob の父親である"
is_student(charlie).      // 単項述語

// オプションの @ プレフィックス (レガシー/明示的)
@father(bob, diana).
```

### ルール (Rules)
ルールは、既存の事実から新しい事実を導き出す方法を定義します。**本体 (body)**（右辺）が真であれば、**頭部 (head)**（左辺）も真であると推論されます。

```rust
// "x が y の父であり、かつ y が z の父であれば、x は z の祖父である"
grandparent(x, z) :- father(x, y), father(y, z).

// 再帰的なルールもサポートされています
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

- ルール内の変数は小文字で始めます（慣習的に `x`, `y`, `z`）。
- `,` は論理積 (AND) を表します。
- `.` でステートメントを終了します。

### クエリ (Queries)
`?` サフィックスを使用して、知識ベース (KB) に問い合わせることができます。クエリの結果はテンソルとして返されます。

```rust
// 1. 真偽値クエリ (True/False)
// 0次元テンソルを返します: [1.] (真) または [0.] (偽)
let is_father = @father(alice, bob)?; 
println("Is alice father of bob? {}", is_father);

// 2. 変数クエリ (検索)
// $変数名 を使用して「誰が？」または「何を？」を問います。マッチしたリストを返します。
// 結果: 形状 [N, 1] のテンソル。エンティティID（表示時は名前）を含みます。
let children = @father(alice, $child)?;
println("Children of alice: {}", children);
```

## 2. シンボル出力

TLは、エンティティ名（シンボル）を内部的に一意の整数IDに自動的にマッピングします。論理クエリの結果（エンティティIDを含むテンソル）を表示する際、ランタイムは自動的にこれらのIDを元の名前に解決して表示します。

```rust
father(alice, bob).

fn main() {
    println("{}", @father(alice, $x)?);
    // 出力:
    // [[bob]]
}
```

## 3. スコープとファイル構成

事実とルールは **グローバルスコープ**（関数の外）で定義する必要があります。関数の中で定義することはできません。

### 単一ファイル (スクリプト形式)
事実、ルール、`main` 関数をすべて同じファイルに記述できます。

```rust
// main.tl
father(alice, bob).
grandparent(x, z) :- father(x, y), father(y, z).

fn main() {
    let res = @grandparent(alice, $x)?;
    println("{}", res);
}
```

### 外部ファイル (モジュール形式)
ロジックを別々のファイルに整理することも可能です。コンパイラは、すべてのインポートされたモジュールから事実とルールを自動的に収集します。

**facts.tl**:
```rust
father(alice, bob).
father(bob, charlie).
```

**logic.tl**:
```rust
// ルールも別のファイルに置くことができます
grandparent(x, z) :- father(x, y), father(y, z).
```

**main.tl**:
```rust
mod facts;
mod logic;

// 現在のスコープに関係やルールをインポートします
use facts::*;
use logic::*;

fn main() {
    // 'facts.tl' の事実と 'logic.tl' のルールは自動的に読み込まれます
    let res = @grandparent(alice, $x)?;
    println("{}", res);
}
```

## 4. テンソルとの統合

クエリ結果は標準的な TL のテンソルであるため、数式やニューラルネットワークの演算にそのまま使用できます。

- **真偽値**: `0.0` または `1.0` (float)。マスキングや条件付きロジックに便利です。
- **検索結果**: エンティティIDの `Int64` テンソル。Embedding層のインデックスとして使用できます。

例: ニューロ・シンボリック統合
```rust
// 論理: すべての先祖を検索
let ancestors = @ancestor(alice, $x)?;

// ニューラル: これらの先祖の埋め込みベクトルを取得
let embeds = embedding(ancestors, weights);

// 平均ベクトルを計算
let query_vec = mean(embeds, 0);
```

## 4. 高度な機能

### 推移的・再帰的論理
TLはDatalogスタイルの完全な再帰をサポートしています。グラフの到達可能性や推移閉包などの複雑な関係を定義できます。

```rust
path(x, y) :- edge(x, y).
path(x, y) :- edge(x, z), path(z, y).
```

### 多重依存
ルールには複数の条件を持たせることができます。すべての条件が満たされた場合にのみ、ルールが成立します。

```rust
compatible(x, y) :- is_friend(x, y), has_same_hobby(x, y).
```

### 自動 KB 初期化
知識ベースを手動で初期化する必要はありません。コンパイラは、すべてのモジュール（インポートされたものを含む）から事実とルールを自動的に集約し、`main()` の開始時に推論エンジンを初期化します。
