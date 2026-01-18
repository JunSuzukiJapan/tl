# 構文シュガーを使った家系図

この例は、TensorLogicのProlog風の論理プログラミング構文シュガーを示しています。

## ファイル

- `facts.tl`: 事実（父親関係）とルール（祖父母、祖先）を含みます。
- `main.tl`: 事実をインポートして論理クエリを実行するメインプログラム。

## 構文のポイント

### 事実（Facts）
```
father(alice, bob).
father(bob, charlie).
```

### ルール（Rules）
```
grandparent(g, c) :- father(g, f), father(f, c).
ancestor(a, d) :- father(a, d).
ancestor(a, d) :- father(a, x), ancestor(x, d).
```

### クエリ（Queries）
```rust
let res = @father(alice, bob)?;       // ブール型クエリ
let res = @ancestor(alice, $x)?;      // 変数クエリ（$xは未束縛）
```

### モジュールシステム
```rust
mod facts;        // facts.tl をサブモジュールとして読み込む
use facts::*;     // すべての関係をインポート
```

## 実行方法

```bash
cd examples/tasks/logic/family_tree_with_syntax_sugar
tl main.tl
```
