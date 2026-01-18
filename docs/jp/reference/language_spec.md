# TensorLogic 言語仕様

TensorLogic (TL) は、高性能なテンソル演算と論理プログラミングのために設計された、LLVMへJITコンパイルされる静的型付けプログラミング言語です。

## 1. コメント

```rust
// これは1行コメントです
/* これは
   複数行コメントです */
```

## 2. データ型

### プリミティブ型
*   **`f32`**: 32ビット浮動小数点数。
*   **`f64`**: 64ビット浮動小数点数。
*   **`i64`**: 64ビット符号付き整数。
*   **`bool`**: ブール値 (`true` または `false`)。
*   **`String`**: UTF-8 文字列。

### Tensor（テンソル）
中核となるデータ構造です。テンソルは多次元配列です。
```rust
let t: Tensor = Tensor::zeros([2, 2], true);
```

### Structs（構造体）
ユーザー定義の複合型です。
```rust
struct Point {
    x: f32,
    y: f32,
}
```

### Enums（列挙型）
バリアントを定義するためのタグ付き共用体です。
```rust
enum Option {
    Some { value: i64 },
    None,
}
```

## 3. 変数

変数はデフォルトでバインディングという意味では不変（Immutable）ですが、Tensorのようなオブジェクトはメソッドを通じて状態を変更できる場合があります。シャドーイング（同名変数による再定義）は許可されています。

```rust
let x = 10;
let x = x + 1; // シャドーイング
```

## 4. 関数

関数は第一級オブジェクトです。

```rust
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

## 5. 制御フロー

### If 式
```rust
let result = if x > 0 { 1 } else { 0 };
```

### While ループ
```rust
while i < 10 {
    print(i);
    i = i + 1;
}
```

### For ループ
範囲（Range）を使用したループ：
```rust
for i in 0..10 {
    print(i);
}
```

### ループ制御 (Loop Control)
ループ内で `break` と `continue` が使用可能です。

```rust
for i in 0..10 {
    if i == 5 { continue; } // 次の反復へスキップ
    if i == 8 { break; }    // ループから脱出
    print(i);
}
```

## 6. テンソル内包表記 (Tensor Comprehension)

テンソルを作成するための強力な構文です。

```rust
// 構文: [indices | clauses { body }]
let A = [i, j | i <- 0..5, j <- 0..5 { i + j }];
```

## 7. クラスとメソッド

メソッドは `impl` ブロックを使用して Struct や Enum に定義できます。

```rust
impl Point {
    fn distance(self) -> f32 {
        (self.x.pow(2.0) + self.y.pow(2.0)).sqrt()
    }
}
```
