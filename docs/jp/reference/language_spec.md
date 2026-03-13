# TensorLogic 言語仕様

TensorLogic (TL) は、高性能なテンソル演算と論理プログラミングのために設計された、LLVMへJITコンパイルされる静的型付けプログラミング言語です。Rustに似た構文を持ちます。

## 1. コメント

```rust
// これは1行コメントです
/* これは
   複数行コメントです */
```

## 2. データ型

### プリミティブ型
| 型 | 説明 |
|---|---|
| `f32` | 32ビット浮動小数点数 |
| `f64` | 64ビット浮動小数点数 |
| `i8`, `i16`, `i32`, `i64` | 符号付き整数 |
| `u8`, `u16`, `u32`, `u64` | 符号なし整数 |
| `usize` | ポインタサイズの符号なし整数 |
| `bool` | ブール値 (`true` / `false`) |
| `String` | UTF-8 文字列 |
| `Char` | 単一文字 (`'a'`) |

### Tensor（テンソル）
多次元配列を表す中核データ構造です。勾配追跡の有無を型で区別します。
```rust
let t: Tensor = Tensor::zeros([2, 2], false);       // 勾配なし（推論用）
let g: GradTensor = Tensor::zeros([2, 2], true);     // 勾配あり（学習用）
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
タグ付き共用体です。Unit / Tuple / Struct の3種類のバリアントをサポートします。
```rust
enum Shape {
    Circle(f32),           // Tuple バリアント
    Rectangle(f32, f32),   // Tuple バリアント
    Point,                 // Unit バリアント
}
```

### タプル型
```rust
let pair: (i64, f32) = (42, 3.14);
let x = pair.0;  // タプルアクセス
```

### 固定長配列
```rust
let arr: [i64; 3] = [1, 2, 3];
```

### ジェネリック型

TLはRustスタイルのジェネリクスをサポートします。

```rust
struct Pair<A, B> {
    first: A,
    second: B,
}
```

組み込みジェネリック型:
- `Vec<T>` — 可変長配列
- `HashMap<K, V>` — ハッシュマップ
- `Option<T>` — `Some(T)` または `None`
- `Result<T, E>` — `Ok(T)` または `Err(E)`

### Fn 型（関数 / クロージャ型）
関数やクロージャの型を表現します。
```rust
fn apply(f: Fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}
```

## 3. 変数

### 不変変数（デフォルト）
```rust
let x = 10;
let x = x + 1; // シャドーイングは可能
```

### ミュータブル変数
```rust
let mut count = 0;
count = count + 1;  // 再代入が可能
```

### 複合代入演算子
```rust
count += 1;
count -= 1;
count *= 2;
count /= 2;
count %= 3;
```

## 4. 関数

```rust
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

### ジェネリック関数
```rust
fn identity<T>(x: T) -> T {
    x
}
```

### 公開可視性
```rust
pub fn public_function() { }
pub struct PublicStruct { pub field: i64 }
```

## 5. クロージャ（無名関数）

TLはRustスタイルのクロージャをサポートします。外側のスコープの変数をキャプチャできます。

### 基本構文

```rust
// 引数なし
let greet = || println("Hello!");
greet();

// 単一式のクロージャ
let double = |x: i64| x * 2;
println("{}", double(5)); // 10

// ブロック本体
let complex = |x: i64, y: i64| -> i64 {
    let sum = x + y;
    sum * 2
};
```

### 高階関数への受け渡し

```rust
let numbers = Vec::new();
numbers.push(1);
numbers.push(2);
numbers.push(3);

// map: 各要素に関数を適用
let doubled = numbers.map(|x: i64| -> i64 { x * 2 });

// filter: 条件を満たす要素のみ残す
let evens = numbers.filter(|x: i64| -> bool { x % 2 == 0 });
```

### 変数キャプチャ

```rust
let factor = 3;
let multiply = |x: i64| -> i64 { x * factor };  // factor をキャプチャ
println("{}", multiply(5)); // 15
```

### 型注釈

引数の型注釈は省略可能です（型推論で解決される場合）。戻り値の型も省略できます。

```rust
// 型注釈あり
let add = |x: i64, y: i64| -> i64 { x + y };

// 型注釈なし（型推論）
let add = |x, y| x + y;
```

## 6. 制御フロー

### If 式
```rust
let result = if x > 0 { 1 } else { 0 };
```

### While ループ
```rust
while i < 10 {
    i += 1;
}
```

### For ループ
イテレータプロトコル（`len` + `get` メソッド）をサポートする型に対して使用できます。
```rust
// Range
for i in 0..10 { print(i); }

// 半開区間 (start..)
for i in 0..5 { print(i); }

// Vec
let v = Vec::new();
v.push(1); v.push(2);
for item in v { print(item); }

// Tensor
let t = Tensor::zeros([5], false);
for val in t { print(val); }
```

### loop ループ（無限ループ）
```rust
loop {
    if done { break; }
}
```

### ループ制御
```rust
for i in 0..10 {
    if i == 5 { continue; }
    if i == 8 { break; }
}
```

### Match 式
```rust
match value {
    Option::Some(x) => println("got: {}", x),
    Option::None => println("none"),
}
```

### If Let 式
```rust
if let Option::Some(x) = maybe_value {
    println("value is: {}", x);
} else {
    println("no value");
}
```

## 7. 演算子

### 算術演算子
`+`, `-`, `*`, `/`, `%`

### 比較演算子
`==`, `!=`, `<`, `>`, `<=`, `>=`

### 論理演算子
`&&`, `||`, `!`

### ビット演算子
`&` (AND), `|` (OR), `^` (XOR)

### レンジ演算子
`..` — レンジ (`0..10`, `0..`, `..10`)

### 型キャスト
```rust
let x = 42 as f32;
```

### Try 演算子 (`?`)
`Result` 型の値にのみ使用可能。`Err` の場合は早期リターンします。
```rust
fn read_file() -> Result<String, String> {
    let content = File::read("path.txt")?;
    Result::Ok(content)
}
```

## 8. 構造体と impl ブロック

```rust
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        Point { x: x, y: y }
    }

    fn distance(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
```

## 9. トレイト

トレイトはRustと同様のインターフェース機構です。デフォルトメソッドもサポートします。
```rust
trait Display {
    fn display(self) -> String;
}

trait Greetable {
    fn name(self) -> String;
    // デフォルトメソッド
    fn greet(self) -> String {
        let n = self.name();
        n.concat(", hello!")
    }
}

impl Display for Point {
    fn display(self) -> String {
        println("{}, {}", self.x, self.y);
        ""
    }
}
```

### 標準トレイト
- `Index<Idx, Output>` — `[]` 読み取りアクセス
- `IndexMut<Idx, Value>` — `[]` 書き込みアクセス
- `Iterable<T>` — `for` ループのイテレータプロトコル（`len` + `get`）

## 10. テンソル内包表記

```rust
// 構文: [indices | clauses { body }]
let A = [i, j | i <- 0..5, j <- 0..5 { i + j }];
```

暗黙的リダクション解析により、LHS に含まれないインデックスは自動的にリダクションインデックスとして検出されます（Einstein summation convention）。

## 11. 論理プログラミング

TLはDatalogスタイルの論理プログラミングを統合しています。

```rust
// リレーション定義
relation parent(entity, entity);

// ファクト定義
parent("Alice", "Bob");
parent("Bob", "Charlie");

// ルール定義
ancestor(X, Y) :- parent(X, Y);
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y);

// クエリ
?- ancestor("Alice", Who);
```

## 12. モジュールシステム

```rust
use math::{sin, cos};
use utils::*;           // Glob import
use parser as p;        // エイリアス
```
