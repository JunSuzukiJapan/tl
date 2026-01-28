# 組み込み型の実装手順

TL言語に新しい組み込み型を追加したり、既存の組み込み型にメソッドを追加したりする際の手順を説明します。

## 概要

組み込み型の実装は、主に以下の3つのステップで構成されます。

1.  **Runtime (`crates/tl_runtime`)**: 実際の処理を行うRust関数をFFI互換 (`extern "C"`) で実装する。
2.  **Compiler Semantics (`src/compiler/type_registry.rs`, `semantics.rs`)**: コンパイラの型システムにメソッドのシグネチャ情報を登録する。
3.  **Compiler Codegen (`src/compiler/codegen/builtins.rs`, `expr.rs`)**: LLVMコード生成のためにRuntime関数を宣言し、呼び出しロジックを調整する。

## 基本概念：組み込み型の透明性と完全な登録

組み込み型は、**あくまでも便利だから最初から組み込まれてるユーザー定義型の一種でしかありません**。

したがって、システム（コンパイラ・ランタイム）に対して**不透明なブラックボックスであってはなりません**。
**システムに組み込み型の中身が直接わからないのであれば、システムがそのすべてを把握できるように詳細に登録し、ユーザー定義型と同じ仕組みで管理しなければなりません。**

具体的には以下の情報の「完全な登録」により、不透明性を排除する必要があります：
1.  **振る舞い**: メソッドのシグネチャ（引数と戻り値の型）
2.  **メモリ管理**: 確保と**解放（Free）**の仕組み

システムが組み込み型を特別扱いするのではなく、登録された情報に基づいてユーザー定義型と同等に扱えるように設計する必要があります。

---

## 手順 1: Runtimeの実装

まず、`crates/tl_runtime/src/stdlib.rs`（または機能ごとの適切なファイル）に、メソッドの実体となる関数を実装します。

### 1. 関数の宣言
関数は `#[unsafe(no_mangle)]` 属性と `pub extern "C"` を付けて宣言し、C ABI で呼び出せるようにします。

**命名規則**: `tl_<型名>_<メソッド名>`
*   例: `String.len()` → `tl_string_len`
*   例: `HashMap.insert()` → `tl_hashmap_insert`

### 2. 引数と戻り値
Rustの標準型そのままではなく、FFI境界を越えられる型を使用します。

*   **文字列**: `*const c_char` (入力), `*mut c_char` (出力)
*   **オブジェクト（構造体など）**: `*mut Void` (実体は `*mut T`)
    *   `Hashmap` 等の構造体は、ポインタとして受け渡しを行います。

---

## 手順 2: Compilerへの登録

### 1. TypeRegistry (インスタンスメソッド)
`src/compiler/type_registry.rs` にメソッドシグネチャを登録します。

```rust
// new() -> HashMap (TypeRegistryにも登録しておくと整理しやすいが、実動作は後述のStaticMethodCallで処理される場合が多い)
// insert, get 等のインスタンスメソッドはここで必須
map_methods.insert("insert".to_string(), ...);
```

### 2. Static Method (`Type::new()`) の登録
`MyType::new()` のようなスタティックメソッドをサポートするには、`src/compiler/semantics.rs` の `check_expr` (StaticMethodCall処理) のフォールバックロジックに追加が必要です。

```rust
// src/compiler/semantics.rs lines ~4640 (Fallback logic)
("MyType", "new") => {
    // 引数チェックなど
    Ok(Type::UserDefined("MyType".to_string(), vec![]))
}
```

---

## 手順 3: Compilerの実装 (Codegen)

### 1. Runtime関数の宣言 (`src/compiler/codegen/builtins.rs`)

`declare_runtime_functions` に Runtime関数のシグネチャを LLVM `FunctionType` として登録します。
`module.add_function` で追加された関数は、`CodeGenerator` の `get_return_type_from_signature` メソッドによって自動的に戻り値の型が推論されます（`fn_return_types` への手動登録は不要になりました）。

### 2. Static Method のコンパイル (`src/compiler/codegen/expr.rs`)

スタティックメソッドのコード生成ロジックを追加します。`compile_static_method_call` メソッドを編集します。

```rust
// src/compiler/codegen/expr.rs

if type_name == "MyType" && method_name == "new" {
     let fn_val = self.module.get_function("tl_mytype_new").unwrap();
     let call = self.builder.build_call(fn_val, &[], "mytype_new")?;
     // ... 戻り値処理 ...
     return Ok((res, ty.clone()));
}
```

### 3. SRET (Struct Return) の除外設定

ポインタを直接返す型（`HashMap`, `Vec` 等）は、SRET最適化を除外する必要があります。
`codegen/mod.rs` と `codegen/expr.rs` の判定ロジックに型名を追加してください。

---

## よくある間違いと修正

### 誤解: 「特殊化された関数をあらかじめ実装する必要がある」
**修正**: 組み込み型は、**システムに型とメソッドの定義を登録するだけ**であるべきです。
ユーザーが `Vec<i64>` を使用した時点で、システム（CodeGenerator）が必要な特化実装を生成（Monomorphization）します。
開発者が `tl_vec_i64_push`, `tl_vec_f32_push` などを手動（またはマクロ）で事前に大量に用意する必要はありません。システムがどのように特化を処理するか（例：`i64` は `ptr` 実装をキャストして使う、あるいはLLVM IRをインライン生成する等）は、Codegen層の責務です。

**ランタイム (`crates/tl_runtime`) には、いかなる関数も実装してはいけない。**

### 名前マングリングによる整合性保証
`src/compiler/codegen/mono.rs` の `mangle_generic_method` 関数を使用することで、**関数名をどこかに個別に登録したり記憶したりすることなく、呼び出し側（Compiler）と呼ばれる側（Backend/Symbol）の関数名が自動的に一致する** 仕組みになっています。

```rust
// src/compiler/codegen/mono.rs
pub fn mangle_generic_method(...) -> String {
    // 例: tl_vec_i64_pop
    format!("tl_{}{}_{}", base_type, suffix, method)
}
```
この機構により、ジェネリック型の特殊化は完全に自動化されており、手動での名前管理は不要（かつ有害）です。

---

## 実装の注意点とベストプラクティス (Lessons Learned)

`Option<T>` などのジェネリック型を実装する際に発生しやすい問題と、その回避策です。

### 1. Codegenにおける型注釈の正規化 (Normalization)

`Let` 文などで変数に型注釈が付いている場合、ASTレベルではプリミティブ型も `UserDefined("I64", [])` のように表現されることがあります。
しかし、Codegen フェーズでは `Type::I64` と `Type::UserDefined("I64", ...)` は**別の型**として扱われます。これが原因で `BinOp` (二項演算) などで `Type mismatch` エラーが発生することがあります。

**対策**:
Codegen (`stmt.rs`, `expr.rs`) で型注釈を扱う際は、ジェネリック引数に含まれるプリミティブ型名（"I64", "Bool" 等）を検出し、正規化（`Type::I64` 等への変換）を行う必要があります。

```rust
// 例: UserDefined("I64") -> Type::I64 への正規化ロジック
if let Type::UserDefined(name, empty) = inner {
    if empty.is_empty() {
        let norm = match name.as_str() {
            "I64" => Some(Type::I64),
            "Void" => Some(Type::Void),
            _ => None
        };
        // ... 適用 ...
    }
}
```

### 2. `Void` 型の取り扱いとパニック回避

`Option::none()` のように、ジェネリック型引数がコンテキストから推論できない（またはデフォルト値が必要な）場合、安易に `Type::Void` を使用すると、LLVM コード生成時 (`get_llvm_type`) に `Void type encountered` パニックを引き起こす可能性があります（`Void` はサイズを持たないため）。

**対策**:
*   **ダミー型の使用**: レイアウトを確定させるため、`Void` の代わりに `I64` などの安全な型プレースホルダーを使用する。
    *   例: `Option<Void>` ではなく `Option<I64>` として生成する（タグが `None` なら値はアクセスされないため安全）。
*   **ポインタキャスト**: 生成された値を特定の型変数に代入する際は、明示的なポインタキャスト (`build_pointer_cast`) を行って整合性を保つ。

### 3. 型注釈の強制適用 (Type Forcing)

`Option::none()` -> `Option<I64>` (Default) を `Option<String>` 型の変数に代入する場合など、右辺値の型と左辺の型が一致しないことがあります。

**対策**:
`Let` 文の処理 (`stmt.rs`) において、型注釈が存在する場合はそちらを優先し、右辺値の型情報を上書き（およびキャスト）するロジックを組み込むことで、後続の処理での型不一致を防げます。
