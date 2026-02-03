# 組み込み型の実装手順 (New Architecture)

組み込み型（`Vec`, `HashMap`, `Option` など）の実装手順を解説します。

> [!NOTE]
> このドキュメントは、主に **ジェネリックな組み込み型**（`Vec<T>` や `Option<T>` など）の実装を対象としています。ジェネリックでない単純な組み込み型については、[simple_builtin_type_implementation.md](./simple_builtin_type_implementation.md) を参照してください。

以前は Runtime/Compiler 間で手動の連携が必要でしたが、現在は **TL言語自身による定義とAST注入** を用いた新しいアーキテクチャに移行しています。

## 概要

組み込み型の実装（追加）は、以下の4つのステップで行います。

> [!IMPORTANT]
> **手順 1〜3 は、ビルド時に毎回自動で行われる処理ではありません。**
> これらは「新しいジェネリックな組み込み型を追加・定義するときに、開発者が行う（または一回限りのスクリプトで生成する）手順」です。
> 現状のビルドプロセスで `codegen` 内で自動的に `.tl` がパースされるわけではない点に注意してください。

1.  **TLコード記述**: 組み込みたい型とそのメソッドを、通常の **TL言語** で記述する。
2.  **AST取得 (Dev Time)**: 1で記述したTLコードを開発ツール等でパースし、AST (抽象構文木) を取得する。
3.  **Rustコード生成 (Dev Time)**: 2で得られたASTを再現するためのRustコード（AST Builder）を作成/生成する。
4.  **コンパイラへの注入 (Runtime)**: 3で用意したRustコードを用いて、コンパイラ起動時に `TypeRegistry` 等へASTを登録する。

この手法により、組み込み型は「コンパイラが特別扱いする黒魔術的な型」ではなく、**「標準ライブラリとしてプリロードされる、ただのジェネリックなユーザー定義型」** として扱われるようになります。

---

## 手順の詳細

### 1. TL言語での型定義（開発者が作成）
実装したい型を `.tl` ファイルとして作成し、**`src/compiler/codegen/builtin_types/` ディレクトリに配置します**。
これは「こうあるべき」型の定義書となります。

```rust
// src/compiler/codegen/builtin_types/vec.tl

// 実際のデータ構造定義
struct Vec<T> {
    ptr: ptr<T>,  // 内部ポインタ
    cap: I64,
    len: I64,
    // 必要に応じてアライメントなどのフィールド
}

impl Vec<T> {
    fn new() -> Vec<T> {
        // ...
    }

    fn push(self, item: T) {
// ...
    }
}
```

### 2. ASTの取得（開発時の作業）
開発ツール（手元のスクリプトや一時的なテストコードなど）を使用して、この `.tl` ファイルをパースし、どのようなAST構造になるかを確認します。

```rust
// build_helper.rs (concept)
let ast = parser::parse_file("stdlib/vec.tl")?;
// astの内容を出力して確認
```

### 3. Rustコード (AST Builder) の生成（開発時の作業）
取得した AST を再構築するための Rust コードを作成（またはスクリプトで自動生成）します。
この生成されたコードが、最終的にコンパイラのソースコードの一部（`vec.rs` など）になります。
「コンパイラが起動時に実行するコード」を、ここで準備します。

**生成されるRustコードのイメージ:**
```rust
// generated_builtins.rs

pub fn create_vec_struct() -> StructDef {
    StructDef {
        name: "Vec".to_string(),
        generics: vec!["T".to_string()],
        fields: vec![
            ("ptr".to_string(), Type::Ptr(Box::new(Type::Generic("T")))),
            // ...
        ],
    }
}

pub fn create_vec_impl() -> ImplBlock {
    ImplBlock {
        target_type: Type::UserDefined("Vec", vec![Type::Generic("T")]),
        generics: vec!["T"],
        methods: vec![
            FunctionDef {
                 name: "push".to_string(),
                 args: vec![("item".to_string(), Type::Generic("T"))],
                 return_type: Type::Void,
                 body: vec![ /* ... ASTの本体 ... */ ],
                 // ...
            },
            // ... 他のメソッド
        ],
    }
}
```

### 4. コンパイラへの注入と実装

コンパイラの初期化フェーズ（`register_builtins`）で、生成されたビルダー関数を呼び出し、通常のユーザー定義型と同じように登録します。

**重要な点**:
ここで登録されるのは **ジェネリックなAST (Generic AST)** です。
ユーザーが `Vec<I64>` を使用した瞬間に、コンパイラの既存の単相化 (Monomorphization) ロジックが走り、このジェネリックASTから具体的な実装 (`Vec_I64`) が生成されます。

したがって、**型ごとの手動実装（`builtin_impls.rs` での手書き）は不要** になります。

---

## 旧方式との違い

| 項目 | 旧方式 (Manual FFI) | 新方式 (AST Injection) |
| :--- | :--- | :--- |
| **定義場所** | Runtime(Rust) と Compiler(Rust) に分散 | **TLファイル (`.tl`) に集約** |
| **型チェック** | コンパイラ内で手動で整合性を取る | **TLのパーサ/型チェッカーが検証** |
| **メンテナンス** | 変更時に3箇所の修正が必要 | **`.tl` ファイルの修正のみ** |
| **特殊化** | Codegen内で手動ハンドリング | コンパイラの汎用単相化ロジックにお任せ |

---

## 開発者へのメモ

*   この方式に移行することで、`src/compiler/codegen/builtins.rs` や `builtin_impls.rs` の大部分は削除・自動化されるはずです。
*   現在、`Option` など一部の型で発生している「コンパイラマジック（型定義が存在せず、Codegenで無理やり生成している状態）」は、この方式によって解消され、明示的な型定義を持つようになります。

---

## 実装時の注意点 (Lessons Learned)

`Option<T>` や `Result<T, E>` の実装を通じて得られた、特に注意すべきポイントです。

### 1. ランタイム関数のJITマッピング漏れに注意
ジェネリック型（特に `Enum` や `Struct`）の操作において、コンパイラが自動生成するコード（`emit_deep_clone` や `unwrap` など）は、`tl_ptr_acquire`, `tl_ptr_inc_ref`, `tl_ptr_release` などの低レベルなメモリ管理関数を呼び出すことがあります。
これらの関数が `tl_runtime` クレートに実装されていても、**`src/compiler/codegen/builtins.rs` でJITエンジンに明示的にマッピングされていないと、LLVM実行時にシンボルが見つからずクラッシュしたり、正しく動作しない** ことがあります。
新しい組み込み型で複雑なメモリ操作が必要になる場合は、これらのマッピングが漏れていないか確認してください。

### 2. LLVM Codegenにおける `Type::I32` の扱い
TLの `i32` 型は、LLVM IR上でも `i32` として扱われるべきですが、以前の実装ではデフォルトで `i64` やポインタ型 (`ptr`) にフォールバックしてしまう箇所がいくつか存在しました。
特に以下の箇所での型マッピングが正確でないと、"Invalid generated function" や "Call parameter type does not match function signature" などのLLVM検証エラーが発生します。

*   **`compile_impl_blocks` / `compile_fn_proto`**: 関数の引数や戻り値の型定義。`Type::I32` を明示的に `context.i32_type()` にマップする必要があります。
*   **`compile_struct_defs`**: 構造体のフィールド定義。
*   **`compile_expr` (`ExprKind::FieldAccess`)**: フィールドから値をロードする際、そのフィールドの本来の型に基づいて `load` 命令の型（`i32` か `i64` かなど）を正しく指定する必要があります。

### 3. Enumのジェネリクスとメモリレイアウト
`Result<T, E>` のように複数のジェネリック型引数を持つ `Enum` を実装する場合、単相化（Monomorphization）が正しく行われているか注意が必要です。
特に `Result<Point, String>` のような「構造体とポインタ型が混在するケース」では、タグのサイズやペイロードのアライメントが正しく計算されないと、メモリ破壊（セグフォ）の原因になります。
AST注入方式では、TLコンパイラの標準的な構造体/Enum生成ロジックを利用するため、基本的には安全ですが、複雑なケースでは `test_result_complex.tl` のように、構造体を含むテストケースを作成して検証することを強く推奨します。
