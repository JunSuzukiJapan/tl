# 組み込み型の実装手順 (New Architecture)

組み込み型（`Vec`, `HashMap`, `Option` など）の実装手順を解説します。

> [!NOTE]
> このドキュメントは、主に **ジェネリックな組み込み型**（`Vec<T>` や `Option<T>` など）の実装を対象としています。ジェネリックでない単純な組み込み型は、従来の方式でも比較的容易に追加可能です。

以前は Runtime/Compiler 間で手動の連携が必要でしたが、現在は **TL言語自身による定義とAST注入** を用いた新しいアーキテクチャに移行しています。

## 概要

組み込み型の実装は、以下の4つのステップで行います。

1.  **TLコード記述**: 組み込みたい型とそのメソッドを、通常の **TL言語** で記述する。
2.  **AST解析**: 1で記述したTLコードをコンパイラ（のパーサ）で読み込み、AST (抽象構文木) を取得する。
3.  **Rustコード生成**: 2で得られたASTを解析し、**「そのASTと同等のASTをプログラム的に組み立てるRustコード」** を生成する（マクロやビルドスクリプトを使用）。
4.  **コンパイラへの注入**: 3で生成されたRustコードを用いて、特殊化前（Generic）のASTとしてコンパイラの `TypeRegistry` や `CodeGenerator` に登録する。

この手法により、組み込み型は「コンパイラが特別扱いする黒魔術的な型」ではなく、**「標準ライブラリとしてプリロードされる、ただのジェネリックなユーザー定義型」** として扱われるようになります。

---

## 手順の詳細

### 1. TL言語での型定義 (`src/compiler/codegen/builtin_types/`)

まず、実装したい型を `.tl` ファイルとして作成し、**`src/compiler/codegen/builtin_types/` ディレクトリに配置します**。

```rust
// src/compiler/codegen/builtin_types/vec.tl

// 実際のデータ構造定義
struct Vec<T> {
    ptr: ptr<T>,  // 内部ポインタ
    cap: I64,
    len: I64,
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

### 2. ASTの取得

開発ツール（ビルドスクリプト等）を使用して、この `.tl` ファイルをパースします。

```rust
// build.rs (concept)
let ast = parser::parse_file("stdlib/vec.tl")?;
```

### 3. Rustコード (AST Builder) の生成

取得した AST を再構築するための Rust コードを生成します。
これは、コンパイラが起動時に実行するコードになります。

**生成されるRustコードのイメージ:**
```rust
// generated_builtins.rs

pub fn create_vec_ast() -> StructDef {
    StructDef {
        name: "Vec".to_string(),
        generics: vec!["T".to_string()],
        fields: vec![
            ("ptr".to_string(), Type::Ptr(Box::new(Type::Generic("T")))),
            // ...
        ],
        // ...
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
