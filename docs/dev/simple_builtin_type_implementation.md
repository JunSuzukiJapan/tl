# 単純な組み込み型の実装手順

TL言語に**ジェネリックでない単純な組み込み型**（例: 独自の数値型、システムハンドルラッパーなど）を追加する手順を解説します。

複雑なジェネリック型（`Vec<T>`など）の実装については、[builtin_type_implementation.md](./builtin_type_implementation.md) を参照してください。

## 概要

単純な組み込み型の実装は、以下の手順で行います。
メソッドの探索・呼び出しには**TypeManager**を使用し、ジェネリック型と統一されたインターフェースで管理します。

1.  **Runtime (`crates/tl_runtime`)**: 実際の処理を行うRust関数をFFI互換 (`extern "C"`) で実装する。
2.  **Compiler Codegen (`TypeManager`)**: `TypeManager` に型とメソッドを登録する。**実装が既に存在する（Runtime関数がある）メソッド**として登録する点がポイントです。

---

## 手順 1: Runtimeの実装

従来通り、`crates/tl_runtime/src/stdlib.rs`（または機能ごとの適切なファイル）に、メソッドの実体となる関数を実装します。

### 1. 関数の宣言
関数は `#[unsafe(no_mangle)]` 属性と `pub extern "C"` を付けて宣言し、C ABI で呼び出せるようにします。

**命名規則**: `tl_<型名>_<メソッド名>`
*   例: `String.len()` → `tl_string_len`
*   例: `MyType.process()` → `tl_mytype_process`

### 2. 引数と戻り値
Rustの標準型そのままではなく、FFI境界を越えられる型を使用します。

*   **文字列**: `*const c_char` (入力), `*mut c_char` (出力)
*   **オブジェクト（構造体など）**: `*mut Void` (実体は `*mut T`)
    *   ポインタとして受け渡しを行います。

---

## 手順 2: Compilerへの登録 (TypeManager)

`src/compiler/codegen/builtin_types/non_generic` ディレクトリ内に新しいモジュール（例: `mytype.rs`）を作成し、`TypeManager` への登録処理を記述します。

### 1. モジュールの作成と登録関数の定義

```rust
use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::expr;
// その他必要なインポート

pub fn register_my_types(manager: &mut TypeManager) {
    let mut my_type = CodeGenType::new("MyType");

    // Static Method の登録
    my_type.register_static_method(
        "new", 
        expr::StaticMethod::Evaluated(compile_mytype_new)
    );

    // Instance Method の登録
    my_type.register_instance_method(
        "process", 
        expr::InstanceMethod::Evaluated(compile_mytype_process)
    );

    manager.register_type(my_type);
}
```

### 2. メソッドの種類について (Evaluated vs Unevaluated)

TypeManagerに登録するメソッドには大きく分けて2種類あります。単純な組み込み型では主に **`Evaluated`** を使用します。

*   **`Evaluated` (実装済みメソッド)**:
    *   Runtime側に実装（`tl_mytype_...`）が存在し、コンパイラは引数を評価した後、単にそのRuntime関数を呼び出すコードを生成します。
    *   単純な組み込み型は通常これを使用します。
    *   登録時に `expr::StaticMethod::Evaluated` または `expr::InstanceMethod::Evaluated` を使用します。

*   **`Unevaluated` (ジェネリック/コンパイラマジック)**:
    *   コンパイラ内で特殊な処理（ASTの操作や特殊なIR生成）が必要な場合に使用します。例えば `Tensor` のリテラル処理など。
    *   引数が評価される前の状態で渡されます。

### 3. コンパイル関数の実装

各メソッドに対応するコンパイル関数（`compile_mytype_new` など）を実装します。ここでLLVM IRを生成してRuntime関数を呼び出します。

```rust
fn compile_mytype_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // 1. Runtime関数の取得
    let fn_val = codegen.module.get_function("tl_mytype_new")
        .ok_or("tl_mytype_new not found")?;

    // 2. 引数の準備 (必要に応じてキャストなど)
    // ...

    // 3. 関数呼び出し (Builder::build_call)
    let call = codegen.builder.build_call(fn_val, &[], "new_res")
        .map_err(|e| e.to_string())?;

    // 4. 戻り値の処理
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return".into()),
    };

    Ok((res, Type::Struct("MyType".to_string(), vec![])))
}
```

---

## 手順 3: モジュールの統合

1.  作成したモジュール（`mytype.rs`）を `src/compiler/codegen/builtin_types/non_generic/mod.rs` に追加します。
2.  `src/compiler/codegen/mod.rs` (または `builtin_types/mod.rs` の初期化処理) で、`register_my_types` を呼び出して登録を有効にします。

---

## チェックリスト
1. [ ] Runtime関数実装 (FFI, `stdlib.rs`)
2. [ ] Codegen: `non_generic` ディレクトリにモジュール作成
3. [ ] Codegen: `TypeManager` への登録 (`Evaluated` を使用)
4. [ ] Codegen: コンパイル関数 (LLVM IR生成, Runtime関数呼び出し) の実装
5. [ ] Codegen: モジュールの公開と初期化処理への追加
