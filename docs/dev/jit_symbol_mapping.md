# JITランタイム関数のデッドコード削除回避について

Rustで実装されたランタイム関数が、JITコンパイルされたコード（ユーザーの `.tl` コード）からのみ呼び出される場合、Rustコンパイラおよびリンカはそれらの関数を「未使用（dead code）」と判断し、リリースビルド時などにバイナリから削除してしまうことがあります。

これが起きると、JIT実行時に「シンボルが見つからない」というエラーが発生したり、関数ポインタが不正になりセグメンテーションフォールトが発生したりします。

## 回避策：明示的なマッピングの登録

この問題を回避するための推奨される方法は、**`src/compiler/codegen/builtins.rs` において、`ExecutionEngine` に明示的に関数マッピングを登録すること** です。

### 手順

1.  **ランタイム関数の定義**:
    関数は `pub` かつ `extern "C"` で定義し、`#[no_mangle]` 属性を付けてください。
    ```rust
    // src/runtime/stdlib.rs
    #[no_mangle]
    pub extern "C" fn tl_my_function(arg: i64) { ... }
    ```

2.  **マッピングの追加**:
    `src/compiler/codegen/builtins.rs` の `declare_runtime_functions` 関数内で、以下のように `add_global_mapping` を行います。

    ```rust
    // src/compiler/codegen/builtins.rs

    // 1. LLVM上の関数シグネチャを宣言（まだの場合）
    if module.get_function("tl_my_function").is_none() {
        let fn_type = ...; // シグネチャ定義
        module.add_function("tl_my_function", fn_type, None);
    }

    // 2. グローバルマッピングの登録（これが重要）
    if let Some(f) = module.get_function("tl_my_function") {
        execution_engine.add_global_mapping(
            &f,
            runtime::stdlib::tl_my_function as usize // Rust関数のアドレスを取得
        );
    }
    ```

### なぜこれで解決するのか？

`runtime::stdlib::tl_my_function as usize` と記述することで、Rustコード内でその関数のアドレスを取得する処理が発生します。これにより、Rustコンパイラは「この関数は使用されている」と判断し、デッドコード削除の対象から外します。

また、`ExecutionEngine` にアドレスを教えることで、JITコードがその関数を呼び出す際に、動的なシンボル解決（`dlsym` など）に頼らず、確実に正しいメモリアドレスにジャンプできるようになります。
