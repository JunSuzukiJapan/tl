# TL 標準ライブラリ 実装ガイド

このドキュメントでは、TensorLanguage (TL) コンパイラにおいて、TL言語で記述された標準ライブラリ（シグネチャおよび実装）を追加または変更する方法について解説します。

TL の標準ライブラリは `rust-embed` を用いて、コンパイラのバイナリファイル内に仮想ファイルシステムとして統合されています。これにより、環境ごとに外部ファイルを配置しなくても `cargo install` 等によって完全なコンパイラを提供できます。

## 1. 全体アーキテクチャ

標準ライブラリの読み込み機構は以下のファイル構成・機能で実現されています。

- **`src/compiler/codegen/builtin_types/` 配下**
  すべての `.tl` ソースファイル（組み込みインタフェースの定義等）と、それをRust側で読み込むためのモジュールが存在します。
- **`assets.rs`**
  `rust-embed` クレートを利用してディレクトリごとバイナリに格納し、実行時にファイル名からテキストをロードするための `BuiltinAssets` 構造体を定義しています。
- **`mod.rs` (Builtin_types)**
  ロードした TL の定義をコンパイラの `TypeManager` へ登録し、各ファイルのソースを結合（Inject）するためのエントリポイントです。

## 2. 新しい標準ライブラリを追加する手順

新しいデータ型、あるいは関数群を標準ライブラリとして提供する場合、以下のステップを踏みます。

### ステップ1: `.tl` ファイルの作成

`generic` または `non_generic` ディレクトリ配下に、実装・宣言したい TL のソースコードを配置します。

例: `src/compiler/codegen/builtin_types/non_generic/my_library.tl`
```tl
pub struct MyLibrary {
    handle: i64,
}

impl MyLibrary {
    // 実際の実装を持たせるか、Rust (FFI) にデリゲートする関数プロトタイプ
    pub fn new() -> MyLibrary {
        // ...
    }
}
```

### ステップ2: Rust ロードモジュールの作成

対応する Rust コード (`my_library.rs`) を作成し、`assets.rs` 経由で文字列をロードして `BuiltinLoader` に渡す仕組みを作ります。

例: `src/compiler/codegen/builtin_types/non_generic/my_library.rs`
```rust
use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_my_library() -> crate::compiler::builtin_loader::BuiltinTypeData {
    // assets モジュールから文字列ソースを動的に取得
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/my_library.tl");
    
    BuiltinLoader::load_module_data(&source, "MyLibrary")
        .expect("Failed to load MyLibrary")
}
```

### ステップ3: モジュールへの登録

`src/compiler/codegen/builtin_types/mod.rs` にステップ2で作成したモジュールと実行ロジックを登録します。

1. `pub mod my_library;` などで公開設定。
2. `load_all_builtins(codegen: &mut CodeGenerator)` メソッド内の中央付近で以下のように登録を追加します。

```rust
// MyLibrary の登録
let my_lib_data = non_generic::my_library::load_my_library();
codegen.type_manager.register_builtin(my_lib_data.clone());
if let Some(def) = my_lib_data.struct_def.clone() {
    codegen.struct_defs.insert(def.name.clone(), def);
}
codegen.generic_impls.entry("MyLibrary".to_string()).or_default().extend(my_lib_data.impl_blocks);
```

### ステップ4: FFI 関数のマッピング (シグネチャの場合のみ)

TLソースコード上のメソッド本体が `extern` や LLVM IR の組み込み側を叩く設計になっている場合（システムI/O, メモリ操作, スレッドなど）、
`src/compiler/codegen/builtins.rs` の `declare_runtime_functions` で LLVM 関数として宣言し、`execution_engine.add_global_mapping(...)` を用いて、Rust 側の関数ポインタと JIT 側の関数とを紐づける作業が必要です。

### ステップ5: `src/main.rs` ドキュメント参照の追加

最後に `src/main.rs` の `load_builtins()` 内にある `paths` 配列に、ファイルを確実に追加します。
（このステップにより、ユーザーがコンパイルを走らせるたびに仮想アセットが解析され、ユーザのコード空間にマージされます）

```rust
let paths = [
    // ...
    "non_generic/my_library.tl", // ← これを追加
];
```

## 3. デバッグと検証

追加・編集が完了したら、以下のコマンドでコンパイラ自身と TL スクリプトがエラーなくパースされることを検証します。

```bash
cargo check
python3 scripts/verify_tl_files.py
```
