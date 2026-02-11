# TL Backend Architecture

## Overview

TL (TensorLogic) 言語は、パフォーマンスとポータビリティを両立するために、明確に分離されたマルチバックエンドアーキテクチャを採用しています。

- **CPU Backend** (`crates/tl_cpu`): 常に利用可能なベースライン実装。
- **GPU Backend** (`crates/tl_metal`): 高性能な Metal/GPU 実装 (macOS)。
- **Runtime Interface** (`crates/tl_runtime`): コンパイラが参照する統一インターフェース。

## Design Principles

### 1. Backend Separation
各バックエンドは独立したクレートとして実装されます。
- `tl_cpu`: 純粋な CPU 演算、`CpuTensor` を提供。
- `tl_metal`: Metal API を使用した GPU 演算、`MetalTensor` を提供。

### 2. Runtime as Facade
`tl_runtime` は具体的なバックエンド実装を直接持たず（理想的には）、バックエンドクレートを再エクスポート (`pub use`) したり、共通のユーティリティを提供するファサードとして機能します。
コンパイラ (`builtins.rs`) は `tl_runtime` のシンボルを参照しますが、実体は各バックエンドクレートにあります。

### 3. JIT Linking Strategy (`builtins.rs`)
JIT コンパイル時、`builtins.rs` の `map_tensor_fn!` マクロによって、ターゲットデバイスに応じた関数ポインタがマッピングされます。

- **CPU Mode**: `tl_cpu::ffi::*` の関数をリンク。
- **GPU Mode**: `tl_runtime` (実体は `tl_metal::ffi::*`) の関数をリンク。

### 4. No Wrapper Pattern
従来の `tl_tensor_release_safe` のように、ランタイム関数内で `if is_cpu` で分岐するパターンは廃止されました。
各バックエンドは同名の FFI 関数 (`extern "C"`) を定義し、JIT レベルでリンク先を切り替えることで、ランタイムオーバーヘッドをゼロにします。

## Crates Responsibilities

| Crate | Responsibility |
|:---|:---|
| `tl_cpu` | CPU テンソル演算、Autograd (CPU)、FFI 実装 (CPU) |
| `tl_metal` | Metal テンソル演算、MPS シェーダー管理、FFI 実装 (GPU) |
| `tl_runtime` | FFI インターフェース定義、バックエンドの再エクスポート、共通ユーティリティ |
| `tl_backend` | 共通トレイト (`GpuTensor`, `GpuOps`) 定義 |

## FFI Function Mapping

全てのテンソル操作関数は、以下のようにマッピングされます：

```rust
// builtins.rs
map_tensor_fn!("tl_tensor_add", runtime::tl_tensor_add, cpu_ffi::tl_cpu_tensor_add);
```

- `runtime::tl_tensor_add`: `tl_metal::ffi::tl_tensor_add` への re-export
- `cpu_ffi::tl_cpu_tensor_add`: `tl_cpu::ffi::tl_cpu_tensor_add`

## Future Work

- `tl_runtime` に残存している Metal 依存コード (`tensor_ops_ext.rs` 等) を順次 `tl_metal` へ移動し、`tl_runtime` を純粋なインターフェースにする。
