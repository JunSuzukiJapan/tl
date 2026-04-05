# CUDA バックエンド 未実装・修正チェックリスト

> [!NOTE]
> CPU/Metal バックエンドはすべて完了済み。本ドキュメントは CUDA (`tl_cuda`) のみを対象とする。

---

## 1. ロードマップ未完了項目

### テンソル操作

- [x] `masked_fill_impl` — マスク付き値埋め
  - CPU/Metal: 実装済み
  - CUDA: `ops/special.rs` に CUDA カーネル実装済み

- [x] `to_dtype` / `to_f32` / `to_i64` — 型変換
  - CPU/Metal: 実装済み
  - CUDA: `backend_impl.rs` + `ffi_ops.rs` に実装済み

- [x] `std(dim?)` / `var(dim?)` — 標準偏差・分散
  - CPU/Metal: `var_impl`, `std_impl` 実装済み
  - CUDA: `ops/reduce.rs` に GPU カーネル実装済み

### NN 層

- [x] `layer_norm` — LayerNorm CUDA 実装
  - CPU/Metal: 実装済み
  - CUDA: `ops/nn.rs` + `cuda_kernels/autograd.cu` に GPU カーネル実装済み

- [x] `batch_norm` — BatchNorm CUDA 実装
  - CPU/Metal: 実装済み
  - CUDA: `ops/nn.rs` + `cuda_kernels/autograd.cu` に GPU カーネル実装済み

---

## 2. `unimplemented!()` の修正

### tensor.rs

- [x] `ones()` の I64/I32 dtype サポート (`crates/tl_cuda/src/tensor.rs`)
  - F32, I64, I32 対応済み

- [x] `ones()` / `randn()` の F64 dtype サポート
  - 現在: 追加および実装済み
  - 対応: `tl_backend`, `tl_cuda`, `tl_metal` に `DType::F64` を追加し、CUDA で実装

### graph.rs

- [x] `CudaGraph::replay()` の実装 (`crates/tl_cuda/src/graph.rs`)
  - 現在: 実装済み
  - 対応: `cudaGraphExec_t` ラッパーとして実装

---

## 3. 最適化

- [x] Flash Attention — fused attention kernel
  - 現在: `autograd.cu` の `sdpa_kernel` は `flash_attention_kernel` として実装済み
  - 最適化: タイル化 + online softmax + shared memory による Flash Attention v2 完了

---

## 優先順位

1. **最優先**: F64 DType 追加 + `ones`/`randn` 対応
2. **高優先**: Flash Attention（SDPA 性能改善）
3. **低優先**: `CudaGraph::replay()`
