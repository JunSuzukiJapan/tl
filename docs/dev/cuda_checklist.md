# CUDA バックエンド 未実装・修正チェックリスト

> [!NOTE]
> CPU/Metal バックエンドはすべて完了済み。本ドキュメントは CUDA (`tl_cuda`) のみを対象とする。

---

## 1. ロードマップ未完了項目

### テンソル操作

- [ ] `masked_fill_impl` — マスク付き値埋め
  - CPU/Metal: 実装済み
  - 場所: `crates/tl_cuda/src/ops/` に追加

- [ ] `to_dtype` / `to_f32` / `to_i64` — 型変換
  - CPU/Metal: 実装済み
  - 場所: `crates/tl_cuda/src/tensor.rs` または `ops/`

- [ ] `std(dim?)` / `var(dim?)` — 標準偏差・分散
  - CPU/Metal: `var_impl`, `std_impl` 実装済み
  - CUDA kernel: mean → 差の二乗 → mean のリダクションパイプライン

### NN 層

- [ ] `layer_norm` — LayerNorm CUDA 実装
  - CPU/Metal: 実装済み
  - 現状: `unimplemented!()` で即パニック

- [ ] `batch_norm` — BatchNorm CUDA 実装
  - CPU/Metal: 実装済み
  - `tl_device_tensor_batch_norm` dispatch でエラー

---

## 2. `unimplemented!()` の修正

### tensor.rs (2箇所)

- [ ] `ones()` の非F32 dtype サポート ([tensor.rs:227](file:///Users/junsuzuki/Program/Rust/tl/crates/tl_cuda/src/tensor.rs#L227))
  - 現在: `DType::F32` 以外で `unimplemented!` パニック
  - 対応: F64, I64 等のdtype branch を追加

- [ ] `randn()` の非F32 dtype サポート ([tensor.rs:247](file:///Users/junsuzuki/Program/Rust/tl/crates/tl_cuda/src/tensor.rs#L247))
  - 現在: `DType::F32` 以外で `unimplemented!` パニック
  - 対応: F64 branch を追加（I64 の乱数は意味がないのでエラーでも可）

### graph.rs (全体がスタブ)

- [ ] `CudaGraph::replay()` の実装 ([graph.rs:21](file:///Users/junsuzuki/Program/Rust/tl/crates/tl_cuda/src/graph.rs#L21))
  - 現在: `GpuGraph` trait の `replay()` が `unimplemented!`
  - 対応: `cudaGraphExec_t` ラッパーとして実装（将来課題）

---

## 3. 将来最適化（低優先度）

- [ ] Flash Attention — cutlass/flash-attn integration
  - 現在: CPU fallback の `scaled_dot_product_attention` で動作
  - 最適化: CUDA 専用の fused attention kernel

---

## 優先順位

1. **最優先**: `ones`/`randn` の dtype パニック修正（ランタイムクラッシュ防止）
2. **高優先**: `masked_fill`, `to_dtype`, `std`/`var`, `layer_norm`, `batch_norm`
3. **低優先**: `CudaGraph::replay()`, Flash Attention 最適化
