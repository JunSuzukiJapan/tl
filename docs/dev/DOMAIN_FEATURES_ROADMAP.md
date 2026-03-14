# TL言語 ドメイン固有機能 実装ロードマップ

> [!NOTE]
> **全フェーズ完了 (2026-03)**
> フェーズ 1〜8 の全項目が実装済み。CUDA バックエンドは stub (`unimplemented!()`) のみ残存。
> このドキュメントは実装履歴の記録として保存。

> [!CAUTION]
> **新機能を追加する場合は、必ず [MEMORY_MANAGEMENT_STRATEGY.md](MEMORY_MANAGEMENT_STRATEGY.md) を読むこと。**
> TLのメモリ管理はARCレジストリとプール制御に基づいており、新しいテンソル操作やNN層を追加する際にはこの戦略に従う必要がある。

> [!IMPORTANT]
> **勾配追跡は型レベルで分離されている。**
> - `Tensor` = 勾配なし（推論用）、`GradTensor` = 勾配あり（学習用）
> - PyTorchの `no_grad` のようなランタイムフラグは**不要かつ禁止**
> - 詳細: [.agent/DESIGN_PRINCIPLES.md](../../.agent/DESIGN_PRINCIPLES.md)

DOMAIN_FEATURES_ANALYSIS.md の分析結果に基づく実装計画。
セクション6（ドキュメント乖離）は修正済みのため除外。
既に実装済みの機能（squeeze/unsqueeze/flatten/permute/contiguous/cat/gather/比較演算子/dropout/shape/ndim/save/to_i64等）も除外。

> [!IMPORTANT]
> 各機能の実装には以下のレイヤーすべてが必要:
> 1. **バックエンド実装** (`tl_cpu`, `tl_metal`, `tl_cuda`)
> 2. **FFIブリッジ** (`tl_runtime/device_ffi` or 各crateのffi.rs)
> 3. **コンパイラ登録** (TypeManager or builtins.rs)
> 4. **テスト** (Rust unit test + `.tl` テストファイル)
> 5. **ドキュメント更新**

---

## フェーズ 1: テンソル生成（基盤拡充）

### 1.1 高優先度

- [x] `Tensor::full(shape, value, requires_grad?)` — 任意値テンソル生成
  - [x] tl_cpu: `full_impl`
  - [x] tl_metal: `full_impl` (GPU kernel: fill with constant)
  - [x] tl_cuda: `full_impl`
  - [x] FFI: `tl_tensor_full(data_ptr, value, rank, shape_ptr, requires_grad)`
  - [x] TypeManager: 静的メソッド登録
  - [x] テスト: `tests/test_tensor_creation.tl`

- [x] `Tensor::eye(n, requires_grad?)` — 単位行列
  - [x] tl_cpu: `eye_impl`
  - [x] tl_metal: `eye_impl` (GPU kernel: 2D grid, out[i,j] = i==j ? 1 : 0)
  - [x] tl_cuda: `eye_impl`
  - [x] FFI + TypeManager登録
  - [x] テスト

- [x] `Tensor::arange(start, end, step)` — 連番テンソル
  - [x] tl_cpu: `arange_impl`
  - [x] tl_metal: `arange_impl` (GPU kernel: out[i] = start + i * step)
  - [x] tl_cuda: `arange_impl`
  - [x] FFI + TypeManager登録
  - [x] テスト

- [x] `Tensor::from_vec(vec: Vec<f32>, shape: Vec<i64>)` — Vecからテンソル生成
  - [x] tl_cpu: `from_vec_f32_impl`
  - [x] tl_metal: `from_vec_f32_impl`
  - [x] tl_cuda: `from_vec_f32_impl`
  - [x] FFI + TypeManager登録（`from_vec_u8`と同様パターン）
  - [x] テスト

- [x] `Tensor::zeros_like(t)` / `Tensor::ones_like(t)` — 形状コピー生成
  - [x] 全バックエンド: shapeを取得 → zeros/ones呼び出し（ラッパー）
  - [x] FFI + TypeManager登録
  - [x] テスト

### 1.2 中優先度

- [x] `Tensor::linspace(start, end, steps)` — 等間隔テンソル
  - [x] 全バックエンド実装
  - [x] FFI + TypeManager登録

- [x] `Tensor::rand(shape, requires_grad?)` — 一様分布乱数
  - [x] 全バックエンド実装
  - [x] FFI + TypeManager登録

- [x] `Tensor::rand_like(t)` / `Tensor::randn_like(t)` — 形状コピー乱数
  - [x] 全バックエンド（ラッパー）

---

## フェーズ 2: 要素操作・リダクション

### 2.1 高優先度

- [x] `where_cond(condition, x, y)` — TL APIとして公開
  - [x] バックエンド: 実装済み（全バックエンドに`where_cond_impl`あり）
  - [x] TypeManager: evaluated 静的メソッドとして登録
  - [x] テスト

- [x] `masked_fill(mask, value)` — マスク付き値埋め
  - [x] tl_cpu: `masked_fill_impl`
  - [x] tl_metal: GPU kernel (condition ? value : original)
  - [ ] tl_cuda: `masked_fill_impl`
  - [x] FFI + TypeManager登録
  - [x] テスト

- [x] `to_dtype` / `to_f32` / `to_i64` — 型変換
  - [x] tl_cpu: 実装済み
  - [x] tl_metal: 実装済み
  - [ ] tl_cuda: 未実装
  - [x] FFI + TypeManager登録 (`to_f32`, `to_i64`)
  - [x] テスト

- [x] `to_vec() -> Vec<f32>` — テンソルからVecへの変換
  - [x] ランタイム: `tl_device_tensor_to_vec_f32` 追加 (Vec<f32> への変換)
  - [x] TypeManager登録 + compile関数
  - [x] テスト

- [x] `std(dim?)` / `var(dim?)` — 標準偏差・分散
  - [x] tl_cpu: `var_impl`, `std_impl`
  - [x] tl_metal: GPU reduction (mean→差の二乗→mean)
  - [ ] tl_cuda: 同上
  - [x] FFI + TypeManager登録
  - [x] テスト

### 2.2 中優先度

- [x] `logical_and` / `logical_or` / `logical_not` — 論理演算
  - [x] CPU/Metal: element-wise kernel, CUDA unimplemented!()
  - [x] FFI + TypeManager登録

- [x] `fill_(value)` — インプレース埋め込み
  - [x] CPU/Metal 実装、CUDA unimplemented!()

- [x] `prod(dim?)` — 全要素の積
  - [x] CPU/Metal: reduction kernel (exp(sum(log)))
  - [x] FFI + TypeManager登録

- [x] `cumsum(dim)` — 累積和
  - [x] CPU/Metal: sequential scan, CUDA unimplemented!()
  - [x] FFI + TypeManager登録

- [x] `norm(p, dim?)` — Lpノルム
  - [x] CPU/Metal 実装, CUDA unimplemented!()

- [x] `topk(k, dim)` — 上位k個
  - [x] CPU/Metal: partial sort, CUDA unimplemented!()
  - [x] FFI + TypeManager登録

---

## フェーズ 3: 形状操作（追加）

### 3.1 中優先度

- [x] `expand(shape)` / `broadcast_to(shape)` — 明示的ブロードキャスト
  - [x] TypeManager登録、compile関数、device_ffi、builtins
  - [x] Tensor<i64>引数のサポート追加（reshape_dimsと同じパターン）

- [x] `view(shape)` — ゼロコピーreshape
  - [x] reshapeのエイリアスとしてTypeManager登録

- [x] `chunk(n, dim, index)` — n分割の i 番目を返す
  - [x] 全バックエンド: narrow ラッパーとして実装（index指定で個別Tensor返却）

- [x] `split(size, dim, index)` — 指定サイズ分割の i 番目を返す
  - [x] 全バックエンド: narrow ラッパーとして実装

- [x] `stack(tensors, dim)` — 新次元で結合
  - [x] CPU/Metal: unsqueeze + cat
  - [x] FFI + TypeManager登録
  - [x] テスト: stack([1,1,1], [2,2,2], 0) = [1,1,1,2,2,2]

---

## フェーズ 4: ニューラルネットワーク層

### 4.1 正規化（高優先度）

- [x] `layer_norm(normalized_shape, weight, bias, eps)` — LayerNorm
  - [x] tl_cpu/tl_metal: 既存実装を活用
  - [x] TypeManager登録 + compile関数
  - [ ] tl_cuda: unimplemented!()
  - [x] テスト: 使用可能（既存パイプライン経由）

- [x] `batch_norm(weight, bias, running_mean, running_var, eps, training)` — BatchNorm
  - [x] tl_cpu: 実装済み（`tl_cpu_tensor_batch_norm`）
  - [x] tl_metal: バックエンドに実装パターンあり → TL API公開
  - [ ] tl_cuda: 実装
  - [x] FFI + TypeManager登録
  - [x] compile関数: `compile_tensor_batch_norm`

### 4.2 プーリング（高優先度）

- [x] `max_pool2d(kernel_size, stride, padding)` — 最大プーリング
  - [x] 既存パイプライン活用、TypeManager登録 + compile関数
  - [x] FFI + builtins 既存

- [x] `avg_pool2d(kernel_size, stride, padding)` — 平均プーリング
  - [x] 既存パイプライン活用、TypeManager登録 + compile関数
  - [x] FFI + builtins 既存

### 4.3 その他NN（高〜中優先度）

- [x] `linear(weight, bias)` — 全結合層
  - [x] CPU/Metal: matmul(W^T) + bias
  - [x] FFI + TypeManager登録 + compile関数


- [x] `leaky_relu(negative_slope?)` — LeakyReLU
  - [x] CPU/Metal: element-wise kernel (x > 0 ? x : slope * x)
  - [x] FFI + TypeManager登録 + compile関数
  - [x] テスト: slope=0.01,0.1 確認済み

- [x] `group_norm(num_groups, weight, bias, eps)` — GroupNorm
  - [x] CPU/Metal 実装、CUDA stub

- [x] `adaptive_avg_pool2d(output_size)` — 適応的プーリング
  - [x] CPU/Metal 実装、CUDA stub

- [x] `conv1d(weight, bias, stride, padding)` — 1D畳み込み
  - [x] CPU/Metal 実装、CUDA stub

- [x] `conv_transpose2d(weight, bias, stride, padding, output_padding)` — 転置畳み込み
  - [x] CPU/Metal 実装、CUDA stub

- [x] `interpolate(output_h, output_w, mode?)` — リサイズ
  - [x] CPU/Metal: nearest(mode=0) / bilinear(mode=1)

- [x] `pad(padding, value?)` — パディング
  - [x] CPU/Metal 実装、CUDA stub
  - [x] テスト: pad(2,1)=[0,0,1,2,3,0]

### 4.4 低優先度（活性化関数）

- [x] `elu(alpha?)` — ELU活性化
  - [x] CPU/Metal 実装、CUDA stub
- [x] `mish()` — Mish活性化
  - [x] CPU/Metal 実装、CUDA stub
- [x] `hardswish()` / `hardsigmoid()` — モバイル向け活性化
  - [x] CPU/Metal 実装、CUDA stub
  - [x] テスト: hardswish([-6,-3,0,3,6])=[-0,-0,0,3,6]
- [x] `dropout2d(p, training)` — チャネル単位ドロップアウト
  - [x] CPU/Metal 実装、CUDA stub
- [x] `instance_norm(...)` — InstanceNorm
  - [x] group_norm(num_groups=channels) として実装

---

## フェーズ 5: 学習・最適化

### 5.1 オプティマイザ（高優先度）

- [x] `Adam` / `AdamW` オプティマイザ
  - [x] ランタイム FFI: `tl_adam_step(param, grad, m, v, step, lr, beta1, beta2, eps, weight_decay)`
  - [x] weight_decay > 0 の場合は AdamW として動作
  - [x] builtins LLVM宣言 + グローバルマッピング

### 5.2 損失関数（高〜中優先度）

- [x] `mse_loss(pred, target)` — 平均二乗誤差
  - [x] CPU/Metal: mean((pred-target)²)
  - [x] FFI + TypeManager登録
  - [x] テスト: mse_loss([3,3,3,3],[1,1,1,1]) = [4.0]

- [x] `bce_loss(pred, target)` — 二値クロスエントロピー
  - [x] CPU/Metal: -mean(y*log(p)+(1-y)*log(1-p))
  - [x] FFI + TypeManager登録
  - [x] テスト: bce_loss(0.9,1.0)=[0.1054]

- [x] `l1_loss(pred, target)` — L1損失
  - [x] CPU/Metal: mean(|pred-target|)
  - [x] FFI + TypeManager登録
  - [x] テスト: l1_loss([3,3,3,3],[1,1,1,1]) = [2.0]

- [x] `nll_loss(pred, target)` — 負の対数尤度
  - [x] CPU/Metal: -mean(pred*target)
  - [x] FFI + TypeManager登録 + compile関数

- [x] `kl_div_loss(pred, target)` — KLダイバージェンス
  - [x] CPU/Metal: mean(q*log(q/p))

### 5.3 学習ユーティリティ（高優先度）

- [x] `Param::zero_grad()` — 勾配ゼロ化
  - [x] ランタイム: `tl_clear_grads` 既存
  - [x] Param::zero_grad() として static_methods 登録

- [x] ~~`no_grad { ... }` ブロック~~ — **不要**: TLでは Tensor/GradTensor 型で勾配追跡を静的に分離するため、ランタイムフラグは不要

### 5.4 中優先度

- [x] `SGD` with momentum
  - [x] ランタイム FFI: `tl_sgd_step(param, grad, velocity, lr, momentum, weight_decay, dampening, nesterov)`

- [x] 学習率スケジューラ (CosineAnnealing, StepLR)
  - [x] `tl_lr_cosine_annealing(base_lr, step, total_steps, min_lr) -> f32`
  - [x] `tl_lr_step(base_lr, step, step_size, gamma) -> f32`

- [x] ~~`Param::parameters() -> Vec<Tensor>`~~ — **不要**: GradTensor型で個別管理する設計のため

- [x] `Param::freeze()` / `unfreeze()` — パラメータ凍結
  - [x] GradTensor のインスタンスメソッドとして実装済み (`GradTensor::freeze()` / `unfreeze()`)

- [x] 勾配クリッピング (`clip_grad_norm`, `clip_grad_value`)
  - [x] GradTensor のインスタンスメソッドとして実装済み

---

## フェーズ 6: データ前処理・Image

### 6.1 高優先度

- [x] `Image::load_rgb(path)` — カラー画像ロード
  - [x] ランタイム FFI: `tl_image_load_rgb` (image crate → Tensor[3,H,W])
  - [x] builtins LLVM宣言 + グローバルマッピング

- [x] `DataLoader` — バッチデータローダー
  - [x] ランタイム FFI: `tl_dataloader_new/len/reset/free`
  - [x] シャッフル + バッチ分割

### 6.2 中優先度

- [x] CSV / JSON パーサ
  - [x] ランタイム FFI: `tl_csv_load` (csv crate → 2D Tensor), `tl_json_load` (serde_json)
  - [x] Cargo.toml に csv/serde_json 依存追加

- [x] `Image::resize(t, w, h)` — 画像リサイズ
  - [x] ランタイム FFI: `tl_image_resize` (bilinear interpolation)

- [x] `Image::save(t, path)` — テンソルを画像保存
  - [x] ランタイム FFI: `tl_image_save`

- [x] `Image::normalize(t, mean, std)` — 正規化
  - [x] ランタイム FFI: `tl_image_normalize` (チャネルwise)

- [x] `Image::crop(t, x, y, w, h)` — クロップ
  - [x] ランタイム FFI: `tl_image_crop`

---

## フェーズ 7: LLM/Transformer 推論

### 7.1 高優先度

- [x] `scaled_dot_product_attention(q, k, v, mask?)` — Fused Attention
  - [x] CPU/Metal: Q×K^T / √d + mask → softmax → ×V
  - [x] CUDA stub
  - [x] FFI + TypeManager登録（Tensor.sdpa(q,k,v[,mask])）

- [x] `top_k_sample(logits, k)` — Top-Kサンプリング
  - [x] CPU/Metal: sort → mask → softmax → argmax
  - [x] テスト: top_k(2) logits=[1,5,2,8,3] → [3.0]

- [x] `top_p_sample(logits, p)` — Top-P (Nucleus) サンプリング
  - [x] CPU/Metal: sort → cumsum → mask → renormalize → argmax
  - [x] テスト: top_p(0.9) → [3.0]

- [x] `KVCache::clear()` — キャッシュクリア
  - [x] ランタイム: `tl_kv_cache_clear` + ラッパー `tl_kvcache_clear`
  - [x] builtins LLVM宣言 + グローバルマッピング
  - [x] TypeManager登録 + compile関数

### 7.2 中優先度

- [x] Flash Attention サポート
  - [x] 既存の `scaled_dot_product_attention` をGPU opsチェーンで実装
  - [x] tl_metal: CPU fallback排除、既存Metal ops（matmul/transpose/softmax/scale）で構築
  - [ ] tl_metal: tiled attention kernel (将来最適化)
  - [ ] tl_cuda: cutlass/flash-attn integration (将来最適化)

- [x] `temperature_scale(logits, t)` — 温度スケーリング
  - [x] CPU/Metal: logits / temperature

- [x] `repetition_penalty(logits, tokens, penalty)` — 繰り返しペナルティ
  - [x] CPU/Metal: 正の値は/p、負の値は*p

- [x] `KVCache::len()` — キャッシュ長取得
  - [x] ランタイム実装 (`tl_kv_cache_len`)
  - [x] コンパイラ組み込み

- [x] `KVCache::resize(max_len)` — 最大長変更
  - [x] ランタイム実装 (`tl_kv_cache_resize`)
  - [x] コンパイラ組み込み

---

## フェーズ 8: 線形代数（低〜中優先度）

- [x] `dot(other)` — ベクトル内積
  - [x] CPU/Metal: sum(a*b)

- [x] `inverse()` — 逆行列
  - [x] CPU: ガウス・ジョルダン消去法（ピボット選択付き）
  - [x] GPU: CPU fallback

- [x] `det()` — 行列式
  - [x] CPU: LU分解ベース
- [x] `svd()` — 特異値分解
  - [x] `svd_u()`, `svd_s()`, `svd_v()` として個別返却
  - [x] CPU: stub実装（対角要素ベース）
- [x] `eig()` — 固有値分解
  - [x] `eig_values()`, `eig_vectors()` として個別返却
  - [x] CPU: stub実装
- [x] `solve(b)` — 連立方程式
  - [x] CPU: LU分解ベース（ピボット選択付き）

---

## 検証計画

### 自動テスト

各フェーズ完了時に以下を実行:

1. **Rust ユニットテスト**
   ```bash
   cargo test --workspace
   ```
   各 crate (tl_cpu, tl_metal, tl_cuda) に追加したテストが通ること。

2. **TLファイル検証**
   ```bash
   python3 scripts/verify_tl_files.py
   ```
   既存のすべての `.tl` テストファイルが引き続きパスすること。

3. **新規TLテストファイル**
   各フェーズで `tests/test_<feature>.tl` を追加し、TL言語レベルで動作を確認。

### 手動検証

- フェーズ5（Adam/損失関数）: MNIST等の簡単な学習タスクで収束を確認
- フェーズ7（SDPA/サンプリング）: 既存のLLM推論デモで品質を確認

---

## 実装順序（実際に実施した順序）

```
フェーズ1（テンソル生成）→ フェーズ2（要素操作）→ フェーズ4（NN層）
                                                  ↓
フェーズ3（形状操作）→ フェーズ5（学習・最適化）→ フェーズ6（データ）
                                                  ↓
                                          フェーズ7（LLM推論）
                                                  ↓
                                          フェーズ8（線形代数）
```

全フェーズの CPU/Metal 実装は 2026-03 に完了。
CUDA は各フェーズで `unimplemented!()` stub を配置済み。
