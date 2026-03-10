# TL言語 ドメイン固有機能 実装ロードマップ

> [!CAUTION]
> **実装を開始する前に、必ず [MEMORY_MANAGEMENT_STRATEGY.md](MEMORY_MANAGEMENT_STRATEGY.md) を読むこと。**
> TLのメモリ管理はARCレジストリとプール制御に基づいており、新しいテンソル操作やNN層を追加する際にはこの戦略に従う必要がある。

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

- [ ] `masked_fill(mask, value)` — マスク付き値埋め
  - [ ] tl_cpu: `masked_fill_impl`
  - [ ] tl_metal: GPU kernel (condition ? value : original)
  - [ ] tl_cuda: `masked_fill_impl`
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `to_dtype(dtype: String)` — 型変換
  - [ ] tl_cpu: f32↔f64, f32↔i64 変換
  - [ ] tl_metal: GPU cast kernel
  - [ ] tl_cuda: GPU cast kernel
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `to_vec() -> Vec<f32>` — テンソルからVecへの変換
  - [ ] tl_cpu: データコピー
  - [ ] tl_metal: GPU→CPU転送 + Vecラップ
  - [ ] tl_cuda: GPU→CPU転送 + Vecラップ
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `std(dim?)` / `var(dim?)` — 標準偏差・分散
  - [ ] tl_cpu: `var_impl`, `std_impl`
  - [ ] tl_metal: GPU reduction (mean→差の二乗→mean)
  - [ ] tl_cuda: 同上
  - [ ] FFI + TypeManager登録
  - [ ] テスト

### 2.2 中優先度

- [ ] `logical_and` / `logical_or` / `logical_not` — 論理演算
  - [ ] 全バックエンド: element-wise kernel
  - [ ] FFI + TypeManager登録

- [ ] `fill_(value)` — インプレース埋め込み
  - [ ] 全バックエンド実装

- [ ] `prod(dim?)` — 全要素の積
  - [ ] 全バックエンド: reduction kernel
  - [ ] FFI + TypeManager登録

- [ ] `cumsum(dim)` — 累積和
  - [ ] 全バックエンド: parallel prefix sum
  - [ ] FFI + TypeManager登録

- [ ] `norm(p, dim?)` — Lpノルム
  - [ ] 全バックエンド実装

- [ ] `topk(k, dim)` — 上位k個
  - [ ] 全バックエンド: partial sort kernel
  - [ ] FFI + TypeManager登録

---

## フェーズ 3: 形状操作（追加）

### 3.1 中優先度

- [ ] `expand(shape)` / `broadcast_to(shape)` — 明示的ブロードキャスト
  - [ ] バックエンド: `broadcast_to_impl` は内部に存在する可能性あり → TL API公開
  - [ ] FFI + TypeManager登録

- [ ] `view(shape)` — ゼロコピーreshape
  - [ ] reshapeのエイリアスまたはcontiguous check付き

- [ ] `chunk(n, dim)` — n分割
  - [ ] 全バックエンド: narrow呼び出しのラッパー
  - [ ] FFI + TypeManager登録

- [ ] `split(sizes, dim)` — 指定サイズ分割
  - [ ] 全バックエンド: narrow呼び出しのラッパー
  - [ ] FFI + TypeManager登録

- [ ] `stack(tensors, dim)` — 新次元で結合
  - [ ] 全バックエンド: unsqueeze + cat
  - [ ] FFI + TypeManager登録

---

## フェーズ 4: ニューラルネットワーク層

### 4.1 正規化（高優先度）

- [ ] `layer_norm(normalized_shape, weight, bias, eps)` — LayerNorm
  - [ ] tl_cpu: 実装
  - [ ] tl_metal: バックエンドに実装パターンあり(Two-Pass) → TL API公開
  - [ ] tl_cuda: 実装
  - [ ] FFI + TypeManager登録
  - [ ] テスト: `tests/test_layer_norm.tl`

- [ ] `batch_norm(weight, bias, running_mean, running_var, eps, training)` — BatchNorm
  - [ ] tl_cpu: 実装
  - [ ] tl_metal: バックエンドに実装パターンあり → TL API公開
  - [ ] tl_cuda: 実装
  - [ ] FFI + TypeManager登録
  - [ ] テスト

### 4.2 プーリング（高優先度）

- [ ] `max_pool2d(kernel_size, stride, padding)` — 最大プーリング
  - [ ] tl_cpu: 実装
  - [ ] tl_metal: ディスパッチパターンあり → TL API公開
  - [ ] tl_cuda: 実装
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `avg_pool2d(kernel_size, stride, padding)` — 平均プーリング
  - [ ] tl_cpu: 実装
  - [ ] tl_metal: 実装
  - [ ] tl_cuda: 実装
  - [ ] FFI + TypeManager登録
  - [ ] テスト

### 4.3 その他NN（高〜中優先度）

- [ ] `linear(weight, bias?)` — 全結合層
  - [ ] matmul + optional bias add のラッパー
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `leaky_relu(negative_slope?)` — LeakyReLU
  - [ ] 全バックエンド: element-wise kernel (x > 0 ? x : slope * x)
  - [ ] FFI + TypeManager登録

- [ ] `group_norm(num_groups, weight, bias, eps)` — GroupNorm
  - [ ] 全バックエンド実装

- [ ] `adaptive_avg_pool2d(output_size)` — 適応的プーリング
  - [ ] 全バックエンド実装

- [ ] `conv1d(weight, bias?, padding, stride)` — 1D畳み込み
  - [ ] 全バックエンド実装

- [ ] `conv_transpose2d(weight, bias?, padding, stride, output_padding)` — 転置畳み込み
  - [ ] 全バックエンド実装

- [ ] `interpolate(size, mode)` — リサイズ
  - [ ] 全バックエンド: bilinear/nearest kernel

- [ ] `pad(padding, value?)` — パディング
  - [ ] 全バックエンド実装

### 4.4 低優先度（活性化関数）

- [ ] `elu(alpha?)` — ELU活性化
- [ ] `mish()` — Mish活性化
- [ ] `hardswish()` / `hardsigmoid()` — モバイル向け活性化
- [ ] `dropout2d(p, training)` — チャネル単位ドロップアウト
- [ ] `instance_norm(...)` — InstanceNorm

---

## フェーズ 5: 学習・最適化

### 5.1 オプティマイザ（高優先度）

- [ ] `Adam` / `AdamW` オプティマイザ
  - [ ] 状態管理構造: m (1st moment), v (2nd moment), step count
  - [ ] TLライブラリ実装（`lib/adam.tl`）またはランタイム実装
  - [ ] `Param::optimizer("adam", lr, beta1, beta2, eps, weight_decay)`
  - [ ] `Param::step()` でオプティマイザの更新適用
  - [ ] テスト: 単純な最適化問題で収束確認

### 5.2 損失関数（高〜中優先度）

- [ ] `mse_loss(pred, target)` — 平均二乗誤差
  - [ ] 全バックエンド: (pred - target)^2 の mean
  - [ ] Autograd対応
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `bce_loss(pred, target)` — 二値クロスエントロピー
  - [ ] 全バックエンド実装

- [ ] `l1_loss(pred, target)` — L1損失
  - [ ] 全バックエンド: |pred - target| の mean

- [ ] `nll_loss(pred, target)` — 負の対数尤度
  - [ ] 全バックエンド実装

- [ ] `kl_div_loss(pred, target)` — KLダイバージェンス（低優先度）

### 5.3 学習ユーティリティ（高優先度）

- [ ] `Param::zero_grad()` — 勾配ゼロ化
  - [ ] ランタイム: 全登録パラメータの勾配をクリア
  - [ ] FFI + TypeManager登録

- [ ] `no_grad { ... }` ブロック — 勾配計算無効化
  - [ ] コンパイラ: 新しい構文の追加（Parser/AST/Codegen）
  - [ ] ランタイム: グローバル勾配有効フラグの切り替え
  - [ ] テスト

### 5.4 中優先度

- [ ] `SGD` with momentum
  - [ ] ランタイムまたはTLライブラリ実装

- [ ] 学習率スケジューラ (CosineAnnealing, StepLR等)
  - [ ] TLライブラリ実装

- [ ] `Param::parameters() -> Vec<Tensor>` — 全パラメータ取得

- [ ] `Param::freeze()` / `unfreeze()` — パラメータ凍結

- [ ] 勾配クリッピング (`clip_grad_norm`, `clip_grad_value`)
  - [ ] ランタイム実装

---

## フェーズ 6: データ前処理・Image

### 6.1 高優先度

- [ ] `Image::load(path)` / `Image::load_rgb(path)` — カラー画像ロード
  - [ ] ランタイム: `image` crate を使用してRGB読み込み → Tensor[3,H,W]
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `DataLoader` — バッチデータローダー
  - [ ] TLライブラリまたはランタイム: シャッフル / バッチ分割
  - [ ] イテレータプロトコル対応

### 6.2 中優先度

- [ ] CSV / JSON パーサ
  - [ ] ランタイム: `serde_json` / `csv` crate
  - [ ] FFI + TypeManager登録

- [ ] `Image::resize(t, w, h)` — 画像リサイズ
  - [ ] ランタイム: image crate のresize
  - [ ] FFI登録

- [ ] `Image::save(t, path)` — テンソルを画像保存
  - [ ] ランタイム実装

- [ ] `Image::normalize(t, mean, std)` — 正規化
  - [ ] テンソル演算のラッパー

- [ ] `Image::crop(t, x, y, w, h)` — クロップ（低優先度）

---

## フェーズ 7: LLM/Transformer 推論

### 7.1 高優先度

- [ ] `scaled_dot_product_attention(q, k, v, mask?)` — Fused Attention
  - [ ] tl_metal: 専用 Metal kernel (Q×K^T / √d + mask → softmax → ×V)
  - [ ] tl_cpu: matmul + scale + mask + softmax + matmul の合成
  - [ ] tl_cuda: 専用 CUDA kernel
  - [ ] FFI + TypeManager登録（グローバル関数 or Tensor メソッド）
  - [ ] テスト: アテンション出力の数値検証

- [ ] `top_k_sample(logits, k)` — Top-Kサンプリング
  - [ ] 全バックエンド: partial sort → mask → renormalize → sample
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `top_p_sample(logits, p)` — Top-P (Nucleus) サンプリング
  - [ ] 全バックエンド: sort → cumsum → mask → renormalize → sample
  - [ ] FFI + TypeManager登録
  - [ ] テスト

- [ ] `KVCache::clear()` — キャッシュクリア
  - [ ] ランタイム実装
  - [ ] FFI + TypeManager登録

### 7.2 中優先度

- [ ] Flash Attention サポート
  - [ ] tl_metal: tiled attention kernel
  - [ ] tl_cuda: cutlass/flash-attn integration

- [ ] `temperature_scale(logits, t)` — 温度スケーリング
  - [ ] テンソル演算のラッパー (logits / temperature)

- [ ] `repetition_penalty(logits, tokens, penalty)` — 繰り返しペナルティ
  - [ ] 全バックエンド実装

- [ ] `KVCache::len()` — キャッシュ長取得
  - [ ] ランタイム実装

- [ ] `KVCache::resize(max_len)` — 最大長変更（低優先度）

---

## フェーズ 8: 線形代数（低〜中優先度）

- [ ] `dot(other)` — ベクトル内積
  - [ ] 全バックエンド: 1D matmul のラッパー

- [ ] `inverse()` — 逆行列
  - [ ] CPU: LAPACK / 手動実装
  - [ ] GPU: batched LU decomposition

- [ ] `det()` — 行列式（低優先度）
- [ ] `svd()` — 特異値分解（低優先度）
- [ ] `eig()` — 固有値分解（低優先度）
- [ ] `solve(b)` — 連立方程式（低優先度）

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

## 実装順序の推奨

```
フェーズ1（テンソル生成）→ フェーズ2（要素操作）→ フェーズ4（NN層）
                                                  ↓
フェーズ3（形状操作）→ フェーズ5（学習・最適化）→ フェーズ6（データ）
                                                  ↓
                                          フェーズ7（LLM推論）
                                                  ↓
                                          フェーズ8（線形代数）
```

フェーズ1/2は他のすべての基盤となるため最初に実装する。
フェーズ4/5はNN学習に必須であり、フェーズ6/7はアプリケーション層。
