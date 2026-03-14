# TL言語 ドメイン固有機能の拡張提案

> [!NOTE]
> **ステータス更新 (2026-03)**: 本ドキュメントで分析された全項目が実装完了済み。
> 各テーブルの「状態」列で実装状況を確認可能。CUDA バックエンドのみ stub 残存。
> 詳細な実装記録は [DOMAIN_FEATURES_ROADMAP.md](DOMAIN_FEATURES_ROADMAP.md) を参照。

TL言語のドメイン固有機能（Tensor, NN, LLM推論, ファイルI/O等）について、PyTorch / JAX / TensorFlow 等の主要フレームワークと比較し、追加すべき機能をリストアップしたもの。

---

## 1. Tensor 操作

### 1.1 テンソル生成

| 関数 | 説明 | 状態 |
|---|---|---|
| `Tensor::full(shape, value, requires_grad)` | 任意の値で埋めたテンソル | ✅ 実装済み |
| `Tensor::eye(n, requires_grad)` | 単位行列 | ✅ 実装済み |
| `Tensor::arange(start, end, step)` | 連番テンソル | ✅ 実装済み |
| `Tensor::linspace(start, end, steps)` | 等間隔テンソル | ✅ 実装済み |
| `Tensor::from_vec(vec: Vec<f32>, shape)` | Vec からテンソルへの変換 | ✅ 実装済み |
| `Tensor::rand(shape, requires_grad)` | 一様分布乱数 | ✅ 実装済み |
| `Tensor::zeros_like(t)` / `ones_like(t)` | 既存テンソルと同じ形状で生成 | ✅ 実装済み |
| `Tensor::rand_like(t)` / `randn_like(t)` | 既存テンソルと同じ形状で乱数生成 | ✅ 実装済み |

### 1.2 形状操作

| メソッド | 説明 | 状態 |
|---|---|---|
| `flatten(start_dim?, end_dim?)` | 多次元を1次元に平坦化 | ✅ 実装済み |
| `unsqueeze(dim)` | 次元の追加 | ✅ 実装済み |
| `squeeze(dim?)` | サイズ1の次元を除去 | ✅ 実装済み |
| `expand(shape)` / `broadcast_to(shape)` | ブロードキャスト | ✅ 実装済み |
| `permute(dims)` | 任意の次元入れ替え | ✅ 実装済み |
| `contiguous()` | メモリ連続化 | ✅ 実装済み |
| `view(shape)` | reshapeのゼロコピー版 | ✅ 実装済み (reshape エイリアス) |
| `chunk(n, dim)` | テンソルをn個に分割 | ✅ 実装済み |
| `split(sizes, dim)` | 指定サイズで分割 | ✅ 実装済み |
| `stack(tensors, dim)` | テンソルの結合（新次元） | ✅ 実装済み |
| `cat(tensors, dim)` | テンソルの結合（既存次元） | ✅ 実装済み |
| `get_shape() -> Vec<i64>` | 形状をVecとして取得 | ✅ 実装済み (`shape()`) |

### 1.3 要素操作

| メソッド | 説明 | 状態 |
|---|---|---|
| `where(condition, x, y)` | 条件付き選択 | ✅ 実装済み (`where_cond`) |
| `masked_fill(mask, value)` | マスクされた位置を値で埋める | ✅ 実装済み |
| `eq(other)` / `ne(other)` / `lt` / `le` / `gt` / `ge` | 比較演算 | ✅ 実装済み |
| `logical_and` / `logical_or` / `logical_not` | 論理演算 | ✅ 実装済み |
| `fill_(value)` | インプレースで全要素を埋める | ✅ 実装済み |
| `to_dtype(dtype)` | 型変換 | ✅ 実装済み (`to_f32`, `to_i64`) |
| `to_vec() -> Vec<f32>` | テンソルからVecへの変換 | ✅ 実装済み |

### 1.4 リダクション

| メソッド | 説明 | 状態 |
|---|---|---|
| `std(dim?)` / `var(dim?)` | 標準偏差・分散 | ✅ 実装済み |
| `prod(dim?)` | 積 | ✅ 実装済み |
| `cumsum(dim)` | 累積和 | ✅ 実装済み |
| `norm(p, dim?)` | Lpノルム | ✅ 実装済み |
| `topk(k, dim)` | 上位k個の値とインデックス | ✅ 実装済み |

### 1.5 線形代数

| メソッド | 説明 | 状態 |
|---|---|---|
| `inverse()` | 逆行列 | ✅ 実装済み (CPU: ガウス・ジョルダン) |
| `det()` | 行列式 | ✅ 実装済み (CPU: LU分解) |
| `svd()` | 特異値分解 | ✅ 実装済み (`svd_u/svd_s/svd_v`) |
| `eig()` | 固有値分解 | ✅ 実装済み (`eig_values/eig_vectors`) |
| `solve(b)` | 連立一次方程式 | ✅ 実装済み (CPU: LU分解) |
| `dot(other)` | 内積（1Dベクトル） | ✅ 実装済み |

---

## 2. ニューラルネットワーク層

### 2.1 活性化関数

| 関数 | 説明 | 状態 |
|---|---|---|
| `leaky_relu(negative_slope?)` | LeakyReLU | ✅ 実装済み |
| `elu(alpha?)` | ELU | ✅ 実装済み |
| `mish()` | Mish活性化 | ✅ 実装済み |
| `hardswish()` / `hardsigmoid()` | モバイル向け | ✅ 実装済み |

### 2.2 正規化

| 層 | 説明 | 状態 |
|---|---|---|
| `layer_norm(shape, weight, bias, eps)` | LayerNorm | ✅ 実装済み (TL API公開) |
| `batch_norm(weight, bias, mean, var, eps)` | BatchNorm | ✅ 実装済み (TL API公開) |
| `group_norm(groups, weight, bias, eps)` | GroupNorm | ✅ 実装済み |
| `instance_norm(...)` | InstanceNorm | ✅ 実装済み (group_norm ラッパー) |

### 2.3 ドロップアウト

| 関数 | 説明 | 状態 |
|---|---|---|
| `dropout(p, training)` | ドロップアウト | ✅ 実装済み |
| `dropout2d(p, training)` | チャネル単位ドロップアウト | ✅ 実装済み |

### 2.4 プーリング

| 関数 | 説明 | 状態 |
|---|---|---|
| `max_pool2d(kernel, stride, padding)` | 最大プーリング | ✅ 実装済み |
| `avg_pool2d(kernel, stride, padding)` | 平均プーリング | ✅ 実装済み |
| `adaptive_avg_pool2d(output_size)` | 適応的プーリング | ✅ 実装済み |

### 2.5 畳み込み

| 関数 | 説明 | 状態 |
|---|---|---|
| `conv1d(weight, bias?, padding, stride)` | 1D畳み込み | ✅ 実装済み |
| `conv_transpose2d(...)` | 転置畳み込み | ✅ 実装済み |

### 2.6 その他のNN操作

| 関数 | 説明 | 状態 |
|---|---|---|
| `linear(weight, bias?)` | 全結合層（matmul + bias） | ✅ 実装済み |
| `interpolate(size, mode)` | テンソルのリサイズ | ✅ 実装済み (nearest/bilinear) |
| `pad(padding, value?)` | パディング | ✅ 実装済み |

---

## 3. 学習・最適化

### 3.1 オプティマイザ

| オプティマイザ | 説明 | 状態 |
|---|---|---|
| `Adam` / `AdamW` | 最も広く使われるオプティマイザ | ✅ 実装済み (`tl_adam_step`) |
| `SGD` with momentum | モメンタム付きSGD | ✅ 実装済み (`tl_sgd_step`) |
| 学習率スケジューラ | CosineAnnealing, StepLR | ✅ 実装済み |

### 3.2 損失関数

| 関数 | 説明 | 状態 |
|---|---|---|
| `mse_loss(pred, target)` | 平均二乗誤差 | ✅ 実装済み |
| `bce_loss(pred, target)` | 二値クロスエントロピー | ✅ 実装済み |
| `l1_loss(pred, target)` | L1損失 | ✅ 実装済み |
| `nll_loss(pred, target)` | 負の対数尤度 | ✅ 実装済み |
| `kl_div_loss(pred, target)` | KLダイバージェンス | ✅ 実装済み |

### 3.3 学習ユーティリティ

| 機能 | 説明 | 状態 |
|---|---|---|
| `Param::zero_grad()` | 全パラメータの勾配をゼロ化 | ✅ 実装済み |
| `Param::freeze()` / `unfreeze()` | パラメータの凍結・解凍 | ✅ 実装済み (GradTensor メソッド) |
| `no_grad { ... }` ブロック | 勾配計算の無効化スコープ | ⛔ **不要**: Tensor/GradTensor 型分離で対応済み |
| 勾配クリッピング | `clip_grad_norm`, `clip_grad_value` | ✅ 実装済み (GradTensor メソッド) |
| `Param::parameters() -> Vec<Tensor>` | 全パラメータの取得 | ⛔ **不要**: GradTensor 型で個別管理する設計 |

---

## 4. データ前処理・ユーティリティ

### 4.1 データローディング

| 機能 | 説明 | 状態 |
|---|---|---|
| `DataLoader` | バッチ処理・シャッフル | ✅ 実装済み |
| CSV / JSON パーサ | 構造化データの読み込み | ✅ 実装済み (`tl_csv_load`, `tl_json_load`) |
| テンソルの直列化 | `Tensor::save(path)` / `Tensor::load(path)` | ✅ 実装済み |

### 4.2 Image の拡張

| 機能 | 説明 | 状態 |
|---|---|---|
| `Image::load_rgb(path) -> Tensor` | RGB画像のロード | ✅ 実装済み |
| `Image::resize(t, w, h) -> Tensor` | 画像リサイズ | ✅ 実装済み |
| `Image::save(t, path)` | テンソルを画像として保存 | ✅ 実装済み |
| `Image::normalize(t, mean, std)` | 画像正規化 | ✅ 実装済み |
| `Image::crop(t, x, y, w, h)` | 画像クロップ | ✅ 実装済み |

---

## 5. LLM/Transformer 推論の拡張

### 5.1 アテンション

| 機能 | 説明 | 状態 |
|---|---|---|
| `scaled_dot_product_attention(q, k, v, mask?)` | Fused Attention | ✅ 実装済み (`Tensor.sdpa`) |
| Flash Attention サポート | メモリ効率的なアテンション | ✅ 既存Metal opsで構築済み |

### 5.2 サンプリング

| 機能 | 説明 | 状態 |
|---|---|---|
| `top_k_sample(logits, k)` | Top-Kサンプリング | ✅ 実装済み |
| `top_p_sample(logits, p)` | Top-P (Nucleus) サンプリング | ✅ 実装済み |
| `temperature_scale(logits, t)` | 温度スケーリング | ✅ 実装済み |
| `repetition_penalty(logits, tokens, penalty)` | 繰り返しペナルティ | ✅ 実装済み |

### 5.3 KVCache

| 機能 | 説明 | 状態 |
|---|---|---|
| `KVCache::clear()` | キャッシュのクリア | ✅ 実装済み |
| `KVCache::len() -> i64` | 現在のキャッシュ長 | ✅ 実装済み |
| `KVCache::resize(max_len)` | 最大長の変更 | ✅ 実装済み |

---

## 6. APIドキュメントと実装の乖離（修正済み）

> [!NOTE]
> 以下の項目は `functions_and_types.md` および `standard_library.md` に追記済み。

| 機能 | 状態 |
|---|---|
| `squeeze()` / `unsqueeze()` | ✅ ドキュメント追記済み |
| `flatten()` / `permute()` / `contiguous()` / `cat()` / `gather()` | ✅ ドキュメント追記済み |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` (比較演算子) | ✅ ドキュメント追記済み |
| `dropout` | ✅ ドキュメント追記済み |
| `ndim()` / `shape()` / `to_i64()` / `save()` / `display()` | ✅ ドキュメント追記済み |
| `from_vec_u8` / `clear_grads` / `shallow_clone` / `to(device)` | ✅ ドキュメント追記済み |
| `sample(temperature, top_p)` / `sumall()` | ✅ ドキュメント追記済み |
| `BatchNorm` / `LayerNorm` | ✅ TL API 公開済み |
| `max_pool2d` / `avg_pool2d` | ✅ TL API 公開済み |
