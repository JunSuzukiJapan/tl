# TL言語 ドメイン固有機能の拡張提案

TL言語のドメイン固有機能（Tensor, NN, LLM推論, ファイルI/O等）について、PyTorch / JAX / TensorFlow 等の主要フレームワークと比較し、追加すべき機能をリストアップする。

---

## 1. Tensor 操作の不足

### 1.1 テンソル生成

| 関数 | 説明 | 優先度 |
|---|---|---|
| `Tensor::full(shape, value, requires_grad)` | 任意の値で埋めたテンソル | **高** |
| `Tensor::eye(n, requires_grad)` | 単位行列 | **高** |
| `Tensor::arange(start, end, step)` | 連番テンソル | **高** |
| `Tensor::linspace(start, end, steps)` | 等間隔テンソル | 中 |
| `Tensor::from_vec(vec: Vec<f32>, shape)` | Vec からテンソルへの変換 | **高** |
| `Tensor::rand(shape, requires_grad)` | 一様分布乱数（`randn`は正規分布のみ） | 中 |
| `Tensor::zeros_like(t)` / `ones_like(t)` | 既存テンソルと同じ形状で生成 | **高** |
| `Tensor::rand_like(t)` / `randn_like(t)` | 既存テンソルと同じ形状で乱数生成 | 中 |

### 1.2 形状操作

| メソッド | 説明 | 優先度 |
|---|---|---|
| `flatten(start_dim?, end_dim?)` | 多次元を1次元に平坦化 | **高** |
| `unsqueeze(dim)` | 次元の追加（KI には記載あるがAPIドキュメント未記載） | **高** |
| `squeeze(dim?)` | サイズ1の次元を除去（KI には記載あるがAPIドキュメント未記載） | **高** |
| `expand(shape)` / `broadcast_to(shape)` | ブロードキャスト | 中 |
| `permute(dims)` | 任意の次元入れ替え（`transpose`は2次元のみ） | 中 |
| `contiguous()` | メモリ連続化 | 中 |
| `view(shape)` | reshapeのゼロコピー版 | 中 |
| `chunk(n, dim)` | テンソルをn個に分割 | 中 |
| `split(sizes, dim)` | 指定サイズで分割 | 中 |
| `stack(tensors, dim)` | テンソルの結合（新次元） | 中 |
| `cat(tensors, dim)` | テンソルの結合（既存次元、可変長対応） | 中 |
| `get_shape() -> Vec<i64>` | 形状をVecとして取得 | **高** |

### 1.3 要素操作

| メソッド | 説明 | 優先度 |
|---|---|---|
| `where(condition, x, y)` | 条件付き選択（KIには`where_cond`あり、APIドキュメント未記載） | **高** |
| `masked_fill(mask, value)` | マスクされた位置を値で埋める | **高** |
| `eq(other)` / `ne(other)` / `lt` / `le` / `gt` / `ge` | 比較演算（KIにはあるがAPIドキュメント未記載） | **高** |
| `logical_and` / `logical_or` / `logical_not` | 論理演算 | 中 |
| `fill_(value)` | インプレースで全要素を埋める | 中 |
| `to_dtype(dtype)` | 型変換（f32→f64, f32→i64 等） | **高** |
| `to_vec() -> Vec<f32>` | テンソルからVecへの変換 | **高** |

### 1.4 リダクション（追加）

| メソッド | 説明 | 優先度 |
|---|---|---|
| `std(dim?)` / `var(dim?)` | 標準偏差・分散 | **高** |
| `prod(dim?)` | 積 | 中 |
| `cumsum(dim)` | 累積和 | 中 |
| `norm(p, dim?)` | Lpノルム | 中 |
| `topk(k, dim)` | 上位k個の値とインデックス | 中 |

### 1.5 線形代数

| メソッド | 説明 | 優先度 |
|---|---|---|
| `inverse()` | 逆行列 | 中 |
| `det()` | 行列式 | 低 |
| `svd()` | 特異値分解 | 低 |
| `eig()` | 固有値分解 | 低 |
| `solve(b)` | 連立一次方程式 | 低 |
| `dot(other)` | 内積（1Dベクトル） | 中 |

---

## 2. ニューラルネットワーク層の不足

### 2.1 活性化関数

| 関数 | 説明 | 優先度 |
|---|---|---|
| `leaky_relu(negative_slope?)` | LeakyReLU | 中 |
| `elu(alpha?)` | ELU | 低 |
| `mish()` | Mish活性化 | 低 |
| `hardswish()` / `hardsigmoid()` | モバイル向け | 低 |

### 2.2 正規化

| 層 | 説明 | 優先度 |
|---|---|---|
| `layer_norm(shape, weight, bias, eps)` | LayerNorm（KIには言及あるが、APIに未公開） | **高** |
| `batch_norm(weight, bias, mean, var, eps)` | BatchNorm（KIには言及あるが、APIに未公開） | **高** |
| `group_norm(groups, weight, bias, eps)` | GroupNorm | 中 |
| `instance_norm(...)` | InstanceNorm | 低 |

### 2.3 ドロップアウト

| 関数 | 説明 | 優先度 |
|---|---|---|
| `dropout(p, training)` | ドロップアウト（KIにはGPU実装あるが、API未公開） | **高** |
| `dropout2d(p, training)` | チャネル単位ドロップアウト | 低 |

### 2.4 プーリング

| 関数 | 説明 | 優先度 |
|---|---|---|
| `max_pool2d(kernel, stride, padding)` | 最大プーリング（KIに実装パターンあり） | **高** |
| `avg_pool2d(kernel, stride, padding)` | 平均プーリング | **高** |
| `adaptive_avg_pool2d(output_size)` | 適応的プーリング | 中 |

### 2.5 畳み込み

| 関数 | 説明 | 優先度 |
|---|---|---|
| `conv1d(weight, bias?, padding, stride)` | 1D畳み込み | 中 |
| `conv_transpose2d(...)` | 転置畳み込み（デコーダ用） | 中 |

### 2.6 その他のNN操作

| 関数 | 説明 | 優先度 |
|---|---|---|
| `linear(weight, bias?)` | 全結合層（matmul + bias） | **高** |
| `interpolate(size, mode)` | テンソルのリサイズ（bilinear等） | 中 |
| `pad(padding, value?)` | ゼロパディング / 定数パディング | 中 |

---

## 3. 学習・最適化の不足

### 3.1 オプティマイザ

現在は `Param::update_all(lr)` のみ（SGD相当）。

| オプティマイザ | 説明 | 優先度 |
|---|---|---|
| `Adam` / `AdamW` | 最も広く使われるオプティマイザ | **高** |
| `SGD` with momentum | モメンタム付きSGD | 中 |
| 学習率スケジューラ | CosineAnnealing, StepLR等 | 中 |

### 3.2 損失関数

| 関数 | 説明 | 優先度 |
|---|---|---|
| `mse_loss(pred, target)` | 平均二乗誤差 | **高** |
| `bce_loss(pred, target)` | 二値クロスエントロピー | 中 |
| `l1_loss(pred, target)` | L1損失 | 中 |
| `nll_loss(pred, target)` | 負の対数尤度 | 中 |
| `kl_div_loss(pred, target)` | KLダイバージェンス | 低 |

### 3.3 学習ユーティリティ

| 機能 | 説明 | 優先度 |
|---|---|---|
| `Param::zero_grad()` | 全パラメータの勾配をゼロ化 | **高** |
| `Param::parameters() -> Vec<Tensor>` | 全パラメータの取得 | 中 |
| `Param::freeze()` / `unfreeze()` | パラメータの凍結・解凍 | 中 |
| `no_grad { ... }` ブロック | 勾配計算の無効化スコープ | **高** |
| 勾配クリッピング | `clip_grad_norm`, `clip_grad_value` | 中 |

---

## 4. データ前処理・ユーティリティの不足

### 4.1 データローディング

| 機能 | 説明 | 優先度 |
|---|---|---|
| `DataLoader` | バッチ処理・シャッフル・並列読み込み | **高** |
| CSV / JSON パーサ | 構造化データの読み込み | 中 |
| テンソルの直列化 | `Tensor::save(path)` / `Tensor::load(path)` （loadはあるがsaveがない） | **高** |

### 4.2 Image の拡張

| 機能 | 説明 | 優先度 |
|---|---|---|
| `Image::load(path) -> Tensor` | カラー画像のロード（現在はgrayscaleのみ） | **高** |
| `Image::load_rgb(path) -> Tensor` | RGB画像のロード | **高** |
| `Image::resize(t, w, h) -> Tensor` | 画像リサイズ | 中 |
| `Image::save(t, path)` | テンソルを画像として保存 | 中 |
| `Image::normalize(t, mean, std)` | 画像正規化（ImageNet等） | 中 |
| `Image::crop(t, x, y, w, h)` | 画像クロップ | 低 |

---

## 5. LLM/Transformer 推論の拡張

### 5.1 アテンション

| 機能 | 説明 | 優先度 |
|---|---|---|
| `scaled_dot_product_attention(q, k, v, mask?)` | Fused Attention | **高** |
| Flash Attention サポート | メモリ効率的なアテンション | 中 |

### 5.2 サンプリング

| 機能 | 説明 | 優先度 |
|---|---|---|
| `top_k_sample(logits, k)` | Top-Kサンプリング | **高** |
| `top_p_sample(logits, p)` | Top-P (Nucleus) サンプリング | **高** |
| `temperature_scale(logits, t)` | 温度スケーリング | 中 |
| `repetition_penalty(logits, tokens, penalty)` | 繰り返しペナルティ | 中 |

### 5.3 KVCache の拡張

| 機能 | 説明 | 優先度 |
|---|---|---|
| `KVCache::clear()` | キャッシュのクリア | **高** |
| `KVCache::len() -> i64` | 現在のキャッシュ長 | 中 |
| `KVCache::resize(max_len)` | 最大長の変更 | 低 |

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
| `BatchNorm` / `LayerNorm` | ⚠️ バックエンド内部のみ（TL APIとして未公開） |
| `max_pool2d` / `avg_pool2d` | ⚠️ バックエンド内部のみ（TL APIとして未公開） |
| `System::exit(code)` | ⚠️ コンパイラ未登録 |
| `Arena` API | ⚠️ 内部API（公開対象外） |

---

## 7. 優先度サマリ

### 最優先
1. **テンソル生成**: `full`, `eye`, `arange`, `from_vec`, `zeros_like`/`ones_like`
2. **NN層**: `layer_norm`, `batch_norm`, `dropout`, `max_pool2d`, `avg_pool2d`, `linear`
3. **学習**: Adam/AdamW, `mse_loss`, `zero_grad`, `no_grad`スコープ
4. **形状操作**: `flatten`, `squeeze`/`unsqueeze` のドキュメント追記
5. **比較演算**: `eq`/`ne`/`lt`/`le`/`gt`/`ge` のドキュメント追記
6. **推論**: `scaled_dot_product_attention`, `top_k_sample`, `top_p_sample`
7. **データ**: `DataLoader`, カラー画像ロード, `Tensor::save`
8. **ドキュメント同期**: 既に実装済みだがドキュメントに未記載の機能を追記

### 中優先
- テンソル: `linspace`, `rand`, `expand`, `permute`, `chunk`, `split`, `stack`
- NN: `leaky_relu`, `group_norm`, `conv1d`, `conv_transpose2d`, `interpolate`, `pad`
- 学習: momentum SGD, 学習率スケジューラ, 勾配クリッピング
- 損失: `bce_loss`, `l1_loss`, `nll_loss`
- リダクション: `prod`, `cumsum`, `norm`, `topk`
- 線形代数: `inverse`, `dot`
- Image: `resize`, `save`, `normalize`
- 推論: Flash Attention, `temperature_scale`, `repetition_penalty`
