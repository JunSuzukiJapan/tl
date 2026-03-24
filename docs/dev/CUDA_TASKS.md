
# CUDA バックエンド対応タスク一覧

`examples/tasks/` 以下の全タスクを CUDA (`--device cuda`) で動作させるために必要な作業をまとめる。

---

## 現状サマリー

| 項目 | 状態 |
|:--|:--|
| IDevice トレイト実装 | ✅ 全メソッド実装済み |
| Autograd (backward/grad/detach) | ✅ 実装済み (35 backward ops) |
| conv2d / batch_norm / layer_norm | ✅ 実装済み |
| 基本演算 (add/sub/mul/div/matmul) | ✅ 実装済み |
| save/load (バイナリ形式) | ✅ 実装済み |
| safetensors (TensorMap) | ✅ tl_runtime 層で実装（デバイス非依存） |
| Arc ベース統一所有権 (V5.0) | ✅ 実装済み |
| FFI autograd 接続 | ✅ 全演算接続済み |
| backward() 中間 grad 解放 | ✅ 修正済み |

---

## 必要な作業

### 1. ✅ FFI 関数の autograd 接続（完了）

**ステータス: 完了** | **ファイル**: `crates/tl_cuda/src/ffi_ops.rs`

全 FFI 関数に autograd が接続済み。`ffi_binary_op!`/`ffi_unary_op!` マクロおよび手動実装により、
全学習対象演算で `set_grad_fn` が呼ばれる状態。

追加で接続した演算:
- `layer_norm` → `LayerNormBackward`
- `tril` → `TrilBackward`
- `slice` → `SliceBackward`

`autograd/ops.rs` に追加した backward 構造体:
- `TrilBackward`, `SliceBackward`, `SqueezeBackward`, `UnsqueezeBackward`

---

### 2. ✅ backward() 中間ノードの grad 解放（完了）

**ステータス: 完了** | **ファイル**: `crates/tl_cuda/src/tensor.rs`

CPU 版と同様に、backward() の計算グラフ解放ステップで中間ノード（`grad_fn` を持つノード）の `grad` もクリアするよう修正済み。
リーフテンソルの grad は `.grad()` で取得するため保持。

---

### 3. 🟡 backward() のアルゴリズム差異確認

**優先度: 中** | **ファイル**: `crates/tl_cuda/src/tensor.rs` L496-551

CUDA の backward() は **worklist（BFS 風）** で実装されている一方、CPU は **トポロジカルソート（DFS）** を使用。

| 項目 | CPU | CUDA |
|:--|:--|:--|
| 走査方式 | DFS → 逆トポロジカル順 | worklist (BFS 風) |
| grad 蓄積 | 全パス蓄積→伝播 | 訪問ごとに即蓄積 |
| visited管理 | HashSet\<usize\> | Vec\<(ptr, Option\<TensorRef\>)\> |

BFS 風では、同一ノードに複数パスから勾配が到達する場合、**蓄積が不完全になるリスク**がある。
複雑な計算グラフ（分岐+合流）を持つタスクで正しく動くか要検証。

**検証方法**: 既存の `examples/tasks/` を `--device cuda` で実行し、CPU との出力を比較。

---

### 4. 🟡 shallow_clone の requires_grad 保持差異

**優先度: 中** | **ファイル**: `crates/tl_cuda/src/tensor.rs` L406-424

CUDA の `shallow_clone()` は `requires_grad` がtrueなら `AutogradMeta` を保持する。
CPU の `shallow_clone()` は常に `autograd: None` を返す。

`detach()` は両方とも `shallow_clone()` を呼ぶため、CUDA 版の detach は grad を計算可能な状態で返す。
学習ループの `board = board.detach(false); board.enable_grad();` パターンで問題ないか確認が必要。

---

### 5. 🟡 CUDA 環境での全タスク動作テスト

**優先度: 中**

以下の全タスクを `--device cuda` で実行し、学習+推論が正常に完了することを検証。

#### 推論のみのタスク
| タスク | TL ファイル | 備考 |
|:--|:--|:--|
| logic | `examples/tasks/logic/*.tl` | 推論のみ |
| tensor_logic | `examples/tasks/tensor_logic/**/*.tl` | 推論のみ |

#### 学習+推論パイプライン
| タスク | 学習 | 推論 | CPU所要時間 |
|:--|:--|:--|:--|
| reverse | `reverse_train.tl` | `reverse_infer.tl` | ~195s |
| addition | `train_add.tl` | `infer_add.tl` | ~3400s |
| heavy | `train_heavy.tl` | `infer_heavy.tl` | ~3560s |
| paper | `train_paper.tl` | `infer_paper.tl` | ~12s |
| recall | `train_recall.tl` | `infer_recall.tl` | ~5s |
| mnist | `train.tl` | `infer.tl` | ~764s |

---

### 6. 🟢 GPU メモリプール最適化

**優先度: 低** | **ファイル**: `crates/tl_cuda/src/buffer_pool.rs`

現在の CUDA バッファプールは `pool_acquire()` / `pool_release()` で基本的な再利用は実装済み。
Metal の Persistent GPU Pool 戦略 (V4.0) のような RSS 膨張対策が CUDA でも必要か、長時間学習で検証。

---

### 7. 🟢 verify_training_pipeline.py の CUDA 対応

**優先度: 低** | **ファイル**: `scripts/verify_training_pipeline.py`

現在のスクリプトは `--device` オプションを `tl` バイナリに渡す仕組みが未対応の可能性がある。
CUDA テスト用にデバイス指定オプションを追加。

---

## 推奨作業順序

1. **FFI autograd 接続** (タスク1) — **最優先**。これなしでは学習が動かない
2. **backward() 中間 grad 解放** (タスク2) — コード変更のみ、デバッグ不要
3. **CUDA 環境テスト** (タスク5) — 全タスクの `--device cuda` 実行
4. **backward アルゴリズム検証** (タスク3) — タスク5の結果に基づき判断
5. **shallow_clone 差異検証** (タスク4) — タスク5の結果に基づき判断
6. **GPU メモリプール最適化** (タスク6) — 長時間学習で問題が出た場合
7. **検証スクリプト CUDA 対応** (タスク7) — 自動化が必要な場合
