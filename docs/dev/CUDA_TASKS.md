
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

---

## 必要な作業

### 1. 🔴🔴 FFI 関数の autograd 未接続（学習パイプライン不動の根本原因）

**優先度: 最高** | **ファイル**: `crates/tl_cuda/src/ffi_ops.rs`

CUDA の FFI 関数のうち、**autograd 対応済みはわずか 3 演算** のみ。
CPU では **30 演算** が `set_grad_fn` を呼んでおり、CUDA では基本演算の勾配が計算できないため、
**学習パイプラインが一切動作しない**状態。

#### autograd 対応状況（CPU vs CUDA）

| 演算 | CPU | CUDA | 備考 |
|:--|:--:|:--:|:--|
| Add | ✅ | ❌ | `AddBackward` (ops定義済み) |
| Sub | ✅ | ❌ | `SubBackward` (ops定義済み) |
| Mul | ✅ | ❌ | `MulBackward` (ops定義済み) |
| Div | ✅ | ❌ | `DivBackward` (ops定義済み) |
| Matmul | ✅ | ❌ | `MatmulBackward` (ops定義済み) |
| AddScalar | ✅ | ❌ | `AddScalarBackward` (ops定義済み) |
| SubScalar | ✅ | ❌ | `SubScalarBackward` (ops定義済み) |
| MulScalar | ✅ | ❌ | `MulScalarBackward` (ops定義済み) |
| DivScalar | ✅ | ❌ | `DivScalarBackward` (ops定義済み) |
| Pow | ✅ | ❌ | `PowBackward` (ops定義済み) |
| Neg | ✅ | ❌ | `NegBackward` (ops定義済み) |
| Exp | ✅ | ❌ | `ExpBackward` (ops定義済み) |
| Log | ✅ | ❌ | `LogBackward` (ops定義済み) |
| Sqrt | ✅ | ❌ | `SqrtBackward` (ops定義済み) |
| Relu | ✅ | ❌ | `ReluBackward` (ops定義済み) |
| Sigmoid | ✅ | ❌ | `SigmoidBackward` (ops定義済み) |
| Tanh | ✅ | ❌ | `TanhBackward` (ops定義済み) |
| Gelu | ✅ | ❌ | `GeluBackward` (ops定義済み) |
| Silu | ✅ | ❌ | `SiluBackward` (ops定義済み) |
| Sumall | ✅ | ❌ | `SumallBackward` (ops定義済み) |
| SumDim | ✅ | ❌ | `SumDimBackward` (ops定義済み) |
| Mean | ✅ | ❌ | `MeanAllBackward`/`MeanDimBackward` (ops定義済み) |
| Reshape | ✅ | ❌ | `ReshapeBackward` (ops定義済み) |
| Transpose | ✅ | ❌ | `TransposeBackward` (ops定義済み) |
| Softmax | ✅ | ✅ | |
| CrossEntropy | ✅ | ✅ | |
| Embedding | ✅ | ✅ | |
| LayerNorm | ✅ | ❌ | `LayerNormBackward` (ops定義済み) |
| Tril | ✅ | ❌ | ❌ ops自体が未定義 |
| Squeeze | ✅ | ❌ | ❌ ops自体が未定義 |
| Unsqueeze | ✅ | ❌ | ❌ ops自体が未定義 |
| Slice | ✅ | ❌ | ❌ ops自体が未定義 |

**修正方針**:

1. **ffi_ops.rs**: CPU の `ffi.rs` を参考に、各FFI関数に `requires_grad()` チェックと `set_grad_fn` 呼び出しを追加
2. **autograd/ops.rs**: 不足している4つの backward ops (`TrilBackward`, `SqueezeBackward`, `UnsqueezeBackward`, `SliceBackward`) を追加
3. 既存の `autograd/ops.rs` に定義済みの backward 構造体は **そのまま利用可能**（27個は定義済みだが ffi_ops から呼ばれていないだけ）

> [!IMPORTANT]
> autograd ops の構造体は 35 個定義済みだが、FFI 関数でそれらを接続する `set_grad_fn` 呼び出しが **3個しかない**。
> 定義と接続の乖離が根本原因。

---

### 2. 🔴 backward() 中間ノードの grad 未解放（メモリリーク）

**優先度: 高** | **ファイル**: `crates/tl_cuda/src/tensor.rs` L545-550

CPU 版で修正済みの問題が CUDA にも存在する。
backward() の計算グラフ解放ステップで `grad_fn = None` はしているが、**中間ノードの `grad` をクリアしていない**。

```rust
// 現在のコード (L545-550)
for entry in visited.iter_mut() {
    let tensor = unsafe { &mut *entry.0 };
    if let Some(ref mut meta) = tensor.autograd {
        meta.grad_fn = None;
        // ← ここで中間ノードの grad もクリアすべき
    }
}
```

**修正方針**: CPU 版 (`crates/tl_cpu/src/tensor.rs` L429-444) と同様に、中間ノード（`grad_fn` を持つノード）の `grad` も `None` にする。リーフテンソルの grad は `.grad()` で取得するため保持。

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
