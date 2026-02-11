
# TensorLogic メモリ管理戦略 V4.5 (ハイブリッド静的解析 + 厳格な参照カウント + スコープベーステンソル管理)

本ドキュメントでは、TensorLogicに実装された **V2.1 メモリ管理戦略** について概説します。
この戦略は、高性能なテンソル演算のための **静的解析 (Arena)** と、複雑なデータ構造（構造体/オブジェクト）のための **厳格な参照カウント (Strict RefCounting)** を組み合わせたものです。

---

## [Japanese] メモリ管理戦略 V2.1 (ハイブリッド静的解析)

### 1. 中心哲学: "テンソルはゼロコスト、構造体は安全性"

V2.1システムは、最も頻繁な計算（テンソル演算）を最適化しつつ、複雑なデータ構造（構造体）の安全性を担保することを目標としています。

-   **テンソル (Arena)**: 関数内の一時的なテンソルは **スタックライクなリニア割り当て (Arena)** を使用します。malloc/freeは発生しません。
-   **構造体 (Heap)**: ユーザー定義の構造体や長寿命なテンソルは、**参照カウント (Reference Counting)** で管理される **ヒープ割り当て** を使用します。
-   **所有権の移動**: スコープ間での所有権移動に関して厳格なルールを設け、メモリリークと二重解放を防ぎます。

### 2. コンパイラパイプライン

#### Phase 1: 形状推論と生存区間解析
コンパイラは全テンソル変数の生存期間を解析します。
-   **生存区間**: 変数の「定義」から「最後の使用」までを追跡。
-   **結果**: どのテンソルがメモリ領域を共有できるかを決定します。

#### Phase 2: スロット割り当て
-   **オフセット割り当て**: 一時テンソル用に関数フレーム内のオフセットを決定します。
-   **スロット共有**: 生存期間が重ならない変数は、同じオフセット（スロット）を再利用します。
-   **結果**: 関数ごとに必要な `TotalFrameSize`（スロット数）が決定されます。

### 3. ランタイムメカニズムとバグ修正 (V2.1)

#### A. メモリアリーナ (関数フレーム)
各スレッドは巨大なアリーナを持ちます。
-   **関数エントリ**: `tl_mem_function_enter(slots)` を呼び出し、フレームを確保します。
-   **メソッド対応**: メソッド（`impl` 内関数）であっても、内部で一時テンソルを使う場合は必ずフレーム確保が必要です（V2.1で修正された点）。これが無いと `No function frame` エラーが発生します。

#### B. 構造体と参照カウント ("Shallow Unregister" パターン)
構造体の返却時におけるメモリ破壊（二重解放）を防ぐため、以下の戦略を採用しています。

1.  **生成時 (Deep Clone)**:
    -   構造体初期化時 (`Point { x: a }`)、フィールドに対して `emit_deep_clone` (実質的な `ptr_acquire`/参照カウント増分) を行います。
    -   ローカル変数 `a` (RC=1) -> 構造体フィールド `.x` (RC=2)。
    -   ローカル変数と構造体の双方が参照を保持します。

2.  **スコープ脱出時 (DecRef)**:
    -   スコープ終了時 (`exit_scope`)、登録された全てのローカル変数の参照カウントがデクリメントされます。
    -   `a` (RC: 2 -> 1)。構造体側の参照は生き残ります。

3.  **返却時 (Shallow Unregister)**:
    -   構造体を関数から返す際、`exit_scope` が構造体自身（コンテナ）を解放しないように `unregister(struct_ptr)` を呼び出します。
    -   **重要**: この際、フィールドに対して再帰的な unregister は **行いません**。
    -   フィールドは「ローカル変数の参照」が `exit_scope` でデクリメントされることで、正しく所有権が構造体側（だけ）に残る状態（RC=1）に遷移します。
    -   結果: 構造体コンテナ（Unregister済み、生存） + フィールド（RC>0、生存）となり、安全に呼び出し元へ返却されます。

---

#### C. 関数引数戦略: "Borrowing" (V2.2 最適化)
さらなるオーバーヘッド削減と責任分界の明確化のため、関数引数に対して特別な扱いを導入します。

-   **借用 (Borrowing)**: 関数に渡された引数は、「呼び出し元（または外部スコープ）が所有している」とみなします。
-   **登録なし (No Registration)**: 呼び出された関数（被呼者）は、引数に対して `tl_mem_register_*` を**呼び出しません**。参照カウントのインクリメントも行いません。
-   **影響とルール**:
    -   引数は、関数呼び出し期間中は常に有効であることが保証された「参照」として扱われます。
    -   **再代入**: 引数変数に新しい値を代入する場合、古い値（引数として渡された値）の参照カウントはデクリメント**しません**（所有していないため）。新しい値は通常通り登録（所有）します。
    -   **返却/構造体への格納**: 引数を関数から返す場合、または構造体のフィールドにセットして永続化する場合は、明示的な `tl_ptr_acquire` (retain) を呼び出し、借用された参照を所有された参照へと昇格させる必要があります。

---

### 4. 比較 (V1 vs V2.1)

| 機能 | V1 (旧) | V2.1 (現在) |
| :--- | :--- | :--- |
| **テンソル割り当て** | 個別 `malloc` | **Arena ポインタ加算** (高速) |
| **構造体管理** | 不完全な所有権管理 | **Strict RefCount + Shallow Unregister** |
| **メソッド実行** | フレームなし (不安定) | **関数同様のフレーム確保** (安定) |
| **スロット再利用** | なし | **あり (Liveness Analysis)** |

---

## [Japanese] V3.0 最適化 (実装済み)

### 1. 戻り値の最適化 (RVO / DPS)
戻り値のオーバーヘッドを排除するために、厳格な「Destination Passing Style (DPS)」が実装されました。
-   **メカニズム**: 呼び出し元が戻り値スロットの「所有者」となります。未初期化のスタックメモリ（またはスロット）を事前に確保し、そのポインタ（`*dest`）を呼び出し先（Callee）に渡します。
-   **実行**:
    1.  呼び出し先は、結果を直接 `*dest` に構築します。
    2.  呼び出し先内部では、`*dest` に対して `tl_ptr_cleanup` を呼び出しません（所有権は呼び出し元にあるため）。
    3.  構造体やテンソルを返す際に、`inc_ref` 操作は一切発生しません。
-   **メリット**: 大きな構造体やテンソルのゼロコピー返却を実現。

### 2. ムーブセマンティクス / ラストユース最適化 (Move Semantics)
関数に渡されたり、代入されたりする際、それが「最後の使用（Last Use）」である変数は「移動（Move）」されます。
-   **メカニズム**:
    1.  **生存区間解析 (V3.1)**: コンパイラは変数が最後に使用されるステートメントを特定します。
    2.  **コード生成**: 最後の使用時点では、`inc_ref` (retain) 操作を省略します。
    3.  **所有権**: 所有権は受信側の変数や関数に効率的に転送されます。
    4.  **クリーンアップなし**: 元の変数はスコープ終了時にデクリメントされません（`CLEANUP_NONE` フラグ等で管理）、二重解放を防ぎます。
-   **メリット**: `inc_ref` / `dec_ref` のペアを削除し、深い呼び出しチェーンにおけるパフォーマンスを劇的に向上させます。

---

## [Japanese] ZST (ゼロサイズ型) 戦略 (V3.2)

### 1. 問題点
`struct Empty {}` や `PhantomData<T>` などのゼロサイズ型（ZST）は0バイトのメモリを占有します。
しかし、`malloc(0)` の挙動は未定義（NULLを返すか、ユニークなポインタを返すか）であり、プラットフォームによって異なります。
以前のコンパイラは、ZSTをポインタではなく「Aggregate値」として返す最適化を行っていました。これは「構造体＝管理ポインタ」という不変条件に違反し、ランタイムが値を無効なアドレス（`0x7`など）として解釈し、`free()` でクラッシュさせる原因となっていました。

### 2. 解決策: "ZST = NULL"
ZSTを安全かつ低コストに扱うため、以下の戦略を採用しました：
-   **コンパイラ**: 構造体初期化 (`compile_struct_init`) において、フィールドがない（空の）構造体の場合、**NULLポインタ** (`i8* null`) 定数を返します。`malloc` は完全にスキップされます。
-   **ランタイム**: `MemoryManager` の各関数 (`inc_ref`, `dec_ref`, `register`, `release`) は、`ptr == NULL` を検知すると **即座にリターン（何もしない）** します。
    -   ZSTの参照カウントは実質的に存在しません。
    -   NULLの解放は安全であるため、二重解放は発生しません。
    -   マップアクセス前にリターンするため、参照カウントマップでのキー衝突も回避されます。

### 3. なぜNULLか？ (Rationale)
-   **パフォーマンス**: 割り当てコストも追跡コストもゼロです。
-   **安全性**: 有効なヒープポインタがNULLになることはありません。
-   **単純性**: 「グローバルZSTシングルトン」や「1バイト割り当て」のような複雑な仕組みをランタイムに導入する必要がありません。

---

## [Japanese] Autograd メモリリーク修正 (V3.3) — ※ V5.0 で置換済み

> [!NOTE]
> V3.3 の `emit_retain` 制限と `tl_clear_grads` 自動挿入は、V5.0 の Arc ベース統一所有権モデルにより不要となりました。
> 以下は歴史的な経緯として残しています。

### 1. 問題点
Autograd を多用するワークロード（例: N-Queens の勾配降下法）において、訓練イテレーションごとにメモリ使用量が線形に増加していました。

### 2. 解決策
1. **`emit_retain()` でのテンソル Acquire を無効化**
2. **ループ内での自動 `tl_clear_grads()`**

### 3. V5.0 での置換
Arc ベースの所有権統一により、テンソルも構造体と同様に参照カウントで管理されるようになりました。詳細は V5.0 セクションを参照。

---

## [Japanese] Persistent GPU Pool 戦略 (V4.0)

### 1. 問題: Metal の RSS 膨張
Metal デバイスでは、GPU メモリの確保と解放を繰り返すと、TL ランタイム内部のバッファプールが安定していても **Resident Set Size (RSS)** が継続的に増加します。これは Metal ドライバの既知の挙動で、解放されたメモリが常に OS に返されるわけではありません。

### 2. 解決策: GPU メモリを解放しない
V4.0 戦略は、GPU メモリを OS に解放しない **Persistent GPU Pool** を実装します：

- **`acquire()`**: 現在無効（Phase 1）。常に `None` を返し、新規確保を強制。
- **`release()`**: テンソルの **コンテンツ**（内部 Candle テンソル、Arc）はドロップしますが、`OpaqueTensor` 構造体メモリは意図的に **リーク** します。

このアプローチにより：
1. 繰り返しの確保/解放サイクルによる **RSS 膨張を防止**
2. メモリ回収に関する **Metal ドライバの問題を回避**
3. プロセス終了時に **OS がメモリを回収**

### 3. 統計 API
監視用の新しい C-ABI 関数：
- `tl_get_gpu_total_allocated_bytes()`: 累計確保バイト数
- `tl_get_gpu_free_count()`: 「解放」（リーク）されたテンソル数
- `tl_get_gpu_free_bytes()`: 解放されたテンソルのバイト数
- `tl_get_gpu_pool_hit_rate()`: プール命中率（現在 0%）
- `tl_dump_gpu_pool_stats()`: 全統計を stderr にダンプ

### 4. 環境変数
- `TL_GPU_PREALLOCATE_MB=<size>`: 起動時に GPU メモリを事前確保（将来実装）

---

## [Japanese] スコープ内テンソル管理 (V4.5)

### 1. 問題: `Type::Struct("Tensor")` の誤認識
パーサーが `-> Tensor` 型注釈を `Type::Struct("Tensor", [])` として解析する。しかし Tensor は LLVM IR 上では**不透明ポインタ** (`ptr`) であり、フィールドアクセス可能な構造体ではない。codegen が Tensor を構造体として扱おうとすると、以下のエラーが発生する：

- **SRET（構造体返し）**: Tensor を返す関数が隠しポインタ引数を誤って使用
- **GEP**: 不透明ポインタに `build_struct_gep` → `"GEP pointee is not a struct"` エラー
- **Deep Clone**: `emit_deep_clone` が Tensor ポインタの構造体フィールド走査を試行
- **ABI アダプタ**: 関数呼び出し後に Tensor ポインタを GEP で構造体にパックしようとして失敗

### 2. 解決策: codegen 境界での型正規化
パーサーを変更する代わりに、型に基づくディスパッチが行われるすべての codegen 境界で `Struct("Tensor")` → `Type::Tensor(F32, 0)` に正規化する：

| 箇所 | ファイル | 修正 |
|:---|:---|:---|
| `uses_sret`（プロトタイプ） | `mod.rs` | `Struct("Tensor")` を SRET から除外 |
| `uses_sret`（関数本体） | `mod.rs` | 同上 |
| `uses_sret`（静的メソッド呼び出し） | `expr.rs` | 同上 |
| `uses_sret`（メソッド呼び出し） | `expr.rs` | 同上 |
| `uses_sret`（関数呼び出し DPS） | `expr.rs` | 同上 |
| `emit_cleanup_vars_in_scope` | `mod.rs` | `Type::Tensor` で `emit_recursive_free` を呼出 |
| `emit_deep_clone` | `stmt.rs` | `Type::Tensor` ブランチにリダイレクト |
| 呼び出し後 ABI アダプタ | `expr.rs` | 構造体ラッピングを型正規化に置換 |

### 3. スコープ内テンソルのライフサイクル

```
┌─ make_tensor() ──────────────────────────────┐
│  let t = Tensor::ones([2, 3])                │
│    → tl_tensor_ones_i64() がポインタを返す    │
│    → t を現在のスコープに登録                  │
│                                              │
│  return t                                    │
│    → tl_tensor_promote(t) (登録解除+浮遊化)   │
│    → emit_all_scopes_cleanup (t をスキップ)   │
│    → ポインタを返す                           │
└──────────────────────────────────────────────┘
         │ ptr (浮遊状態)
         ▼
┌─ main() ─────────────────────────────────────┐
│  let t = make_tensor()                       │
│    → 呼び出しがポインタを返す                  │
│    → tl_tensor_register(t) (呼び出し元で登録)  │
│    → alloca に格納                            │
│                                              │
│  // ... t を使用 ...                          │
│                                              │
│  スコープ脱出:                                │
│    → emit_recursive_free(t, Type::Tensor)    │
│    → tl_tensor_release_safe(t)               │
└──────────────────────────────────────────────┘
```

### 4. 重要な不変条件
**`Struct("Tensor")` は `build_struct_gep` や `emit_struct_copy` に到達してはならない。** これらの関数はフィールドアクセス可能な具体的な LLVM 構造体型を必要とするが、Tensor はランタイム関数（`tl_tensor_acquire`, `tl_tensor_release_safe`, `tl_tensor_promote`）により排他的に管理される不透明な `ptr` である。

---

## [Japanese] Arc ベース統一所有権 (V5.0)

### 1. 動機
V3.3 および以前の戦略では、Autograd テンソルと通常テンソルで異なるメモリ管理を行っていました：
- **通常テンソル**: `Box::into_raw` で割り当て、`Box::from_raw` で解放
- **Autograd テンソル**: `release_safe` が No-Op、`backward()` 完了時に全ノード走査で手動クリーンアップ
- **GradFn が `*mut CpuTensor` 生ポインタで入力を保持**: 所有権が曖昧で dangling pointer のリスク

これにより、`backward()` や `release_safe` 内での特別な条件分岐が必要で、コードの複雑化およびメモリリークの原因となっていました。

### 2. 解決策: Arc ベース統一

Autograd テンソルも通常テンソルも、**同じ `Arc<UnsafeCell<CpuTensor>>` で管理** します。

```rust
// 型エイリアス
pub type TensorRef = Arc<UnsafeCell<CpuTensor>>;

// FFI 割り当て
fn make_tensor(t: CpuTensor) -> *mut OpaqueTensor {
    let arc = Arc::new(UnsafeCell::new(t));
    Arc::into_raw(arc) as *mut CpuTensor
}

// FFI 解放
fn tl_cpu_tensor_release(t: *mut OpaqueTensor) {
    unsafe { drop(Arc::from_raw(t as *const UnsafeCell<CpuTensor>)); }
}
```

### 3. Autograd グラフの所有権

GradFn の入力参照は `TensorRef` (`Arc::clone`) で保持されます：

```rust
// 以前: 生ポインタ（所有権曖昧）
struct AddBackward { a: *mut CpuTensor, b: *mut CpuTensor, ... }

// V5.0: Arc で共有所有（参照カウントで安全）
struct AddBackward { a: TensorRef, b: TensorRef, ... }
```

`tensor_ref_from_ptr(ptr)` で FFI ポインタから `Arc::clone` (RC+1) を取得し、`set_grad_fn` に渡します。

### 4. backward() の簡素化

V5.0 では `backward()` 完了後のクリーンアップが大幅に簡素化されました：

```diff
 // 以前: 全ノード DFS 走査で手動クリーンアップ
-let mut all_nodes = Vec::new();
-Self::collect_all_graph_nodes(self_ptr, &mut visited, &mut all_nodes);
-for &ptr in &all_nodes { /* 手動 grad_fn クリア */ }

 // V5.0: 出力テンソルの grad_fn をクリアするだけ
+if let Some(ref mut meta) = self.autograd {
+    meta.grad_fn = None;  // → GradFn Drop → TensorRef Drop → 連鎖解放
+}
```

出力テンソルの `grad_fn = None` で GradFn が Drop され、その内部の `TensorRef` が Drop され、中間テンソルの Arc RC が連鎖的に減少し、不要なテンソルが自然に解放されます。

### 5. release_safe の統一

`tl_tensor_release_safe` は Autograd の有無にかかわらず **常に `Arc::from_raw(ptr)` で RC-1** を行います：

| テンソル種別 | release_safe の挙動 | 結果 |
|:---|:---|:---|
| 純粋データ (RC=1) | Arc Drop → CpuTensor 解放 | メモリ即座解放 |
| Autograd ノード (RC>1) | Arc RC-1 | GradFn の TensorRef がまだ保持中、テンソルは生存 |
| Autograd ノード (RC=1) | Arc Drop → autograd含む全解放 | backward() 後の自然解放 |

### 6. 変更対象ファイル

| ファイル | 変更内容 |
|:---|:---|
| `tensor.rs` | `TensorRef` 型定義、`backward()` 簡素化、`collect_all_graph_nodes` 削除 |
| `autograd/mod.rs` | `GradFn::inputs()` → `Vec<TensorRef>` |
| `autograd/ops.rs` | 全 19 Backward 構造体を `TensorRef` に変更 |
| `ffi.rs` | `make_tensor`: Box→Arc、全 19 `set_grad_fn`: `tensor_ref_from_ptr` |
| `memory.rs` | テンソルプール廃止、`release_tensor` で Arc RC-1 |
| `memory_ffi.rs` | `release_safe`: `clear_data` → `release` (Arc drop) |
