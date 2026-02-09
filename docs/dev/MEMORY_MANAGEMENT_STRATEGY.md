# TensorLogic Memory Management Strategy V4.5 (Hybrid Static Analysis + Strict RefCounting + Scope-Based Tensor)
# TensorLogic メモリ管理戦略 V4.5 (ハイブリッド静的解析 + 厳格な参照カウント + スコープベーステンソル管理)

This document outlines the **V2.1 Memory Management Strategy** implemented in TensorLogic.
This strategy combines **Static Analysis (Arena)** for high-performance tensor operations and **Strict Reference Counting** for complex data structures (Structs/Objects).

本ドキュメントでは、TensorLogicに実装された **V2.1 メモリ管理戦略** について概説します。
この戦略は、高性能なテンソル演算のための **静的解析 (Arena)** と、複雑なデータ構造（構造体/オブジェクト）のための **厳格な参照カウント (Strict RefCounting)** を組み合わせたものです。

---

## [English] Memory Management Strategy V2.1

### 1. Core Philosophy: "Zero-Overhead for Tensors, Safety for Structs"

The system aims to optimize the most common case (Tensor computations) while ensuring safety for the most complex case (Dynamic Structs).

-   **Tensors (Arena)**: Short-lived tensors within functions use **Stack-Like Linear Allocation** (Arena). No malloc/free.
-   **Structs (Heap)**: User-defined structs and potentially long-lived tensors use **Heap Allocation** managed by **Reference Counting**.
-   **Ownership Transfer**: Explicit rules for passing ownership between scopes to prevent leaks and double-frees.

### 2. Compiler Pipeline (Static Analysis)

The memory optimization happens in several passes:

#### Phase 1: Shape Inference & Liveness Analysis
The compiler analyzes the lifespan of every tensor variable.
-   **Liveness**: Tracks "Start" and "End" statement index.
-   **Result**: Determines which tensors can share memory slots.

#### Phase 2: Slot Allocation (Linear Scan)
-   Assigns **Offsets** in the Function Frame for temporary tensors.
-   **Shared Slots**: Variables with non-overlapping lifespans share the same offset.
-   **Result**: Total `FrameSize` is calculated for the function.

### 3. Runtime Mechanisms

#### A. Memory Arena (Function Frames)
Each thread maintains a large Arena.
-   **Function Entry**: `tl_mem_function_enter(slots)` pushes a new frame.
-   **Method Calls**: Even methods (`impl` blocks) must establish a frame to support temporary tensors (fixed in V2.1).
-   **Alloc**: `get_buffer(slot_id)` returns a pointer to the pre-allocated slot.

#### B. Struct & Reference Counting (The "Shallow Unregister" Pattern)
Structs require careful lifecycle management:

1.  **Creation (Deep Clone)**:
    -   When initializing a struct (`Point { x: a, y: b }`), we call `emit_deep_clone` (Reference Acquire) for fields.
    -   Local variable `a` (RC=1) -> Struct field `.x` (RC=2).
    -   Both the local variable and the struct hold references.

2.  **Scope Exit (DecRef)**:
    -   When a scope ends (`exit_scope`), all registered local variables are decremented.
    -   `a` (RC=2 -> 1). The struct's reference survives.

3.  **Return (Shallow Unregister)**:
    -   When returning a struct, we must prevent `exit_scope` from freeing the *container* struct itself.
    -   We call `unregister(struct_ptr)` (Shallow). This removes the struct container from the scope's cleanup list.
    -   **Crucially**, we do **NOT** recurse into fields.
    -   Fields remain effectively "owned" by their original local variables until `exit_scope` decrements them.
    -   Result: Struct Container (Unregistered, Survives) + Fields (Refcount > 0, Survive).

---
#### C. Function Argument Strategy: "Borrowing" (V2.2 Optimization)
To further reduce overhead and ambiguity, specialized handling for function arguments is introduced:

-   **Borrowing**: Arguments passed to a function are considered "Owned by the Caller" (or external scope).
-   **No Registration**: The callee function does **NOT** register arguments (`tl_mem_register_*`) or increment their reference counts upon entry.
-   **Implication**:
    -   Arguments are treated as "valid references" guaranteed to outlive the function call.
    -   **Reassignment**: If an argument variable is reassigned, the old value is NOT decremented (because it wasn't owned). The new value IS registered.
    -   **Return/Struct**: If an argument is returned or stored in a struct, explicit `tl_ptr_acquire` (retain) is required to promote the borrowed reference to an owned one.

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

## [English] V3.0 Optimizations (Implemented)

### 1. Return Value Optimization (RVO / DPS)
Strict "Destination Passing Style" (DPS) has been implemented to eliminate return value overhead.
-   **Mechanism**: The caller acts as the "Owner" of the return value slot. It pre-allocates uninitialized stack memory (or reuses a slot) and passes a pointer (`*dest`) to the callee.
-   **Execution**:
    1.  The callee constructs the result directly into `*dest`.
    2.  `tl_ptr_cleanup` is NOT called on `*dest` within the callee (ownership remains with caller).
    3.  Returning a struct/tensor involves NO `inc_ref` operations.
-   **Benefit**: Zero-copy returns for large structs and tensors.

### 2. Move Semantics / Last-Use Optimization
Variables that are passed to a function or assigned as their "last use" in a scope are "moved".
-   **Mechanism**: 
    1.  **Liveness Analysis (V3.1)**: The compiler identifies the last statement where a variable is used.
    2.  **Codegen**: At the point of last use, the compiler omits the `inc_ref` (retain) operation.
    3.  **Ownership**: Ownership is effectively transferred to the receiving variable or function.
    4.  **No Cleanup**: The original variable is not decremented at end of scope (handled by `CLEANUP_NONE` flag or skipped), preventing double-free.
-   **Benefit**: Eliminates redundant `inc_ref`/`dec_ref` pairs, critical for performance in deep call chains.

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

## [English] Zero-Sized Types (ZST) Strategy (V3.2)

### 1. The Problem
Zero-Sized Types (ZSTs) like `struct Empty {}` or `PhantomData<T>` occupy 0 bytes.
However, `malloc(0)` is behaviorally undefined (can return NULL or a unique pointer) and inconsistent across platforms.
Historically, the compiler optimized ZSTs by returning an Aggregate Value instead of a Pointer. This violated the "Struct = Managed Pointer" invariant, causing the runtime to treat the aggregate value as an invalid address (`0x7` etc.) and crash during `free()`.

### 2. The Solution: "ZST = NULL"
To handle ZSTs safely without runtime overhead:
-   **Compiler**: When compiling a struct initialization (`compile_struct_init`), if the struct has no fields (empty), it **returns a NULL pointer** (`i8* null`) constant. `malloc` is skipped entirely.
-   **Runtime**: The `MemoryManager` (`inc_ref`, `dec_ref`, `register`, `release`) detects `ptr == NULL` and **returns immediately (No-Op)**.
    -   Reference counts for ZSTs are effectively non-existent (Logically 0 or Infinite).
    -   Double-free is impossible (freeing NULL is safe).
    -   Key collision in RefCount map is avoided because we early-return before map access.

### 3. Why NULL? (Rationale)
-   **Performance**: Zero allocation cost. Zero tracking cost.
-   **Safety**: Valid pointers are never NULL. NULL is universally recognized as "No Object".
-   **Simplicity**: No need to change the runtime architecture to support "Global ZST Singletons" or "1-byte allocations".

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

## [English] Autograd Memory Leak Fix (V3.3)

### 1. The Problem
In autograd-heavy workloads (e.g., N-Queens gradient descent), memory usage grew linearly with each training iteration:
- **Symptom**: 1237 MB → 2168 MB over a single run (+930 MB leak)
- **Root Cause**: `emit_retain()` for Tensor types was calling `tl_tensor_acquire()` excessively (31 times vs 3 times in v0.2.1)

### 2. Analysis (v0.2.1 vs rebuild-from-scratch)
Comparison of LLVM IR revealed the key difference:

| Function Call | v0.2.1 | Current (before fix) |
|---------------|--------|----------------------|
| `tl_tensor_acquire` | **3** | **31** (10x increase) |

The `emit_retain()` function was added after v0.2.1 to fix UAF (Use-After-Free) bugs in struct returns. However, applying it to Tensors caused **over-retention**: tensors held extra references that prevented the computational graph from being freed.

### 3. The Solution
1. **Disable Tensor Acquire in `emit_retain()`**: Tensor types no longer call `tl_tensor_acquire()`. Their lifecycle is already managed by the runtime's memory manager.
2. **Automatic `tl_clear_grads()` in Loops**: The compiler inserts `tl_clear_grads()` at the end of each `for` loop iteration (in the `for_latch` block) to clear gradient storage automatically.

### 4. Results
| Metric | Before | After |
|--------|--------|-------|
| N-Queens Memory | 1237→2168 MB (+930 MB) | **39 MB stable (Zero Leak)** |
| Test Pass Rate | 83.4% | **Maintained** |

### 5. Key Insight
**Tensors have different ownership semantics than Structs.** While structs benefit from `emit_retain()` to prevent UAF, tensors are already managed by the runtime's centralized memory manager (`TensorPool`, `MemoryManager`). Applying struct-like retain logic to tensors creates duplicate reference tracking that interferes with Candle's internal computational graph cleanup.

---

## [Japanese] Autograd メモリリーク修正 (V3.3)

### 1. 問題点
Autograd を多用するワークロード（例: N-Queens の勾配降下法）において、訓練イテレーションごとにメモリ使用量が線形に増加していました：
- **症状**: 1回の実行で 1237 MB → 2168 MB（+930 MB リーク）
- **根本原因**: テンソル型に対する `emit_retain()` が `tl_tensor_acquire()` を過剰に呼び出していた（v0.2.1 の 3回 に対し 31回）

### 2. 分析 (v0.2.1 vs rebuild-from-scratch)
LLVM IR の比較により、決定的な差異が判明しました：

| 関数呼び出し | v0.2.1 | 修正前 (現在) |
|--------------|--------|--------------|
| `tl_tensor_acquire` | **3** | **31** (10倍増) |

`emit_retain()` 関数は、構造体の返却時における UAF (Use-After-Free) バグを修正するために v0.2.1 以降に追加されました。しかし、これをテンソルにも適用したことで **過剰な参照保持** が発生し、計算グラフの解放が妨げられていました。

### 3. 解決策
1. **`emit_retain()` でのテンソル Acquire を無効化**: テンソル型は `tl_tensor_acquire()` を呼び出さなくなりました。ライフサイクルはランタイムのメモリマネージャーで既に管理されています。
2. **ループ内での自動 `tl_clear_grads()`**: コンパイラは各 `for` ループイテレーションの終了時（`for_latch` ブロック）に `tl_clear_grads()` を自動挿入し、勾配ストレージを自動的にクリアします。

### 4. 結果
| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| N-Queens メモリ | 1237→2168 MB (+930 MB) | **39 MB 安定（リークゼロ）** |
| テスト成功率 | 83.4% | **維持** |

### 5. 重要な知見
**テンソルと構造体では所有権セマンティクスが異なります。** 構造体は UAF を防ぐために `emit_retain()` の恩恵を受けますが、テンソルはランタイムの集約されたメモリマネージャー（`TensorPool`、`MemoryManager`）によって既に管理されています。構造体と同様の retain ロジックをテンソルに適用すると、重複した参照追跡が発生し、Candle 内部の計算グラフのクリーンアップを妨害します。

---

## [English] Persistent GPU Pool Strategy (V4.0)

### 1. Problem: Metal RSS Growth
On Metal devices, repeatedly allocating and freeing GPU memory causes **Resident Set Size (RSS)** to grow continuously, even when the TL runtime's internal buffer pool remains stable. This is a known Metal driver behavior where released memory is not always returned to the OS.

### 2. Solution: Never Release GPU Memory
The V4.0 strategy implements a **Persistent GPU Pool** that never deallocates GPU memory back to the OS:

- **`acquire()`**: Currently disabled (Phase 1). Always returns `None`, forcing new allocations.
- **`release()`**: Drops tensor **contents** (internal Candle tensors, Arcs) but intentionally **leaks** the `OpaqueTensor` struct memory.

This approach:
1. **Prevents RSS growth** from repeated alloc/free cycles
2. **Avoids Metal driver issues** with memory reclamation
3. **Memory is reclaimed by OS** when the process exits

### 3. Statistics API
New C-ABI functions for monitoring:
- `tl_get_gpu_total_allocated_bytes()`: Total bytes ever allocated
- `tl_get_gpu_free_count()`: Number of "freed" (leaked) tensors
- `tl_get_gpu_free_bytes()`: Bytes in freed tensors
- `tl_get_gpu_pool_hit_rate()`: Pool hit rate (currently 0%)
- `tl_dump_gpu_pool_stats()`: Dump all stats to stderr

### 4. Environment Variable
- `TL_GPU_PREALLOCATE_MB=<size>`: Pre-allocate GPU memory at startup (future implementation)

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

## [English] Scope-Based Tensor Management (V4.5)

### 1. Problem: `Type::Struct("Tensor")` Misidentification
The parser resolves `-> Tensor` return type annotations as `Type::Struct("Tensor", [])` rather than `Type::Tensor(_, _)`. Since `Tensor` is an **opaque pointer** in LLVM IR (not a struct with accessible fields), this causes failures whenever the codegen treats it as a real struct:

- **SRET (Struct Return)**: Functions returning Tensor incorrectly used a hidden struct-return pointer
- **GEP (GetElementPtr)**: `build_struct_gep` on an opaque pointer → `"GEP pointee is not a struct"` error
- **Deep Clone**: `emit_deep_clone` attempted struct field traversal on Tensor pointers
- **ABI Adapter**: Post-call wrapper tried to pack Tensor pointer into a struct via GEP

### 2. Solution: Type Normalization at Codegen Boundaries
Instead of modifying the parser, we normalize `Struct("Tensor")` → `Type::Tensor(F32, 0)` at every codegen boundary where type-specific dispatch occurs:

| Location | File | Fix |
|:---|:---|:---|
| `uses_sret` (prototype) | `mod.rs` | Exclude `Struct("Tensor")` from SRET |
| `uses_sret` (body) | `mod.rs` | Same |
| `uses_sret` (static call) | `expr.rs` | Same |
| `uses_sret` (method call) | `expr.rs` | Same |
| `uses_sret` (fn call DPS) | `expr.rs` | Same |
| `emit_cleanup_vars_in_scope` | `mod.rs` | Pass `Type::Tensor` to `emit_recursive_free` |
| `emit_deep_clone` | `stmt.rs` | Redirect to `Type::Tensor` branch |
| Post-call ABI adapter | `expr.rs` | Replace struct wrapping with type normalization |

### 3. Tensor Lifecycle in Scopes

```
┌─ make_tensor() ──────────────────────────────┐
│  let t = Tensor::ones([2, 3])                │
│    → tl_tensor_ones_i64() returns ptr        │
│    → t registered in current scope           │
│                                              │
│  return t                                    │
│    → tl_tensor_promote(t) (unregister+float) │
│    → emit_all_scopes_cleanup (skips t)       │
│    → return ptr                              │
└──────────────────────────────────────────────┘
         │ ptr (floating)
         ▼
┌─ main() ─────────────────────────────────────┐
│  let t = make_tensor()                       │
│    → call returns ptr                        │
│    → tl_tensor_register(t) (caller-side)     │
│    → store to alloca                         │
│                                              │
│  // ... use t ...                            │
│                                              │
│  scope exit:                                 │
│    → emit_recursive_free(t, Type::Tensor)    │
│    → tl_tensor_release_safe(t)               │
└──────────────────────────────────────────────┘
```

### 4. Key Invariant
**`Struct("Tensor")` must never reach `build_struct_gep` or `emit_struct_copy`.** These functions require a concrete LLVM struct type with accessible fields, but Tensor is an opaque `ptr` managed exclusively by runtime functions (`tl_tensor_acquire`, `tl_tensor_release_safe`, `tl_tensor_promote`).

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

## [English] V5.0 Hybrid Cleanup Strategy (Autograd Graph Management)

### 1. The Challenge of Autograd Graphs
Autograd computational graphs introduce unique memory management challenges:
- **Cyclic/Long-lived References**: Tensors within the graph hold raw pointers (`*mut CpuTensor`) to their parents to enable backpropagation.
- **Reference Cycles**: The graph structure often outlives the local scope of the variables that created it.
- **Dangling Pointers**: Immediate scope-based deallocation (`release_safe`) causes use-after-free errors when the autograd engine traverses the graph later.

### 2. The Solution: Hybrid Lifecycle Management
We implemented a **Hybrid Strategy** that treats tensors differently based on their participation in Autograd:

| Tensor Type | Lifecycle Manager | Cleanup Trigger | Mechanism |
|:---|:---|:---|:---|
| **Pure Data** (No Autograd) | **Compiler Scope** | Scope Exit | `tl_tensor_release_safe` calls `clear_data` immediately. |
| **Autograd Node** (Requires Grad) | **Graph Engine** | `backward()` / `detach()` | `release_safe` is **No-Op**. Cleanup assumes graph traversal. |

### 3. Implementation Details

#### A. Conditional `release_safe`
The runtime's `tl_tensor_release_safe` checks if a tensor has an active Autograd component.
- **If No Autograd**: It calls `tl_cpu_tensor_clear_data`, freeing the underlying `Vec<f32>` but keeping the struct (safe against double-free).
- **If Autograd**: It does nothing. The tensor remains alive for the graph.

#### B. Event-Driven Graph Cleanup
Since autograd tensors are ignored by scope cleanup, they must be explicitly cleaned up by graph events:

1.  **Backward Pass Cleanup**:
    -   At the end of `backward()`, the engine traverses the topological order.
    -   **Non-Leaf Nodes** (intermediate calculation results) are identified.
    -   Their data buffers (`data_f32`, `shape`) and autograd metadata are cleared immediately.
    -   **Result**: Memory spikes during training are suppressed as intermediate tensors die immediately after use.

2.  **Detach Cleanup**:
    -   When `t.detach()` is called (typically at epoch boundaries), the **source tensor's** entire upstream graph is traversed recursively.
    -   All nodes in the detached graph are effectively "garbage collected".

### 4. Results
This strategy reduced memory growth in the N-Queens solver from **4MB/100 epochs** to **0.33MB/100 epochs** (12x improvement), with zero segmentation faults.

---

## [Japanese] V5.0 ハイブリッドクリーンアップ戦略 (Autograd グラフ管理)

### 1. Autograd グラフの課題
自動微分（Autograd）の計算グラフは、独特なメモリ管理の課題をもたらします：
- **長寿命な参照**: グラフ内のテンソルは、逆伝播を可能にするために親テンソルへの生ポインタ (`*mut CpuTensor`) を保持します。
- **参照サイクル**: グラフ構造は、それを作成した変数のローカルスコープよりも長く生き続けることがよくあります。
- **ダングリングポインタ**: スコープベースの即時解放 (`release_safe`) を行うと、後で Autograd エンジンがグラフを走査する際に `use-after-free` エラー（不正参照）が発生します。

### 2. 解決策: ハイブリッド・ライフサイクル管理
テンソルが Autograd に参加しているかどうかに応じて扱いを変える **ハイブリッド戦略** を実装しました：

| テンソル種別 | 管理主体 | 解放トリガー | メカニズム |
|:---|:---|:---|:---|
| **純粋データ** (Autograd なし) | **コンパイラスコープ** | スコープ脱出 | `tl_tensor_release_safe` が即座に `clear_data` を呼ぶ。 |
| **Autograd ノード** (Grad あり) | **グラフエンジン** | `backward()` / `detach()` | `release_safe` は **No-Op (何もしない)**。クリーンアップはグラフ走査に委ねる。 |

### 3. 実装詳細

#### A. 条件付き `release_safe`
ランタイムの `tl_tensor_release_safe` は、テンソルがアクティブな Autograd コンポーネントを持っているか確認します。
- **Autograd なし**: `tl_cpu_tensor_clear_data` を呼び出し、下層の `Vec<f32>` を解放しますが、構造体自体は残します（二重解放に対して安全）。
- **Autograd あり**: 何もしません。テンソルはグラフのために生き続けます。

#### B. イベント駆動型グラフクリーンアップ
Autograd テンソルはスコープ解放で無視されるため、グラフイベントによって明示的にクリーンアップされる必要があります：

1.  **Backward パス・クリーンアップ**:
    -   `backward()` の終了時に、エンジンはトポロジカル順序を走査します。
    -   **非リーフノード**（中間計算結果）を特定します。
    -   それらのデータバッファ（`data_f32`, `shape`）と autograd メタデータは即座にクリアされます。
    -   **結果**: 学習中の中間テンソルが使用直後に死ぬため、メモリスパイクが抑制されます。

2.  **Detach クリーンアップ**:
    -   `t.detach()` が呼ばれると（通常はエポックの境界で）、**ソーステンソル** の上流グラフ全体が再帰的に走査されます。
    -   切り離されたグラフ内の全ノードは実質的に「ガベージコレクション」されます。

### 4. 結果
この戦略により、N-Queens ソルバーにおけるメモリ増加は **100エポックあたり 4MB** から **0.33MB** に減少（12倍の改善）し、セグメンテーションフォールトも完全に解消されました。
