# TensorLogic Memory Management Strategy V2.1 (Hybrid Static Analysis + Strict RefCounting)
# TensorLogic メモリ管理戦略 V2.1 (ハイブリッド静的解析 + 厳格な参照カウント)

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

## [English] Future Roadmap (V3.0+ Candidates)

### 1. Return Value Optimization (RVO / NRVO)
Currently, returning a complex object (Struct/Tensor) from a function involves incrementing its reference count, returning the pointer, and then potentially assigning it to a new variable (another increment/decrement cycle).
-   **Optimization**: Implement **Destination Passing Style**. The caller provides a pointer to uninitialized memory (the destination) as a hidden argument. The callee constructs the return value directly into this memory.
-   **Benefit**: Eliminates redundant copy/ref-count operations for return values.

### 2. Move Semantics / Last-Use Optimization
Variables that are passed to a function and never used again in the caller scope can be "moved".
-   **Optimization**: Static analysis identifies the "last use" of a variable. The compiler emits a move operation (passing ownership) instead of a copy (incref). The callee takes ownership and is responsible for cleanup (or further moving).
-   **Benefit**: Removes pairs of `inc_ref`/`dec_ref` operations, significantly reducing overhead for chain function calls.

---

## [Japanese] 将来のロードマップ (V3.0+ 候補)

### 1. 戻り値の最適化 (RVO / NRVO)
現在、関数から複雑なオブジェクト（Struct/Tensor）を返す際、参照カウントを増やしてポインタを返し、呼び出し元で変数に代入する（さらに増減が発生する）というコストがかかっています。
-   **最適化**: **Destination Passing Style**（宛先渡しスタイル）を導入します。呼び出し元が「結果を格納するメモリ領域（ポインタ）」を隠し引数として関数に渡し、関数はその領域に直接データを構築します。
-   **メリット**: 戻り値に関する不要なコピーや参照カウント操作を完全に排除できます。

### 2. ムーブセマンティクス / ラストユース最適化 (Move Semantics)
関数に渡された後、呼び出し元で二度と使用されない変数は「移動（Move）」させることができます。
-   **最適化**: 静的解析により変数の「最後の使用（Last Use）」を特定します。コンパイラはコピー（inc_ref）の代わりにムーブ操作（所有権の移動）を行うコードを生成します。受け取った関数側が所有権を持ち、解放（またはさらなる移動）の責任を負います。
-   **メリット**: `inc_ref` と `dec_ref` のペアを削除でき、特に関数チェーン呼び出し時のオーバーヘッドを劇的に削減できます。
