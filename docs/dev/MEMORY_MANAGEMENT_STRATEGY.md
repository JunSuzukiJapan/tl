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

### 4. 比較 (V1 vs V2.1)

| 機能 | V1 (旧) | V2.1 (現在) |
| :--- | :--- | :--- |
| **テンソル割り当て** | 個別 `malloc` | **Arena ポインタ加算** (高速) |
| **構造体管理** | 不完全な所有権管理 | **Strict RefCount + Shallow Unregister** |
| **メソッド実行** | フレームなし (不安定) | **関数同様のフレーム確保** (安定) |
| **スロット再利用** | なし | **あり (Liveness Analysis)** |
