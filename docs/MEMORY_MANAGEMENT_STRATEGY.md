# TensorLogic Memory Management Strategy
# TensorLogic メモリ管理戦略

This document outlines the memory management strategy used in the TensorLogic (tl) runtime and compiler.
本ドキュメントでは、TensorLogic (tl) ランタイムおよびコンパイラで使用されるメモリ管理戦略について概説します。

---

## [English] Memory Management Strategy

### 1. Core Philosophy: Hybrid Reference Counting & Scope-Based Management

TensorLogic uses a **Hybrid Memory Management** system that combines **Reference Counting** with **Scope-Based Automatic Deallocation**.

- **Scopes**: Manage the lifecycle of temporary tensors (intermediate calculation results). When a scope exits, all tensors registered in that scope have their reference count decremented.
- **Reference Counting**: Manages the precise lifecycle of tensors. A tensor is only freed when its reference count reaches zero. This allows tensors to safely outlive the scope that created them (e.g., when assigned to a variable or returned from a function).

### 2. Runtime Mechanisms (MemoryManager)

The `MemoryManager` is a global singleton responsible for tracking all tensor allocations.

#### A. Reference Counting (`refcounts`)
Every tensor pointer (`*mut OpaqueTensor`) tracked by the system has an associated reference count in a `HashMap`.
- **Alloc**: Initial RefCount = 1 (Owned by the Scope).
- **Acquire**: RefCount += 1 (Shared ownership, e.g., variable assignment).
- **Release**: RefCount -= 1. If RefCount == 0, the underlying device memory is freed.

#### B. Scope Stack (`scopes`)
The manager maintains a stack of scopes (Vec of Vecs).
- **Enter Scope**: Pushes a new list to the stack.
- **Register**: Adds a tensor pointer to the current (top) scope.
- **Exit Scope**: Iterates through the current scope's list and calls **Release** on every pointer.

#### C. Handling Address Reuse (Critical)
Since `malloc` (or the underlying allocator) may reuse memory addresses of freed tensors, the `MemoryManager` must handle "Address Re-use" carefully.
- **Detection**: When registering a new tensor, if the address is already present in `is_registered` but NOT in `refcounts` (meaning it was freed), it is identified as a **Stale Record**.
- **Action**: The Stale Record is purged from the scope stack before the new tensor is registered. This prevents the "Double Free" or "Premature Free" bugs where an old scope entry would inadvertently release a new tensor.

#### D. Unregister vs. Release
- **Release**: Decrements RefCount. Used when an owner (Scope or Variable) goes away.
- **Unregister**: Removes the record from the Scope *without* decrementing RefCount.
    - **Use Case**: **Move Semantics**. When a temporary tensor is moved into a struct or specific raw-pointer context where scope management is no longer desired, but ownership is transferred (not shared). *Note: Modern `Let` bindings use Shared Ownership (Acquire) instead of Unregister.*

### 3. Compiler Ownership Model (Codegen)

The compiler generates code to enforce this strategy via LLVM IR.

#### A. Variable Assignment (`Stmt::Let`) = Shared Ownership
When a variable is defined (`let x = y`), we employ a **Shared Ownership** model.
- **Action**: `emit_deep_clone(y)` is called. For tensors, this performs an **Acquire** (Ref += 1).
- **Result**: The temporary `y` remains in the Scope (Ref 1). The variable `x` holds a new reference (Ref 2).
- **Cleanup**: When the scope exits, standard cleanup releases `y` (Ref 2 -> 1). Variable `x` persists until its own scope ends or it is dropped.

#### B. Shadowing = Release
When a variable is shadowed (`let x = ...; let x = ...;`), the old variable's handle is lost.
- **Action**: The compiler emits a **Release** call for the old `x` before overwriting it.
- **Rationale**: Variables are "Owned" references. If we lose the handle, we must release ownership.

#### C. Parameters & Global Registry
- **Parameters**: Global parameters (trainable weights) are registered in a global `VAR_MAP`.
- **Action**: `tl_register_parameter` explicitly **Acquires** result.
- **Lifecycle**: Temporary Scope (Ref 1) + Global Registry (Ref 1) = Ref 2. Scope Exit -> Ref 1 (Alive in Registry).

#### D. Structs & Vectors
- **Recursive Free**: When a container (Struct or Vec) is released, the runtime recursively iterates through its fields/elements and calls **Release** on them.

### 4. Structure Memory Management Strategy

**Allocation**:
- Structs are allocated on the heap via `malloc` but are **not refcounted** in the same way as tensors. They are simple Containers.
- However, Structs *containing* Tensors act as "Owners" of those tensors.

**Lifecycle**:
1.  **Creation (`StructInit`)**:
    -   Memory is allocated (`malloc`).
    -   Fields are populated. If a field is a Tensor, the struct **Acquires** it (Ref += 1) via `emit_deep_clone`.
    -   The Struct pointer is registered to the Scope (`tl_mem_register_struct`).

2.  **Usage**:
    -   Passed by pointer.
    -   Function Returns: When returning a struct, the compiler performs a **Shallow Copy** of the fields to a pre-allocated return slot (SRET). Since it's a copy of the *container* but the *content pointers* remain the same, we do NOT Acquire/Release the fields again during return to avoid overhead, relying on the caller to manage the new container instance.

3.  **Destruction**:
    -   When a struct goes out of scope, the runtime calls `free_struct`.
    -   **Recursive Free**: Crucially, this function iterates over all fields. If a field is a Tensor or another Struct, it calls `release` or `free` on it.
    -   `free(struct_ptr)` is called last.

---

## [Japanese] メモリ管理戦略

### 1. 中心哲学: 参照カウントとスコープベース管理のハイブリッド

TensorLogicは、**参照カウント (Reference Counting)** と **スコープベースの自動解放 (Scope-Based Automatic Deallocation)** を組み合わせた **ハイブリッドメモリ管理** システムを採用しています。

- **スコープ**: 一時的なテンソル（計算の中間結果など）のライフサイクルを管理します。スコープを抜ける際、そのスコープに登録されたすべてのテンソルの参照カウントがデクリメントされます。
- **参照カウント**: テンソルの正確なライフサイクルを管理します。参照カウントが0になった時点でのみ、実際のデバイスメモリが解放されます。これにより、テンソルが作成されたスコープよりも長く生存すること（変数への代入や関数からの返却など）が可能になります。

### 2. ランタイムの仕組み (MemoryManager)

`MemoryManager` は、すべてのテンソル割り当てを追跡するグローバルシングルトンです。

#### A. 参照カウント (`refcounts`)
システムが追跡するすべてのテンソルポインタ (`*mut OpaqueTensor`) は、`HashMap` で参照カウントを保持します。
- **Alloc (割り当て)**: 初期参照カウント = 1（スコープが所有）。
- **Acquire (共有)**: 参照カウント += 1（共有所有権、例：変数への代入）。
- **Release (解放)**: 参照カウント -= 1。0になった場合、メモリを解放します。

#### B. スコープスタック (`scopes`)
マネージャーはスコープのスタック（リストのリスト）を保持します。
- **Enter Scope**: 新しいリストをスタックにプッシュします。
- **Register**: 現在の（最上位の）スコープにテンソルポインタを追加します。
- **Exit Scope**: 現在のスコープリストを走査し、各ポインタに対して **Release** を呼び出します。

#### C. アドレス再利用のハンドリング（重要）
`malloc`（アロケータ）は解放されたメモリのアドレスを再利用する可能性があるため、`MemoryManager` は「アドレス再利用」を慎重に扱います。
- **検出**: 新しいテンソルを登録する際、そのアドレスが `is_registered` (登録済み) であるが `refcounts` には存在しない（解放済み）場合、それは **Stale Record（古いレコード）** とみなされます。
- **対処**: 新しい登録を行う前に、この古いレコードをスコープスタックから削除（Purge）します。これにより、古いスコープエントリが誤って新しいテンソルを解放してしまう「二重解放」や「早すぎる解放」バグを防ぎます。

#### D. Unregister (登録解除) と Release (解放) の違い
- **Release**: 参照カウントを減らします。所有者（スコープや変数）がいなくなる場合に使用します。
- **Unregister**: 参照カウントを減らさずに、スコープからレコードを削除します。
    - **用途**: **Moveセマンティクス**。一時テンソルの所有権を構造体や特定のポインタコンテキストに完全に「移動」させ、スコープ管理から外したい場合に使用します。 *注: 現在の `Let` 束縛は Unregister ではなく共有所有権 (Acquire) を使用しています。*

### 3. コンパイラの所有権モデル (Codegen)

コンパイラは、この戦略を適用するために以下のルールで LLVM IR を生成します。

#### A. 変数代入 (`Stmt::Let`) = 共有所有権 (Shared Ownership)
変数が定義される際 (`let x = y`)、**共有所有権** モデルを採用します。
- **動作**: `emit_deep_clone(y)` を呼び出します。テンソルの場合、これは **Acquire** (Ref += 1) を実行します。
- **結果**: 一時変数 `y` はスコープに残ります (Ref 1)。変数 `x` は新しい参照を持ちます (Ref 2)。
- **クリーンアップ**: スコープ終了時、標準のクリーンアップで `y` が解放されます (Ref 2 -> 1)。変数 `x` は自身のスコープが終わるか、ドロップされるまで生存します。

#### B. シャドーイング = Release
変数がシャドーイングされる場合 (`let x = ...; let x = ...;`)、古い変数のハンドルは失われます。
- **動作**: コンパイラは上書きする前に、古い `x` に対して **Release** 呼び出しを生成します。
- **理由**: 変数は「所有」参照であるため、ハンドルを失う場合は所有権を手放す（解放する）必要があります。

#### C. パラメータとグローバルレジストリ
- **パラメータ**: グローバルな学習可能パラメータ（重みなど）は `VAR_MAP` に登録されます。
- **動作**: `tl_register_parameter` は明示的に結果を **Acquire** します。
- **ライフサイクル**: 一時スコープ (Ref 1) + グローバルレジストリ (Ref 1) = Ref 2。スコープ終了 -> Ref 1（レジストリ内で生存）。

#### D. 構造体とベクタ
- **再帰的解放**: コンテナ（構造体やVec）が解放される際、ランタイムはそのフィールドや要素を再帰的に走査し、それぞれの要素に対して **Release** を呼び出します。

### 4. 構造体のメモリ管理戦略

**割り当て (Allocation)**:
- 構造体は `malloc` によってヒープに割り当てられますが、テンソルのような **参照カウント管理は行われません**。これらは単なるコンテナとして扱われます。
- ただし、テンソルを *含む* 構造体は、それらのテンソルの「所有者」として機能します。

**ライフサイクル**:

1.  **作成 (`StructInit`)**:
    -   メモリが割り当てられます (`malloc`)。
    -   フィールドが埋められます。フィールドがテンソルである場合、構造体はそれを **Acquire** (Ref += 1) します (`emit_deep_clone` 経由)。これにより、テンソルは構造体が存在する限り生存します。
    -   構造体自体のポインタがスコープに登録されます (`tl_mem_register_struct`)。

2.  **使用**:
    -   ポインタ渡しで関数に渡されます。
    -   **関数戻り値**: 構造体を返す際、コンパイラは呼び出し元が確保した戻り値用スロット (SRET) にフィールドの **シャローコピー (Shallow Copy)** を行います。コンテナは複製されますが、中身のポインタ（テンソル）は同じものを指すため、戻り値処理中に再度の Acquire/Release は行いません（呼び出し元が新しいコンテナインスタンスを管理します）。

3.  **破棄 (Destruction)**:
    -   構造体がスコープを抜ける際、ランタイムは `free_struct` を呼び出します。
    -   **再帰的解放 (Recursive Free)**: 重要な点として、この関数はすべてのフィールドを走査します。フィールドがテンソルや別の構造体である場合、それらに対して `release` または `free` を呼び出します。
    -   最後に `free(struct_ptr)` が呼び出され、コンテナ自体のメモリが解放されます。
