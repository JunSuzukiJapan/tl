# TensorLogic Memory Management Strategy V2 (Hybrid Static Analysis)
# TensorLogic メモリ管理戦略 V2 (ハイブリッド静的解析)

This document outlines the **V2 Memory Memory Management Strategy** implemented in TensorLogic.
This strategy resolves the performance bottlenecks of the V1 (RefCounting + Scope) system by introducing **Static Analysis** and **Liveness-based Memory Reuse**.

本ドキュメントでは、TensorLogicに実装された **V2 メモリ管理戦略** について概説します。
この戦略は、**静的解析** と **生存区間ベースのメモリ再利用** を導入することで、V1（参照カウント+スコープ）システムのパフォーマンスボトルネックを解消します。

---

## [English] Memory Management Strategy V2

### 1. Core Philosophy: "Zero-Overhead Abstraction for Memory"

The V2 system aims to determine **at compile time** exactly *when* memory is needed and *when* it can be reused.

- **Static Analysis**: The compiler analyzes the lifespan of every tensor variable (Liveness Analysis).
- **Stack-Like Allocation**: Tensors are allocated in a pre-calculated "Function Frame" in a linear Arena, rather than individual `malloc` calls.
- **Memory Reuse (Aliasing)**: Two variables that never exist simultaneously (e.g., `a` in the first half of a function and `b` in the second half) are assigned the **same memory slot**.
- **Hybrid Approach**: If tensor shapes cannot be determined at compile-time (Dynamic Shapes), the system falls back to the V1 Dynamic Allocator (malloc/free).

### 2. Compiler Pipeline (Static Analysis)

The memory optimization happens in several passes before CodeGen:

#### Phase 1: Shape Inference & Liveness Analysis
The compiler walks through the AST to:
1.  **Infer Shapes**: Propagate tensor shapes (e.g., `[10, 10] * [10, 10] -> [10, 10]`).
2.  **Calculate Liveness**: Track the "Start" and "End" statement index for every variable.
    -   *Start*: When the variable is defined.
    -   *End*: The last time the variable is used.

#### Phase 2: Greedy Slot Allocation (Linear Scan)
Using the liveness intervals, the compiler runs a **Greedy Allocation Algorithm**:
1.  It maintains a list of "Active Intervals" (variables currently in memory).
2.  For each new variable, it tries to find a **free offset** in the current Function Frame that comfortably fits the tensor size.
3.  If a previous variable has "died" (its End < Current Start), its offset is marked as free and **reused**.
4.  **Result**: A map of `Variable -> Offset` and a total `FrameSize` required for the function.

#### Phase 3: Destination Passing Style (DPS)
For operations that return large tensors (e.g., `matmul`), the compiler passes the **destination pointer** (calculated from the Offset) as a hidden argument.
-   **V1**: `let c = matmul(a, b)` -> `c = malloc(...); matmul(a, b)` -> return pointer.
-   **V2**: `matmul(a, b, &frame[offset_c])`. The operation writes directly into the pre-allocated slot. Zero copy, zero malloc.

---

### 3. Runtime Mechanisms

The Runtime has been updated to support this "Arena-based" model seamlessly.

#### A. Memory Arena (Per-Thread)
Each thread maintains a large, contiguous block of memory (The Arena).
-   **Stack-like usage**: When entering a function, we simply move the "Stack Pointer" of the arena forward by `FrameSize`.
-   **Alloc**: `ptr = arena_base + current_offset + variable_offset`. (Integer addition, extremely fast).
-   **Free**: When exiting the function, we simply move the "Stack Pointer" back. No individual `free()` calls.

#### B. Fallback Mechanism (Dynamic Tensors)
If the shape inference fails (e.g., loading data from a file with unknown dimensions), the compiler marks the variable as **Dynamic**.
-   Dynamic variables are handled via the V1 **MemoryManager** (malloc + refcount).
-   They are NOT placed in the Arena to prevent fragmentation or overflow.

---

### 4. Benefits (V1 vs V2)

| Feature | V1 (Old) | V2 (New) |
| :--- | :--- | :--- |
| **Allocation** | `malloc` per tensor (Slow) | Pointer bump (Instant) |
| **Deallocation** | `free` per tensor (Slow) | Function exit (Instant) |
| **Peak Memory** | High (Cumulative) | **Minimal** (Reused via Liveness) |
| **Fragmentaton** | High | Low (Contiguous stacks) |
| **Analysis** | None (Runtime only) | **Full Static Analysis** |

---

### 5. Code Example & Visualization

```rust
fn process() {
    // Offset 0, Size 1MB
    let a = Tensor::zeros([512, 512]); 
    
    // Offset 1MB, Size 1MB
    let b = a + 1.0;
    
    // 'a' is dead here. Memory available from 0..1MB.
    
    // Offset 0, Size 1MB (REUSES 'a's memory!)
    let c = b * 2.0; 
}
```

In V1, this would require 3MB peak memory.
In V2, due to Liveness Analysis + Aliasing, this requires only **2MB** (or even less with advanced reusing), and consistently reuses the cache-hot memory at Offset 0.

---

## [Japanese] メモリ管理戦略 V2 (ハイブリッド静的解析)

### 1. 中心哲学: "メモリのためのゼロオーバーヘッド抽象化"

V2システムの目標は、**コンパイル時** に「いつメモリが必要で、いつ再利用できるか」を完全に決定することです。

- **静的解析**: コンパイラはすべてのテンソル変数の生存期間（Lifetime）を解析します（生存区間解析）。
- **スタックライクな割り当て**: テンソルは個別の `malloc` ではなく、リニアな「アリーナ」領域上の事前に計算された「関数フレーム」内に割り当てられます。
- **メモリ再利用 (Aliasing)**: 同時に存在しない2つの変数（例：関数の前半の `a` と後半の `b`）は、**全く同じメモリスロット** に割り当てられます。
- **ハイブリッドアプローチ**: コンパイル時にテンソル形状が確定しない場合（Dynamic Shapes）は、自動的にV1の動的アロケータ（malloc/free）にフォールバックします。

### 2. コンパイラパイプライン (静的解析)

CodeGen（コード生成）の前に、以下の最適化パスが実行されます。

#### Phase 1: 形状推論と生存区間解析 (Shape Inference & Liveness Analysis)
ASTを走査し、以下を行います：
1.  **形状推論**: テンソルの形状を伝播させます（例: `[10, 10] * [10, 10] -> [10, 10]`）。
2.  **生存区間計算**: 各変数の「開始」と「終了」のステートメントインデックスを追跡します。
    -   *Start*: 変数が定義された時点。
    -   *End*: 変数が最後に使用された時点。

#### Phase 2: Greedyスロット割り当て (Linear Scan Allocation)
生存区間情報を用いて、**Greedy割り当てアルゴリズム** を実行します：
1.  「アクティブな区間」（現在メモリ上にある変数）のリストを維持します。
2.  新しい変数に対し、現在の「関数フレーム」内でサイズが収まる **空きオフセット** を探します。
3.  過去の変数で「寿命が尽きた」もの（End < Current Start）があれば、そのオフセットを空きとしてマークし、**再利用** します。
4.  **結果**: `変数 -> オフセット` のマップと、関数全体で必要な `TotalFrameSize` が決定されます。

#### Phase 3: DPS (Destination Passing Style)
巨大なテンソルを返す操作（`matmul`など）において、コンパイラは（オフセットから計算された）**出力先ポインタ** を隠れ引数として関数に渡します。
-   **V1**: `let c = matmul(a, b)` -> `c = malloc(...); matmul(a, b)` -> ポインタを返す。
-   **V2**: `matmul(a, b, &frame[offset_c])`. 操作は事前に割り当てられたスロットに直接書き込みます。ゼロコピー、ゼロmallocです。

---

### 3. ランタイムメカニズム

ランタイムはこの「アリーナベース」モデルをシームレスにサポートするよう更新されました。

#### A. メモリアリーナ (スレッドごと)
各スレッドは巨大な連続メモリブロック（アリーナ）を保持します。
-   **スタック的な使用**: 関数に入る際、アリーナの「スタックポインタ」を `FrameSize` 分だけ進めます。
-   **割り当て**: `ptr = arena_base + current_offset + variable_offset`。（単なる整数加算であり、極めて高速）。
-   **解放**: 関数を抜ける際、アリーナの「スタックポインタ」を元に戻すだけです。個別の `free()` は不要です。

#### B. フォールバック機構 (動的テンソル)
形状推論に失敗した場合（例：次元不明のファイル読み込み）、コンパイラはその変数を **Dynamic** とマークします。
-   Dynamic変数は、従来の V1 **MemoryManager** (malloc + refcount) で扱われます。
-   これらはアリーナには配置されず、断片化やオーバーフローを防ぎます。

---

### 4. 比較 (V1 vs V2)

| 機能 | V1 (旧) | V2 (新) |
| :--- | :--- | :--- |
| **割り当て** | テンソル毎に `malloc` (遅い) | ポインタの加算 (一瞬) |
| **解放** | テンソル毎に `free` (遅い) | 関数終了時に一括 (一瞬) |
| **ピークメモリ** | 高い (累積的) | **最小限** (生存区間による再利用) |
| **断片化** | 高い | 低い (連続スタック) |
| **解析** | なし (実行時のみ) | **完全な静的解析** |

---

### 5. コード例と可視化

```rust
fn process() {
    // オフセット 0, サイズ 1MB
    let a = Tensor::zeros([512, 512]); 
    
    // オフセット 1MB, サイズ 1MB
    let b = a + 1.0;
    
    // ここで 'a' はもう使用されない (Dead)。 0..1MB の領域は空く。
    
    // オフセット 0, サイズ 1MB ('a' のメモリを再利用！)
    let c = b * 2.0; 
}
```

V1では、ピーク時に3MBが必要でした。
V2では、静的解析とエイリアシングにより、わずか **2MB**（さらに高度な再利用ならそれ以下）で済み、かつキャッシュ効率の良いオフセット0の領域を再利用し続けます。
