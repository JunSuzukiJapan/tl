# Vec<T> デザインと実装方針

`Vec<T>` (可変長配列) の実装におけるメモリ管理、レイアウト、成長戦略に関する方針をまとめます。

## 1. メモリレイアウト (Memory Layout)

**方針: 連続したメモリ領域 (Contiguous Memory) を使用する**

追加で確保したメモリをリンク構造（Linked Chunks）にするのではなく、常に**単一の連続したメモリブロック**として管理します。
これにより、インデックスアクセス `v[i]` が O(1) で高速に行え、CPUキャッシュ効率も良くなります。

**構造体定義 (TLイメージ):**
```rust
struct Vec<T> {
    ptr: ptr<T>,  // ヒープ上の連続メモリへのポインタ
    cap: I64,     // 確保済み容量 (要素数)
    len: I64,     // 現在の要素数
}
```

## 2. 初期化戦略 (Initialization)

**方針: `Vec::new()` はメモリ確保を行わない (Lazy Allocation), `with_capacity` は確保する**

*   **`Vec::new()`**:
    *   **理由**: Rustの仕様（調査結果）に準拠します。Rustの `Vec::new()` はメモリ割り当てを行わず、最初の `push` 時まで確保を遅延させます。これにより、空の `Vec` を作成するコストがほぼゼロになります。
    *   初期状態: `ptr: null`, `cap: 0`, `len: 0`

*   **`Vec::with_capacity(n)`** (初期実装に含める):
    *   ユーザーが「すぐに使いたい」とわかっている場合に明示的に使用します。
    *   `n > 0` なら、初期状態で `n * sizeof(T)` バイトを確保します。

## 3. 成長戦略 (Growth Strategy)

**方針: 容量不足時は倍々に拡張する (Exponential Growth)**

`push` 等で `len == cap` となった場合、以下のルールで再割り当て（Reallocation）を行います。

1.  **新しい容量 (`new_cap`) の決定**:
    *   現在の `cap` が `0` の場合: **4** から開始（`Vec::new` 後の最初の `push`）。
    *   現在の `cap > 0` の場合: **`cap * 2`** (倍増)。

2.  **再割り当て処理**:
    *   `realloc(ptr, new_cap * sizeof(T))` を使用します。
    *   所有権やMove Semanticsについては、Option/Resultと同様、ポインタ管理されたオブジェクトとして扱います。

## 4. 要素のメモリ管理 (Element Memory Management)

（変更なし）

## 5. 考慮すべきリスクと対策

*   **ZST (Zero Sized Types)**:
    *   `Vec<Void>` や空の構造体など、`sizeof(T) == 0` の場合。
    *   **方針**: **現時点ではコンパイルエラーまたはランタイムエラーとする**。
    *   `Vec` の実装を単純化するため、ZSTのサポートは後回しにします。`malloc(0)` を避けるためのガードを入れます。

*   **メモリ断片化**:
    *   頻繁な `realloc` は断片化を招く可能性がありますが、倍々ゲームの戦略（償却計算量 O(1)）により、再割り当て回数は `log N` に抑えられるため許容範囲とします。

---

## 実装ステップ

1.  `src/compiler/codegen/builtin_types/generic/vec.tl` の作成
2.  `realloc` 関数のランタイムマッピング追加 (`builtins.rs`)
3.  `Vec` 用の Codegen ロジック実装（特に `drop` 時のループ処理）
