# HashMap<K, V> デザインと実装方針

`HashMap<K, V>` (ハッシュマップ) の実装におけるデータ構造、アルゴリズム、メモリ管理に関する方針をまとめます。
`Vec<T>`（可変長配列）の整備が進んだため、これを基盤として実装します。

## 1. データ構造とメモリレイアウト (Data Structure)

**方針: Open Addressing (Linear Probing) を採用し、`Vec` をバッキングストアとする**

単一の `Vec` を使用し、その中にエントリーをフラットに格納します。ポインタによるチェーン（Chaining）はメモリアロケーション回数が増えるため、キャッシュ効率の良い Open Addressing を選択します。
削除処理（Tombstone）の単純化のため、エントリーは `enum` で状態管理します。

**構造体定義 (TLイメージ):**
```rust
// エントリーの状態を表すEnum
enum Entry<K, V> {
    Empty,           // 空きスロット
    Occupied(K, V),  // データあり
    Deleted,         // 削除済み (Tombstone: 検索チェーンを切らないために必要)
}

struct HashMap<K, V> {
    entries: Vec<Entry<K, V>>, // 実データ。capacityはこのVecのlen等しい。
    len: I64,                  // 現在格納されている要素数 (Occupiedの数)
}
```

### メモリ上の配置
すべてのデータは `entries` ベクタ内の連続領域に配置されます。
`Vec` がジェネリックに対応したため、`K` や `V` の型サイズに応じてメモリレイアウトは自動的に最適化（単相化）されます。

## 2. ハッシュ関数 (Hashing)

**方針: コンパイラ組み込み関数 `hash<T>(val: T) -> I64` を提供する**

TL言語にはまだトレイトシステム（Traits）がないため、任意の型 `T` に対してハッシュ値を計算する汎用的なメカニズムが必要です。

*   **`hash<T>(val: T)`**:
    *   コンパイラ組み込み（Intrinsic）として定義。
    *   Codegen時に `T` の型を見て、適切なハッシュアルゴリズム（またはランタイム関数）を呼び出すようにディスパッチします。
    *   **I64/F64等**: 値をそのまま使うと分布が偏るため、**乗算とビットシフトを組み合わせた簡易なミキシング関数**（Knuth's Multiplicative Hash等）を通す。
    *   **String**: **SipHash** を採用する。Rustランタイム側（`tl_runtime`）に実装されたSipHash関数を呼び出すことで、堅牢性とDoS耐性を確保する。
    *   **Struct/Enum**: 初期実装では**メモリアドレス（ポインタ値）**をベースに、簡易なビットミキシングを行ったものをハッシュ値とする。
        *   **重要**: これにより、構造体のキーは**参照等価性（Reference Equality）**を持つことになる。つまり、内容は同じでもインスタンスが異なると「別のキー」として扱われる。

## 3. 基本操作アルゴリズム

### 初期化 (`new`)
*   `entries`: 長さ・容量 0 の `Vec` で初期化（Lazy Allocation）。
*   `len`: 0。

### 挿入 (`insert`)
1.  **Load Factor チェック**: `(len + 1) / entries.len() > 0.75` ならリサイズ（後述）。
2.  **ハッシュ計算**: `h = hash(key)`.
3.  **探索 (Linear Probing)**:
    *   `idx = h % entries.len()` から開始。
    *   `Entry::Empty` または `Entry::Deleted` を見つけたら、そこに `Occupied(key, value)` を書き込んで終了。
    *   `Entry::Occupied(k, v)` で `k == key` なら、値を更新して終了。
    *   それ以外（衝突）なら `idx = (idx + 1) % entries.len()` して次へ。

### 検索 (`get`)
1.  `h = hash(key)`.
2.  `idx = h % entries.len()` から開始。
3.  ループ:
    *   `Entry::Empty`: キーが存在しない -> `Option::None` を返す。
    *   `Entry::Occupied(k, v)`: `k == key` なら `Option::Some(v)` を返す。
    *   `Entry::Deleted`: 無視して次へ。
    *   それ以外: 次へ。

### 削除 (`remove`)
1.  検索と同じロジックでキーを探す。
2.  見つかったら `Entry::Deleted` に書き換える（物理削除はしない）。
3.  `len` をデクリメント。

## 4. 成長戦略 (Resizing Strategy)

**方針: 容量不足時は新しい `Vec` に全要素を再ハッシュ (Rehash)**

*   **トリガー**: Load Factor (占有率) が 0.75 を超えた場合。
*   **新しい容量**: 現在の容量の **2倍**（初期サイズは 4 または 8）。
*   **プロセス**:
    1.  新しいサイズの `new_entries: Vec<Entry<K, V>>` を確保し、すべて `Empty` で埋める。
    2.  古い `entries` をイテレートし、`Occupied(k, v)` の要素のみを取り出す。
    3.  取り出した要素を新しい `new_entries` に再挿入（再ハッシュ計算とLinear Probing）。`Deleted` はここで消滅する（ガーベージコレクション効果）。
    4.  `self.entries` を `new_entries` に置き換える。

## 5. 考慮すべきリスクと対策

*   **無限ループ**:
    *   Load Factor を適切に管理すれば、必ず `Empty` スロットが存在するため、Linear Probing は必ず停止する。
    *   ただし、Hash DoS 攻撃などは考慮しない。

*   **ジェネリクスのコード肥大化**:
    *   `Entry<K, V>` という Enum が型ごとに生成されるため、バイナリサイズが増加する可能性がある。これは許容する。

*   **等価性チェック (`==`) の制約**:
    *   **構造体・Enumの場合**: 前述の通り、ハッシュ計算にアドレスを使用するため、等価性チェックも**ポインタの比較（アドレス一致）**で行う必要がある。
    *   **問題点**: ユーザーが直感的に期待する「値の等価性（すべてのフィールドが一致すれば同じキー）」は、初期実装ではサポートされない（`Point{x:1}` と `Point{x:1}` は別のキーになる）。
    *   **将来の拡張**: 将来的にはトレイトシステムを導入し、`Hash` と `Eq` インターフェースを実装させることで、ユーザー定義の振る舞いを可能にする。

*   **初期化時の `entries` サイズ**:
    *   `Vec` の仕様により `Vec::new()` はアロケーションしないため、最初の `insert` 時に必ずリサイズ（初期確保）が発生する。これは許容する。

## 実装ステップ

1.  **`src/compiler/codegen/builtin_types/generic/hashmap.tl` の作成**
    *   `Entry` Enum と `HashMap` Struct の定義。
    *   `new`, `insert`, `get` 等のメソッド実装（TL言語で記述）。
2.  **`hash<T>` 組み込み関数の実装**
    *   Parsing/Semantics での関数解決に追加。
    *   Codegen でのハッシュ計算ロジック生成 (`builtin_functions.rs` 等)。
3.  **コンパイラへのAST注入**
    *   `generic_type_loader` (仮称) 等を通じて `hashmap.tl` を読み込む仕組みの確認（`Vec` と同様）。
4.  **テスト**
    *   `tests/manual/test_hashmap_generic.tl` で動作検証。
