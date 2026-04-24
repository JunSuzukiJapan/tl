
# TensorLogic メモリ管理戦略 (ARC ベースのスコープ管理)

本ドキュメントでは、TensorLogic コンパイラのメモリ管理戦略を定義します。
すべての参照カウント管理対象型は、**対称的な ARC（Automatic Reference Counting）ライフサイクル**に基づいて管理されます。

※ ネイティブ関数 (FFI / C ABI) 境界における安全なメモリおよび型の受け渡し方については、[FFI_IMPLEMENTATION_GUIDE.md](./FFI_IMPLEMENTATION_GUIDE.md) もあわせて参照してください。

---

## 1. 中心哲学

### 1.1 対称的 ARC モデル

- すべてのインクリメント（RC+1）には、**必ず 1対1 で対応するデクリメント（RC-1）が存在する**。
- デクリメントは原則として **スコープを抜けた時** に行われる。
- 参照カウントが 0 になった時点で、メモリが解放される。

### 1.2 管理対象型

以下の型は ARC の管理対象であり、**すべて同一のルール** で管理されます。
本ドキュメントで「構造体」と記述する箇所は、これらすべてを含みます。

- **構造体 (Struct)** — ユーザー定義のデータ型
- **タプル (Tuple)** — 無名の複合データ型
- **構造を持つ列挙型 (Enum with payload)** — バリアントがフィールドを持つ列挙型
- **テンソル (Tensor)** — `Arc<UnsafeCell<CpuTensor>>` で管理される不透明ポインタ
- **文字列 (String)** — ヒープ割り当てされる文字列型
- **コレクション (Vec, HashMap 等)** — 内部にポインタを持つコンテナ型

プリミティブ型（`i64`, `f32`, `bool` 等）は値コピーであり、ARC の対象外です。

---

## 2. ARC ライフサイクルペア一覧

すべてのインクリメントが、最終的にどのタイミングでデクリメント（回収）されるのかをペアで列挙します。

### 2.1 変数の初期化と代入

```text
インクリメント：変数の初期化と代入 (Variable Assignment / Initialization)
デクリメント：その変数のスコープを抜けた時
```

変数に別な値が再代入された場合、古い値は「名前のない暗黙の一時ローカル変数」として現在のスコープに登録し直され、スコープを抜ける時に一括してデクリメントされます。

### 2.2 一時変数の生成

```text
インクリメント：メソッドや関数の戻り値・一時変数の生成 (Temporary Expression)
デクリメント：その一時変数のスコープを抜けた時
```

変数に代入されなかった一時変数（式の評価結果など）も、「名前のない暗黙の一時ローカル変数」として現在のスコープに登録・管理され、すべてスコープを抜ける時に一括してデクリメントされます。

### 2.3 関数引数（Callee 側でインクリメント）

```text
インクリメント：関数が呼び出された際、呼び出された関数側（引数を受け取った時）
デクリメント：呼び出された関数の実行が終了し、関数内の引数スコープを抜けた時
```

引数のインクリメントは **呼び出し側（Caller）ではなく、呼び出された関数側（Callee）** で行われます。
`self` 引数も他の引数とまったく同じルールで処理されます。

### 2.4 コレクション・構造体への格納

```text
インクリメント：コレクション・配列・構造体への格納 (Storing in Collection / Struct)
デクリメント：親となるコレクションや構造体が破棄された（親のスコープを抜けた）時
```

コレクション自体がスコープを抜けて破棄される際に、内部要素も連鎖的にデクリメントされます。途中で `pop` などで削除されてどこにも代入されなかった値も、一時的に暗黙の変数に入れられ、最終的に親ブロックのスコープを抜けた時にデクリメントされます。

### 2.5 if-let / match パターンマッチ分解

```text
インクリメント：if-let / match による列挙型のパターンマッチ分解 (Pattern Destructuring)
デクリメント：分解で束縛された変数のスコープ（if-let の then ブロック、match アーム本体）を抜けた時
```

`if-let` や `match` で列挙型の内部値を取り出す場合、取り出された各フィールドに対してインクリメントを行い、それぞれ独立した変数としてスコープに登録します。
元の列挙型の値自体は、通常の変数スコープルール（§2.1）に従い管理されます。

### 2.6 クロージャへのキャプチャ

```text
インクリメント：クロージャへのキャプチャ (Closure Value Capture)
デクリメント：クロージャ（関数オブジェクト）自体のスコープを抜け、破棄された時
```

### 2.7 非同期タスクへの保持

```text
インクリメント：非同期タスク（Poll / Future）への保持 (Storing in Async Task / State Machine)
デクリメント：非同期タスクが完了し、タスクの戻り値（Poll オブジェクト等）自体がスコープを抜け、破棄された時
```

---

## 3. 例外: ムーブセマンティクス (Last-Use Optimization)

```text
条件：変数がその後二度と使われない最後の使用（Last Use）である場合
→ インクリメントを省略する（所有権の移転 = ゼロコスト）
→ 元の変数はスコープ脱出時にもデクリメントしない（所有権は既に移転済み）
```

**メカニズム**:
1. **生存区間解析**: コンパイラは変数が最後に使用されるステートメントを特定する。
2. **コード生成**: 最後の使用時点では、`inc_ref` (retain) 操作を省略する。
3. **所有権移転**: 所有権は受信側の変数や関数に効率的に転送される。
4. **クリーンアップなし**: 元の変数はスコープ終了時にデクリメントされない（`CLEANUP_NONE` フラグで管理）。

**効果**: `inc_ref` / `dec_ref` のペアを削除し、深い呼び出しチェーンにおけるパフォーマンスを劇的に向上させる。

---

## 4. 戻り値の最適化 (SRET / DPS)

### 4.1 SRET の判定

以下の条件を満たす型は SRET（Struct Return = Destination Passing Style）で返却される：

| 型 | SRET |
|:---|:---|
| `Struct(name, _)` （`name != "Tensor"` かつ `name != "String"`） | **はい** |
| `Tensor` | いいえ（ポインタ返却） |
| `String` | いいえ（ポインタ返却） |
| `Enum`, `Tuple` | いいえ（ポインタ返却） |
| プリミティブ (`i64`, `f32`, `bool` 等) | いいえ（値返却） |

### 4.2 SRET のメカニズム

```
┌─ Caller ────────────────────────────────────┐
│  let sret_alloca = alloca(ptr)              │
│  call callee(sret_alloca, args...)          │
│  result = load(sret_alloca)                 │
│  → result をスコープに登録                   │
└─────────────────────────────────────────────┘

┌─ Callee ────────────────────────────────────┐
│  // sret_dest = 引数0（隠しポインタ）        │
│  let result = ... （値を構築）               │
│  store(result, sret_dest)                   │
│  unregister(result)  ← スコープから除外      │
│  emit_all_scopes_cleanup()                  │
│  return void                                │
└─────────────────────────────────────────────┘
```

- **Caller** が `alloca` でポインタを事前確保し、Callee に隠し第1引数として渡す。
- **Callee** は結果を構築後、`sret_dest` にストアし、自スコープから `unregister` して所有権を Caller に移転する。
- Callee の `exit_scope` では result を解放しない（`unregister` 済みのため）。

### 4.3 非 SRET の戻り値

テンソルや Enum 等の非 SRET 型は、ポインタを直接返す：

```
┌─ Callee ────────────────────────────────────┐
│  let t = Tensor::ones([2, 3])               │
│    → ランタイムがポインタを返す              │
│    → t をスコープに登録                      │
│                                             │
│  return t                                   │
│    → unregister(t) (スコープから除外)        │
│    → emit_all_scopes_cleanup (t をスキップ)  │
│    → ポインタを返す                          │
└─────────────────────────────────────────────┘
         │ ptr (浮遊状態)
         ▼
┌─ Caller ────────────────────────────────────┐
│  let t = callee()                           │
│    → ポインタを受け取り、スコープに登録       │
└─────────────────────────────────────────────┘
```

---

## 5. Deep Copy (True Deep Clone)

変数の代入（`let x = y`）において、`y` が L-value（既存の変数）の場合、**True Deep Copy** が行われます。

### 5.1 動作

```rust
fn emit_deep_clone(val, ty) -> new_val:
    match ty:
        Tensor     → tl_tensor_acquire(val)    // Arc::clone, RC+1
        Struct     → 新メモリ確保 (malloc + register)
                     各フィールドを再帰的に emit_deep_clone
        Enum       → 新メモリ確保 + タグコピー + ペイロードを再帰 Deep Clone
        String     → tl_string_clone(val)      // 新しい文字列を確保
        Primitive  → 値コピー（RC 操作なし）
```

### 5.2 R-value の場合（ムーブ）

R-value（関数呼び出しの戻り値、式の評価結果など）の場合は Deep Clone を行わず、テンポラリリストから除外して所有権を直接移転します（§3 のムーブセマンティクスと同等）。

---

## 6. スコープとクリーンアップ

### 6.1 スコープの種類

| スコープ | 範囲 |
|:---|:---|
| 関数スコープ | 関数全体。`tl_mem_function_enter` / `tl_mem_function_exit` で囲まれる |
| ブロックスコープ | `if` / `match` / `loop` / `{ }` の各ブロック。`enter_scope` / `exit_scope` で囲まれる |
| イテレーションスコープ | `for` / `while` ループの各イテレーション |

### 6.2 ループイテレーションスコープ

`for` / `while` ループでは、各イテレーションの終わりにスコープ内の変数とテンポラリがクリーンアップされます：

```
for i in 0..n {
    // enter_scope()
    let tmp = some_expr()    ← スコープに登録
    ...
    // イテレーション末:
    //   emit_cleanup_vars_in_scope()  ← tmp をデクリメント
    //   tl_mem_exit_scope()
    //   tl_mem_enter_scope()          ← 次のイテレーション用
}
```

### 6.3 再代入時の古い値

変数に新しい値が代入されると、古い値は「暗黙の一時変数」として現在のスコープに登録されます。
古い値のデクリメントはスコープ脱出時に一括して行われます。

### 6.4 emit_recursive_free

管理対象型のデクリメントは `emit_recursive_free` によって再帰的に行われます：

- **構造体**: コンテナの `dec_ref` → RC が 0 になった場合、全フィールドを再帰的に free → コンテナ自体を `free`
- **テンソル**: `tl_tensor_release_safe` → Arc の `dec_ref` → RC が 0 になった場合、CpuTensor を解放
- **Enum**: タグに応じたバリアントのフィールドを再帰的に free → コンテナ自体を `free`
- **Vec**: 全要素を再帰的に free → 内部バッファを free → コンテナを free

### 6.5 Tensor / GradTensor のメモリ管理

Tensor と GradTensor は、他の管理対象型（Struct, String, Vec 等）とは**根本的に異なるメモリ管理モデル**に従います。

#### 中心原則

1. **Tensor と GradTensor は同じ扱いをする** — 参照カウントの増減ルール、スコープ管理、プール返却はすべて同一。
2. **参照カウントは常に増減させる** — Deep Clone 時に RC+1、スコープ脱出時に RC-1。
3. **RC=0 になったらプールに返す（メモリ解放は行わない）** — テンソルのメモリは OS に返さず、サイズごとにプールして使い回す。
4. **GradTensor は grad 部分を追加で考慮する** — autograd 計算グラフへの参照を持つため、グラフのクリーンアップを適切に行う。

#### codegen 側のルール

| 操作 | Tensor / GradTensor |
|:---|:---|
| `emit_cleanup_vars_in_scope` | RC-1 を行う。RC=0 の場合はプールに返す（free しない） |
| `add_temp_with_mode` | 追跡する |
| `emit_recursive_free` | 専用ブランチで処理（null チェック → RC-1 → プール返却） |
| 関数の戻り値 | `unregister` で管理。Caller がスコープに登録 |

#### ランタイム側のライフサイクル

```
テンソル生成
  tl_tensor_randn / tl_tensor_zeros / tl_tensor_ones 等
    → 新規テンソルを生成（プールに同サイズがあればプールから取得）
    → 初期 RC = 1

テンソル共有 (Deep Clone ではなく RC+1)
  tl_tensor_acquire(ptr)
    → RC + 1
    → 同じポインタを返す（データのコピーは行わない）

RC 減算 (スコープ脱出時) — Tensor / GradTensor 共通
  RC - 1
  RC > 0 → 他の参照が生きているため、何もしない
  RC = 0 → プールに返す（メモリ解放は行わない）
    → テンソルは (要素数, dtype, device) のキーでフリーリストに分類される
    → 次回同サイズのテンソル生成時にプールから再利用される

GradTensor 固有の処理
  RC = 0 でプールに返す前に、grad 部分のクリーンアップを行う
    → autograd グラフへの参照を解放する
    → grad テンソル自体もプールに返す
```

#### ⚠️ GradTensor 管理の重要な制約

1. **`enable_grad()` の戻り値は `Void` でなければならない**
   - `enable_grad()` は in-place の void FFI。戻り値を `GradTensor` 型にすると、codegen が self ポインタを `_discard` 変数に格納し、スコープ cleanup で元のテンソルに `tl_tensor_release_safe` が呼ばれてデータが破壊される。

2. **`get_grad()` はデータの独立コピーを返さなければならない**
   - GPU (Metal) では `get_grad()` が返すテンソルの Buffer と、元テンソルの `autograd.grad.buffer` が同じ Arc を共有すると、GradTensor の cleanup 時に Buffer の参照カウントが不整合を起こす。
   - CPU では `shallow_clone()` でデータ Vec がクローンされるため問題にならないが、Metal では `Arc<Buffer>` の共有が直接メモリ寿命に影響する。
   - Metal の `get_grad()` は `clone_data()` で GPU バッファを Blit コピーし、独立した Buffer を持つテンソルを返す。

#### プールに返す（free しない）理由

1. **パフォーマンス**: テンソルの malloc/free は高コスト。プール再利用により割り当てオーバーヘッドをゼロに近づける。
2. **GPU RSS 安定化**: Metal/CUDA ドライバは free したメモリを OS に即座に返さない。プール方式により RSS の不要な膨張を防ぐ。
3. **autograd グラフの安全性**: `.grad()` が返すテンソルは autograd グラフ内のテンソルと参照を共有する場合がある。free すると共有先が use-after-free になるが、プール返却なら安全。


---

## 7. ZST (ゼロサイズ型) 戦略

### 7.1 問題

`struct Empty {}` 等のゼロサイズ型（ZST）は 0 バイトのメモリを占有します。
`malloc(0)` の挙動はプラットフォーム依存で未定義です。

### 7.2 解決策: "ZST = NULL"

- **コンパイラ**: フィールドがない構造体の初期化時に **NULL ポインタ** (`i8* null`) を返す。`malloc` は完全にスキップ。
- **ランタイム**: `inc_ref`, `dec_ref`, `register`, `release` は `ptr == NULL` を検知すると即座にリターン（No-Op）。

### 7.3 理由

- **パフォーマンス**: 割り当てコストも追跡コストもゼロ。
- **安全性**: 有効なヒープポインタが NULL になることはない。
- **単純性**: グローバルシングルトンや 1 バイト割り当てのような複雑な仕組みが不要。

---

## 8. テンソルプール管理 (GPU Persistent Pool)

### 8.1 概要

GPU テンソルは **一度確保したメモリを OS に返さず、サイズごとにプールして使い回す** 戦略（Persistent Pool）で管理されます。これにより Metal/CUDA ドライバの RSS（Resident Set Size）膨張問題を回避します。

CPU テンソルにはプールは使用されず、Arc の参照カウントのみで管理されます。

### 8.2 プールのキー設計

テンソルは `(要素数, dtype_id, device_id)` のタプルをキーとしてフリーリストに分類されます：

```rust
// tensor_pool.rs
free_lists: HashMap<(usize, u8, u8), Vec<*mut OpaqueTensor>>
```

| dtype_id | 型 | バイト/要素 |
|:---|:---|:---|
| 0 | F32 | 4 |
| 1 | F64 | 8 |
| 2 | I32 | 4 |
| 3 | I64 | 8 |
| 4 | U8 | 1 |
| 5 | F16 | 2 |
| 6 | BF16 | 2 |

### 8.3 ライフサイクル

```
テンソル生成:
  1. pool_acquire(要素数, dtype, device) でフリーリストを検索
     HIT  → 既存ディスクリプタを取得 → drop_in_place で中身クリア → 再利用
     MISS → 新規 malloc + register_new_allocation
  2. ランタイムがデータを書き込み、ポインタを返す

テンソル不要:
  1. pool_release(ptr, 要素数, dtype, device)
  2. active セットから削除
  3. フリーリストに追加（OS には返さない）
  4. 次回同サイズの acquire で再利用される

プロセス終了:
  OS がプロセスの全メモリを回収
```

### 8.4 CPU と GPU の比較

| | CPU | GPU (Metal / CUDA) |
|:---|:---|:---|
| テンソル表現 | `Arc<UnsafeCell<CpuTensor<f32>>>` | `*mut OpaqueTensor` (プール管理) |
| 寿命管理 | Arc RC のみ (RC=0 で Drop) | Persistent Pool (解放しない) |
| 再利用 | なし（毎回新規 alloc） | サイズ別フリーリストから再利用 |
| スコープ管理 | No-op (V6 で廃止) | No-op |
| RSS 特性 | ~40MB で安定 | ワークロード依存で安定化 |

### 8.5 Autograd グラフピンニング

Autograd の計算グラフは、逆伝播のためにテンソルへの参照を保持します。この参照により `Arc::strong_count > 1` となり、テンソルのメモリが解放されません。

- **CPU**: Arc RC が自然に管理するため問題にならない
- **GPU**: Candle バックエンドのバッファが「ピンニング」され、プールに戻せない場合がある
- **対策**: `tl_tensor_detach` で明示的にグラフから切り離す。`tl_clear_grads()` がループ末尾に自動挿入される

### 8.6 モニタリング API

| 関数 | 動作 |
|:---|:---|
| `tl_get_gpu_total_allocated_bytes()` | 累計確保バイト数 |
| `tl_get_gpu_free_count()` | フリーリスト内テンソル数 |
| `tl_get_gpu_pool_hit_rate()` | プール命中率 |
| `tl_dump_gpu_pool_stats()` | 統計の詳細ダンプ |

デバッグ: 環境変数 `TL_POOL_DEBUG=1` で acquire/release の詳細ログを stderr に出力

---

## 9. ARC ランタイム API

### 9.1 グローバル・サイドテーブル

TL は **Global Side-Table** 方式で参照カウントを管理します。オブジェクトのヘッダにカウントを埋め込まず、グローバルな `HashMap<usize, usize>` でポインタアドレスごとのカウントを追跡します。

```rust
static REF_COUNTS: LazyLock<Mutex<HashMap<usize, usize>>> = ...;
```

### 9.2 コア FFI 関数

| 関数 | 動作 |
|:---|:---|
| `tl_ptr_inc_ref(ptr)` | RC+1（未登録の場合は 2 で初期化） |
| `tl_ptr_dec_ref(ptr) -> bool` | RC-1（0 になったら `true` を返す） |
| `tl_ptr_acquire(ptr) -> ptr` | `inc_ref` + ポインタ返却 |
| `tl_ptr_release(ptr)` | `dec_ref` → 0 なら `libc::free` |
| `tl_mem_register(ptr)` | スコープマネージャにポインタを登録 |
| `tl_mem_unregister(ptr)` | スコープマネージャからポインタを除外 |
| `tl_mem_enter_scope()` | 新しいスコープを開始 |
| `tl_mem_exit_scope()` | 現在のスコープを終了し、登録済みポインタを解放 |
| `tl_mem_function_enter(slots)` | 関数フレームを確保 |
| `tl_mem_function_exit()` | 関数フレームを解放 |
