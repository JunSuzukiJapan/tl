# TL言語 `async`/`await` 実装計画

## 設計方針

Rustと同じ **ステートレス・ステートマシン方式**（stackless coroutines）を採用する。

```tl
async fn fetch_data(url: String) -> String {
    let conn = connect(url).await;
    let data = conn.read().await;
    data
}

fn main() {
    let result = AsyncRuntime::block_on(fetch_data("example.com"));
    println(result);
}
```

### Future はトレイトとして定義する

`Future` は組み込み型ではなく、関連型 `Output` を持つ**トレイト**として定義する。

```tl
// 標準ライブラリとして提供される定義（builtin）
trait Future {
    type Output;
    fn poll(self) -> Poll<Self::Output>;
}

enum Poll<T> {
    Ready(T),
    Pending,
}
```

> **Note — Pin と Context の省略について**  
> Rustの`Future::poll`は `Pin<&mut Self>` と `Context<'_>` を受け取るが、TLでは:
> - `Pin`: TLの構造体はポインタ渡しが基本のため、move安全性はコンパイラが暗黙的に保証する。明示的なPinは不要。
> - `Context`/`Waker`: Phase 3で簡易実装を追加。初期フェーズでは省略。

コンパイラは `async fn foo() -> T` を以下に変換する：

```
async fn foo(arg: i64) -> String
        ↓ コンパイラ変換
// 1. ステートマシン構造体（コンパイラが生成、ユーザーには不透明）
struct FooState {
    discriminant: i32,         // 現在の状態（サスペンションポイント番号）
    arg: i64,                  // awaitをまたぐ引数
    local_x: i64,              // awaitをまたぐローカル変数
    sub_future: *mut SubState, // 内部でawaitするFuture
}

// 2. Future トレイトの自動 impl（コンパイラが生成）
impl Future for FooState {
    type Output = String;
    fn poll(self) -> Poll<String> { ... }  // switch によるステートマシン
}

// 3. コンストラクタ（async fn 呼び出し時に実行）
fn foo(arg: i64) -> FooState { ... }  // FooState は impl Future
```

`.await` は `Future` トレイトを実装する任意の型に使用できる。  
型推論: `expr.await` の型 = `<typeof(expr) as Future>::Output`

---

## 前提: 現在のトレイト実装状況

| 機能 | 状況 |
|------|------|
| `type Output;`（trait 内）のパース | ✅ 実装済み（`TraitDef.associated_types`） |
| `type Output = T;`（impl 内）のパース | ✅ 実装済み（`TraitImplBlock.associated_types`） |
| 関連型の意味解析（型推論・解決） | ❌ **未実装**（`semantics.rs` で `associated_types` が無視されている） |
| 関連型のコード生成 | ❌ **未実装** |
| `T::Output` 形式の型参照 | ❌ **未実装** |

**∴ Phase 0 として関連型セマンティクスの完成が必要。**

---

## Phase 0: 関連型セマンティクスの完成（前提条件）

**対象ファイル**: `src/compiler/semantics.rs`, `src/compiler/type_engine.rs`, `src/compiler/ast.rs`

### 0.1 型表現の拡張

- [ ] **0.1.1** `Type::AssocType { base: Box<Type>, trait_name: String, assoc_name: String }` を追加  
  （例: `<T as Future>::Output` を表現する型）
- [ ] **0.1.2** Parser: `<T as Trait>::Assoc` 形式の型をパース（Fully Qualified Syntax）
- [ ] **0.1.3** Parser: `T::Assoc` の省略形をパース（単一トレイト境界で一意に決まる場合）

### 0.2 セマンティクスでの関連型解決

- [ ] **0.2.1** `SemanticAnalyzer` に `TraitRegistry`（trait名 → `TraitDef`）のルックアップを追加
- [ ] **0.2.2** `impl Trait for Type` の `associated_types` を `SemanticAnalyzer` に登録する処理を実装
- [ ] **0.2.3** `<T as Trait>::Assoc` の型解決: `T` の `impl Trait` を探し、対応する `type Assoc = ...` を返す
- [ ] **0.2.4** トレイト境界内での関連型解決: `T: Future` のとき `T::Output` を解決可能にする
- [ ] **0.2.5** 関連型に対する型ユニフィケーション（`TypeEngine::unify` を拡張）

### 0.3 コード生成での関連型対応

- [ ] **0.3.1** `impl Trait for Type` ブロックの `associated_types` を codegen のシンボルテーブルに登録
- [ ] **0.3.2** `<T as Trait>::Assoc` をコード生成時に具象型に置換するパスを実装

### 0.4 検証

- [ ] **0.4.1** `type Item` を使う `Iterator` トレイトの簡単な end-to-end テストを作成

```tl
// テスト例
trait Converter {
    type Output;
    fn convert(self) -> Self::Output;
}

struct Wrapper { val: i64 }

impl Converter for Wrapper {
    type Output = String;
    fn convert(self) -> String {
        String::from_int(self.val)
    }
}
```

---

## Phase 1: 構文層（Lexer / Parser / AST）

**対象ファイル**: `src/compiler/lexer.rs`, `src/compiler/parser.rs`, `src/compiler/ast.rs`

- [ ] **1.1** Lexer に `Async` トークンを追加（`logos` マクロ）
- [ ] **1.2** Lexer に `Await` トークンを追加（`logos` マクロ）
- [ ] **1.3** `FunctionDef` に `is_async: bool` フィールドを追加
- [ ] **1.4** `ExprKind::Await(Box<Expr>)` バリアントを追加（`.await` 後置記法）
- [ ] **1.5** Parser: `async fn name(...)` のパース（`fn` の前に `async` を検出してフラグをセット）
- [ ] **1.6** Parser: `expr.await` のパース（`.` の後が `Await` キーワードの場合を特別処理）
- [ ] **1.7** パースエラーメッセージの追加（`async fn` の構文違反）

> **Note — `Type::Future` は追加しない**  
> `Future` はトレイトなので、型としては `Type::Struct("FooState", [])` のような具体型か、
> トレイト境界 `T: Future` で表現する。コンパイラ内部での `impl Future` 型の追跡には
> 既存の `TraitBound` 機構を使用する。

```rust
// ast.rs への追加
pub struct FunctionDef {
    // ... 既存フィールド ...
    pub is_async: bool,   // NEW
}

pub enum ExprKind {
    // ... 既存バリアント ...
    Await(Box<Expr>),     // NEW: expr.await
}
```

---

## Phase 2: Future トレイト + 型システム

**対象ファイル**: `src/compiler/semantics.rs`, `src/compiler/type_engine.rs`,
`src/compiler/codegen/builtin_types/`

### 2.1 組み込みトレイト定義

- [x] **2.1.1** `Poll<T>` を組み込み enum として TL ランタイムに追加
- [x] **2.1.2** `Future` トレイトを組み込みトレイトとして登録（`SemanticAnalyzer` 初期化時）
- [x] **2.1.3** `Future` の組み込み TL 定義ファイルを作成（`builtin_types/future.tl`）

```tl
// builtin_types/future.tl（コンパイラが自動ロード）
enum Poll<T> {
    Ready(T),
    Pending,
}

trait Future {
    type Output;
    fn poll(self) -> Poll<Self::Output>;
}
```

### 2.2 async fn の型チェック

- [ ] **2.2.1** `async fn foo() -> T` の戻り型を `impl Future<Output = T>` として型システムに登録
- [ ] **2.2.2** コンパイラ内部で各 `async fn` に対応するステートマシン型（`FooState`）を生成・追跡
- [ ] **2.2.3** 生成された `FooState` 型に `impl Future<Output = T>` を自動付与

### 2.3 `.await` の型推論

- [x] **2.3.1** `SemanticAnalyzer` に `in_async_fn: bool` コンテキストフラグを追加
- [x] **2.3.2** `expr.await` の型推論: `expr` の型 `T` が `Future` を実装しているか確認
- [x] **2.3.3** `T::Output`（`<T as Future>::Output`）を解決して `.await` 式の型とする（Phase 0.2 依存）
- [x] **2.3.4** `async fn` 外での `.await` 使用をエラーとして報告
- [x] **2.3.5** `Future` を実装しない型への `.await` 適用をエラーとして報告
- [ ] **2.3.6** `async fn foo() -> Result<T, E>` + `?` 演算子の組み合わせ検証

---

## Phase 3: ランタイム基盤

**対象ファイル**: `crates/tl_runtime/src/executor_ffi.rs`（新規）

- [x] **3.1** `tl_executor_block_on(poll_fn, state) -> u64` を tl_runtime に実装（スピンループ）
- [x] **3.2** `tl_task_spawn(poll_fn, state) -> i64` を tl_runtime に実装（タスクID を返す）
- [x] **3.3** `tl_task_join(task_id) -> u64` を tl_runtime に実装
- [ ] **3.4** `TASK_REGISTRY: HashMap<i64, TaskState>` グローバル管理（`THREAD_REGISTRY` パターンに倣う）
- [ ] **3.5** `AsyncRuntime` 組み込み型の TL 定義ファイルを作成（`builtin_types/async_runtime.tl`）
- [ ] **3.6** `AsyncRuntime::block_on` を codegen のビルトイン関数として登録
- [ ] **3.7** 簡易 `Waker` FFI 構造体の定義（コールバックポインタ1つの最小実装）

```rust
// crates/tl_runtime/src/executor_ffi.rs（新規）
#[no_mangle]
pub extern "C" fn tl_executor_block_on(
    poll_fn: *const (),
    state: *mut (),
) -> u64 {
    loop {
        let result = unsafe { call_poll(poll_fn, state) };
        match result {
            Poll::Ready(val) => return val,
            Poll::Pending => std::hint::spin_loop(),
        }
    }
}
```

---

## Phase 4: コード生成（ステートマシン変換）

**対象ファイル**: `src/compiler/codegen/mod.rs`, `src/compiler/codegen/expr/mod.rs`

### 4.1 サスペンションポイント解析

- [ ] **4.1.1** `async fn` 本体を走査して `.await` 式を列挙するパスを実装
- [ ] **4.1.2** 各サスペンションポイントに番号（state 番号）を割り振る
- [ ] **4.1.3** await をまたぐローカル変数の liveness 解析（既存の liveness analysis を拡張）
- [ ] **4.1.4** state struct に格納すべき変数のリストを生成

### 4.2 ステートマシン構造体の生成

- [ ] **4.2.1** `{fn_name}State` LLVM 構造体型を動的に生成  
  フィールド: `discriminant: i32`, await をまたぐローカル変数, サブ Future ポインタ群
- [ ] **4.2.2** フィールドのインデックスマッピングを管理するデータ構造を実装
- [ ] **4.2.3** ステートマシン構造体のヒープ確保・解放ヘルパーを実装

### 4.3 `poll` 関数の生成

- [ ] **4.3.1** `poll_{fn_name}(state: *mut {fn_name}State) -> Poll<T>` のプロトタイプを生成
- [ ] **4.3.2** `discriminant` への switch 命令をエントリポイントとして生成
- [ ] **4.3.3** 各 state ブロック（`state_0:`, `state_1:`, ...）を生成
- [ ] **4.3.4** state ブロック間での変数の load/store（state struct 経由）を実装
- [ ] **4.3.5** サスペンション時: `discriminant` を次の state 番号に更新 + `Poll::Pending` を return
- [ ] **4.3.6** 完了時: `Poll::Ready(val)` を return

```
; 生成されるLLVMコードのイメージ
define %Poll_String @poll_foo(%FooState* %self) {
entry:
  %disc = load i32, i32* getelementptr(%self, 0, 0)
  switch i32 %disc, label %invalid [
    i32 0, label %state_0
    i32 1, label %state_1
    i32 2, label %state_done
  ]

state_0:
  ; sub_future_0 = connect(arg) を作成して state struct に保存
  store i32 1, i32* %self.discriminant
  br label %state_1

state_1:
  %sf0 = load %ConnectState*, %self.sub_future_0
  %r1 = call %Poll_Conn @poll_connect(%sf0)
  ; Pending → Return Poll::Pending
  ; Ready(conn) → conn を state に保存、discriminant = 2
  ...

state_done:
  ret %Poll_String { Ready, %result }
}
```

### 4.4 コンストラクタ関数の生成

- [ ] **4.4.1** 元の `async fn` 名のシグネチャ `foo(args...) -> FooState` を生成（戻り型は impl Future）
- [ ] **4.4.2** `tl_alloc` で state struct をヒープ確保
- [ ] **4.4.3** `discriminant = 0`（初期状態）にセット
- [ ] **4.4.4** 引数を state struct にコピー
- [ ] **4.4.5** `{poll_fn_ptr, state_ptr}` のファットポインタ（クロージャと同形式）を返す

### 4.5 `Future` トレイト impl の自動生成

- [ ] **4.5.1** `impl Future for FooState` ブロックをコンパイラが自動的に生成・登録
- [ ] **4.5.2** `type Output = T;` を関連型として設定
- [ ] **4.5.3** `fn poll(self) -> Poll<T>` を Phase 4.3 で生成した LLVM 関数に接続

### 4.6 `.await` 式のコンパイル（async fn 内部）

- [ ] **4.6.1** `.await` の対象 Future を評価してサブ Future ポインタを state struct に格納
- [ ] **4.6.2** `discriminant` に次の state 番号をセットするコードを生成
- [ ] **4.6.3** `poll` ループを生成（`Poll::Pending` なら外側の `poll` 関数から return）
- [ ] **4.6.4** `Poll::Ready(val)` の場合は `val` を `.await` 式の値として継続

### 4.7 統合・動作検証

- [ ] **4.7.1** `async fn` の codegen を `compile_fn` から分岐させる（`is_async` フラグで切り替え）
- [ ] **4.7.2** 最小構成の `async fn`（await 1回）の end-to-end テストを作成
- [ ] **4.7.3** 複数の await をまたぐケースのテスト
- [ ] **4.7.4** ネストした `async fn`（async fn が async fn を await する）のテスト
- [ ] **4.7.5** `async fn` + `Result<T, E>` + `?` 演算子の組み合わせテスト
- [ ] **4.7.6** ユーザー定義の `impl Future` が `.await` できることを確認

---

## Phase 5: 組み込み非同期プリミティブ

**対象ファイル**: `src/compiler/codegen/builtin_types/` 以下（新規ファイル群）

- [ ] **5.1** `sleep(ms: i64) -> impl Future<Output = ()>` — タイマー Future
- [ ] **5.2** `AsyncChannel<T>` — 非同期チャネル
  - [ ] **5.2.1** `async fn send(val: T)` — バッファが満杯なら suspend
  - [ ] **5.2.2** `async fn recv() -> T` — データがなければ suspend
- [ ] **5.3** `AsyncRuntime::spawn(fut) -> TaskHandle<T>` — バックグラウンドタスク起動
- [ ] **5.4** `TaskHandle` に `impl Future` を実装（`await` でタスク完了を待機）
- [ ] **5.5** `join!(f1, f2, f3)` マクロ — 複数 Future の並行実行
- [ ] **5.6** `select!(f1, f2, f3)` マクロ — 最初に完了した Future を選択

---

## 依存関係

```
Phase 0 (関連型セマンティクス)  ← 全フェーズの前提
    │
    ▼
Phase 1 (構文: async/await キーワード)
    │
    ▼
Phase 2 (Future トレイト + 型推論)
    │               ╲
    ▼                ▼
Phase 4 (Codegen)   Phase 3 (Runtime)  ← Phase 2 と並行可能
    │
    ▼
Phase 5 (組み込みプリミティブ)
```

---

## 既存コードの活用ポイント

| 既存実装 | async/await への転用方法 |
|---------|------------------------|
| `TraitDef.associated_types` / `TraitImplBlock.associated_types`（パース済み） | Phase 0 の関連型解決の入力データとして直接利用 |
| クロージャのファットポインタ `{fn_ptr, env_ptr}` | Future の `{poll_fn, state_ptr}` として同様の形式を採用 |
| `Thread::spawn` のトランポリン合成パターン（`codegen/expr/mod.rs:3480–3598`） | `poll` 関数の動的生成に同パターンを流用 |
| `THREAD_REGISTRY` グローバル管理（`tl_runtime/src/thread_ffi.rs`） | `TASK_REGISTRY` として同パターンを複製 |
| `crossbeam_channel`（既存依存） | `AsyncChannel` の実装基盤として流用 |
| Liveness Analysis（既存の codegen） | await をまたぐ変数の特定に活用 |
| クロージャのキャプチャ変数 store/load パターン | state struct への変数退避・復元に同パターンを流用 |

---

## 実装開始の推奨順序

1. **Phase 0**（関連型セマンティクス）— **全フェーズの前提条件**。既存のパース済みデータを活用するため着手コストは低い
2. **Phase 1**（Lexer/AST）— 既存コードを壊さず追加のみ。Phase 0 と並行可能
3. **Phase 2.1**（`Poll`/`Future` 組み込み定義）— Phase 0 完了後、Codegen 前に固める
4. **Phase 3**（Runtime）— Phase 4 と並行して進められる
5. **Phase 2.2〜2.3 + Phase 4**（型推論 + Codegen）— 全フェーズ中最難関
6. **Phase 5**（プリミティブ）— Phase 4 完了後
