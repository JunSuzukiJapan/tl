# メモリマネージャー API リファレンス

ソース: `crates/tl_runtime/src/memory_manager.rs`

このドキュメントは、TLランタイムで使用されるメモリマネージャーのパブリックAPIの概要を説明します。LLVMが生成したコードで使用されるC ABI関数と、内部のRust APIに分かれています。

## C API (LLVM コード生成用)

これらの関数は `#[no_mangle]` および `extern "C"` でエクスポートされており、JITコンパイルされたコードから直接呼び出すことができます。

### スコープ管理 (Scope Management)
メモリスコープのライフサイクルを管理する関数です。

```rust
/// 新しいメモリスコープに入ります
#[no_mangle]
pub extern "C" fn tl_mem_enter_scope()

/// 現在のスコープを抜け、そのスコープ内のすべてのアロケーションを解放します
#[no_mangle]
pub extern "C" fn tl_mem_exit_scope()
```

### 関数フレーム管理 (Function Frame Management)
関数の実行フレーム内のメモリスロットを管理する関数です。

> **スロットバッファ (Slot Buffers) とは？**
>
> スロットバッファは、**関数フレーム（呼び出し）に紐づいた一時的な再利用可能メモリ**です。
> - **ライフサイクル**: `tl_mem_function_enter` でフレームが作成され、`tl_mem_function_exit` でそのフレーム内のすべてのバッファが自動的に解放されます。
> - **用途**: 文字列の結合処理や、一時的な計算結果の保持など、関数内でのみ有効な一時領域として使用されます。
> - **再利用と安全性 (Deferred Free)**: バッファサイズが不足した場合、新しいバッファを `malloc` し、**データコピーを行わずに**返します（以前の内容は引き継がれません）。
>   - **古いポインタの生存**: 古いバッファは即座に解放されず、「Deferred Free（遅延解放）リスト」に追加され、関数フレーム終了時 (`tl_mem_function_exit`) までメモリ確保されたまま保持されます。
>   - **安全性**: これにより、古いポインタを介したアクセスは引き続き有効です。新しいポインタは**未初期化**の状態で渡されるため、呼び出し元がデータを構築する必要があります。

```rust
/// 指定されたスロット数で関数フレームに入ります
#[no_mangle]
pub extern "C" fn tl_mem_function_enter(num_slots: i64)

/// 関数フレームを抜け、すべてのスロットバッファを解放します
#[no_mangle]
pub extern "C" fn tl_mem_function_exit()

/// 特定のスロットのバッファを取得します。必要に応じて再確保（realloc）が行われます
#[no_mangle]
pub extern "C" fn tl_mem_get_buffer(slot_id: i64, min_size: i64) -> *mut c_void
```

### アロケーション登録 (Allocation Registration)
ポインタをメモリマネージャーに登録し、追跡および自動解放を行えるようにする関数です。

```rust
/// 構造体のアロケーションを登録します
#[no_mangle]
pub extern "C" fn tl_mem_register_struct(ptr: *mut c_void)

/// デバッグ用の名前付きで構造体を登録します
#[no_mangle]
pub extern "C" fn tl_mem_register_struct_named(ptr: *mut c_void, name: *const c_char)

/// Tensorのアロケーションを登録します
#[no_mangle]
pub extern "C" fn tl_mem_register_tensor(ptr: *mut OpaqueTensor)

/// Vecポインタのアロケーションを登録します
#[no_mangle]
pub extern "C" fn tl_mem_register_vec_ptr(ptr: *mut c_void)

/// 汎用的なポインタ登録（型ロジックを推論します）
#[no_mangle]
pub extern "C" fn tl_mem_register_ptr(ptr: *mut c_void)

/// ポインタの登録を解除します（再代入やreturn時など）
#[no_mangle]
pub extern "C" fn tl_mem_unregister(ptr: *mut c_void)
```

### 参照カウント (Reference Counting / ARC)
共有リソースの参照カウントを操作する関数です。

```rust
// --- 汎用ポインタ操作 ---

/// 参照カウントを増やします
#[no_mangle]
pub extern "C" fn tl_ptr_acquire(ptr: *mut c_void)

/// 参照カウントを減らし、0になった場合は解放します
#[no_mangle]
pub extern "C" fn tl_ptr_release(ptr: *mut c_void)

// --- Tensor 固有の操作 ---

/// Tensorの参照カウントを増やします
#[no_mangle]
pub extern "C" fn tl_tensor_acquire(ptr: *mut OpaqueTensor)

/// Tensorの参照カウントを減らします
#[no_mangle]
pub extern "C" fn tl_tensor_release(ptr: *mut OpaqueTensor)
```

### 検査・デバッグ (Inspection / Debug / Metrics)
メモリマネージャーの状態を検査する関数です。

```rust
/// 現在プールに保存されているTensorの数を取得します
#[no_mangle]
pub extern "C" fn tl_get_pool_count() -> i64

/// 現在生きているTensorの参照カウントエントリ数を取得します
#[no_mangle]
pub extern "C" fn tl_get_refcount_count() -> i64

/// 現在のスコープの深さを取得します
#[no_mangle]
pub extern "C" fn tl_get_scope_depth() -> i64
```

---

## Rust 内部 API (`impl MemoryManager`)

`MemoryManager` 構造体は、C APIの背後にあるコアロジックを実装しています。グローバルな `Mutex<MemoryManager>` を介してアクセスされます。

### ライフサイクル
```rust
pub fn new() -> Self
```

### スコープ操作
```rust
/// 新しいスコープに入ります
pub fn enter_scope(&mut self)

/// 現在のスコープを抜け、そのスコープ内の**すべて**のアロケーションを解放します
pub fn exit_scope(&mut self)
```

### 登録ロジック
```rust
pub fn register_struct(&mut self, ptr: *mut c_void)
pub fn register_tensor(&mut self, ptr: *mut OpaqueTensor)
pub fn register_vec_ptr(&mut self, ptr: *mut c_void)

/// **現在のスコープ**からのみポインタの登録を解除します（ムーブセマンティクス用）
pub fn unregister(&mut self, ptr: *mut c_void)

/// ポインタがいずれかのスコープに既に登録されているか確認します
pub fn is_registered(&self, ptr: *mut c_void) -> bool
```

### 参照カウント
```rust
/// 参照カウントを増やします（汎用）
pub fn acquire_ptr(&mut self, ptr: *mut c_void)

/// 参照カウントを減らし、0になれば解放します（汎用）
/// ポインタが見つかり処理された場合は true を返します
pub fn release_ptr(&mut self, ptr: *mut c_void) -> bool
```

### 関数フレーム
```rust
pub fn function_enter(&mut self, num_slots: usize)
pub fn function_exit(&mut self)
pub fn get_buffer(&mut self, slot_id: usize, min_size: usize) -> *mut c_void
```

---

## Tensor Pool API (`impl TensorPool`)

`TensorPool` は、Tensorメモリを再利用することで、頻繁な `malloc`/`free` 呼び出しを回避します。

```rust
impl TensorPool {
    pub fn new() -> Self

    /// プールからTensorの取得を試みます
    pub fn acquire(
        &mut self, 
        num_elements: usize, 
        dtype_id: u8, 
        device_id: u8
    ) -> Option<*mut OpaqueTensor>

    /// Tensorをプールに解放（返却）します
    pub fn release(
        &mut self, 
        ptr: *mut OpaqueTensor, 
        num_elements: usize, 
        dtype_id: u8, 
        device_id: u8
    ) -> PoolOutcome

    /// プール内のすべてのTensorをクリア（実際にメモリ解放）します
    pub fn clear(&mut self)
    
    /// プールされているTensorの総数を取得します
    pub fn total_count(&self) -> usize
}
```
