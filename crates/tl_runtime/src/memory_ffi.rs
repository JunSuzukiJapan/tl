//! Memory 関連の FFI 関数

use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

static SCOPE_DEPTH: AtomicI64 = AtomicI64::new(0);

/// @ffi_sig () -> void
/// スコープ開始
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_enter_scope() {
    SCOPE_DEPTH.fetch_add(1, Ordering::SeqCst);
}

/// @ffi_sig () -> void
/// スコープ終了
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_exit_scope() {
    SCOPE_DEPTH.fetch_sub(1, Ordering::SeqCst);
}

/// @ffi_sig () -> i64
/// スコープ深度を取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_scope_depth() -> i64 {
    SCOPE_DEPTH.load(Ordering::SeqCst)
}

/// @ffi_sig () -> i64
/// プールカウントを取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_pool_count() -> i64 {
    0
}

/// @ffi_sig () -> i64
/// 参照カウント数を取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_refcount_count() -> i64 {
    0
}

/// @ffi_sig () -> void
/// 関数スコープ開始
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_enter() {
    tl_mem_enter_scope();
    tl_cpu::ffi::tl_cpu_enter_scope();
}

/// @ffi_sig () -> void
/// 関数スコープ終了
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_exit() {
    tl_cpu::ffi::tl_cpu_exit_scope();
    tl_mem_exit_scope();
}

/// @ffi_sig (i64) -> void*
/// バッファ取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_get_buffer(_index: i64) -> *mut c_void {
    std::ptr::null_mut()
}

/// @ffi_sig (Struct*) -> void
/// 構造体メモリ解放。codegen の malloc と対になる libc::free を使用。
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            libc::free(ptr);
        }
    }
}

/// @ffi_sig (void*, i64, i8*, u32) -> void
/// ログアロケーション（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_log_alloc(_ptr: *const c_void, _size: i64, _file: *const std::os::raw::c_char, _line: u32) {
    // スタブ
}

/// @ffi_sig (void*, i8*, u32) -> void
/// ログ解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_log_free(_ptr: *mut c_void, _file: *const std::os::raw::c_char, _line: u32) {
    // スタブ
}

/// @ffi_sig (Tensor*) -> void
/// テンソル登録（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_tensor(_ptr: *mut c_void) {
    // スタブ
}

/// @ffi_sig (Struct*|String*) -> void
/// 構造体登録。REF_COUNTS に RC=1 で初期化する。
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    let key = ptr as usize;
    if let Ok(mut counts) = REF_COUNTS.lock() {
        counts.insert(key, 1);
    }
}

/// @ffi_sig (Struct*, i8*) -> void
/// 名前付き構造体登録。REF_COUNTS に RC=1 で初期化する。
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct_named(ptr: *mut c_void, _name: *const std::os::raw::c_char) {
    if ptr.is_null() { return; }
    let key = ptr as usize;
    if let Ok(mut counts) = REF_COUNTS.lock() {
        counts.insert(key, 1);
    }
}

/// @ffi_sig (void*) -> void
/// 登録解除（スタブ）。Tensor* または Struct* を受け取る。
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_unregister(_ptr: *mut c_void) {
    // スタブ
}

/// @ffi_sig (Tensor*) -> Tensor*
/// テンソル取得（Arc RC +1）
/// FieldAccess 等で構造体フィールドのテンソルを参照する際に呼ばれる。
/// tl_tensor_release_safe と対になり、retain/release の対称性を保証する。
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_acquire(t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    if !t.is_null() {
        unsafe {
            std::sync::Arc::increment_strong_count(
                t as *const std::cell::UnsafeCell<crate::OpaqueTensor>
            );
        }
    }
    t
}

/// @ffi_sig (Tensor*) -> void
/// テンソル解放（Arc ベース）
/// Arc の参照カウントを -1 する。RC=0 になれば Tensor（autograd グラフ含む）が
/// 自然に Drop される。
/// CPU/Metal 共通: Arc::from_raw で復元し drop で RC-1。
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_release_safe(t: *mut crate::OpaqueTensor) {
    if t.is_null() { return; }
    unsafe {
        let arc = std::sync::Arc::from_raw(t as *const std::cell::UnsafeCell<crate::OpaqueTensor>);
        drop(arc);
    }
}

/// @ffi_sig (Tensor*) -> void
/// テンソルファイナライズ（No-op: exit_scope で一括処理）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_finalize(_t: *mut crate::OpaqueTensor) {
    // No-op
}


/// グローバル参照カウントマップ（構造体用）
static REF_COUNTS: std::sync::LazyLock<Mutex<HashMap<usize, usize>>> = 
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// @ffi_sig (Struct*|String*) -> void
/// 参照カウント増加。構造体・String どちらにも使われる。
/// 安全ガード: REF_COUNTS に未登録のポインタ（alloca 等のスタックアドレス）は無視する。
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_inc_ref(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    let key = ptr as usize;
    if let Ok(mut counts) = REF_COUNTS.lock() {
        // REF_COUNTS に登録済みのポインタのみ RC 操作する。
        // tl_mem_register_struct / tl_mem_register_struct_named で
        // 正式に登録されたヒープポインタのみが対象。
        // 未登録ポインタ（alloca スタックアドレス等）は安全に無視する。
        if let Some(entry) = counts.get_mut(&key) {
            *entry += 1;
        }
    }
}

/// @ffi_sig (Struct*|String*) -> bool
/// 参照カウント減少。構造体・String どちらにも使われる。
/// 戻り値: カウントが0になった場合は true (解放すべき)
/// 安全ガード: REF_COUNTS に未登録のポインタは false を返す（解放しない）。
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_dec_ref(ptr: *mut c_void) -> bool {
    if ptr.is_null() { return false; }
    let key = ptr as usize;
    if let Ok(mut counts) = REF_COUNTS.lock() {
        if let Some(count) = counts.get_mut(&key) {
            if *count > 0 {
                *count -= 1;
                if *count == 0 {
                    counts.remove(&key);
                    return true; // should free
                }
            }
        }
        // 未登録ポインタは安全に無視（false を返す）
    }
    false
}

/// @ffi_sig (Struct*|String*) -> Struct*|String*
/// ポインタ取得（参照カウント増加）。構造体・String どちらにも使われる。
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_acquire(ptr: *mut c_void) -> *mut c_void {
    if ptr.is_null() { return ptr; }
    tl_ptr_inc_ref(ptr);
    ptr
}

/// @ffi_sig (Struct*|String*) -> void
/// ポインタ解放（参照カウント減少、0になったら解放）。構造体・String どちらにも使われる。
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_release(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    if tl_ptr_dec_ref(ptr) {
        unsafe { libc::free(ptr); }
    }
}


/// @ffi_sig (i64) -> void*
/// 一時バッファ割り当て
#[unsafe(no_mangle)]
pub extern "C" fn tl_alloc_tmp(size: i64) -> *mut c_void {
    if size <= 0 {
        return std::ptr::null_mut();
    }
    unsafe { libc::malloc(size as usize) }
}

/// @ffi_sig (void*) -> void
/// 一時バッファ解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_free_tmp(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr); }
    }
}

/// @ffi_sig (i8*) -> void
/// メモリトレース（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_trace_mem(_msg: *const std::os::raw::c_char) {
    // スタブ
}

/// @ffi_sig () -> i64
/// Metal プール MB 取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_mb() -> i64 {
    // スタブ実装
    0
}

/// @ffi_sig () -> i64
/// Metal プールカウント取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_count() -> i64 {
    // スタブ実装
    0
}

/// @ffi_sig () -> i64
/// Metal プールバイト取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_bytes() -> i64 {
    // スタブ実装
    0
}

/// @ffi_sig (Tensor*) -> Tensor*
/// テンソル返却準備（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_prepare_return(t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    t // そのまま返す
}

/// @ffi_sig (Tensor*) -> void
/// 登録テンソル（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_register_tensor(_t: *mut crate::OpaqueTensor) {
    // スタブ
}

/// @ffi_sig (Tensor*) -> void
/// テンソル昇格（スコープから除外）（V4.5）
/// Metal: No-op (Persistent Pool)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_promote(_t: *mut crate::OpaqueTensor) {
    // No-op
}

/// @ffi_sig (Tensor*) -> void
/// テンソル登録（スコープに追加）（V4.5）
/// Metal: No-op
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_register(_t: *mut crate::OpaqueTensor) {
    // No-op
}
