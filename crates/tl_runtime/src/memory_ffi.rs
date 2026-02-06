//! Memory 関連の FFI 関数

use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, Ordering};

static SCOPE_DEPTH: AtomicI64 = AtomicI64::new(0);

/// スコープ開始
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_enter_scope() {
    SCOPE_DEPTH.fetch_add(1, Ordering::SeqCst);
}

/// スコープ終了
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_exit_scope() {
    SCOPE_DEPTH.fetch_sub(1, Ordering::SeqCst);
}

/// スコープ深度を取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_scope_depth() -> i64 {
    SCOPE_DEPTH.load(Ordering::SeqCst)
}

/// プールカウントを取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_pool_count() -> i64 {
    0
}

/// 参照カウント数を取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_refcount_count() -> i64 {
    0
}

/// 関数スコープ開始
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_enter() {
    tl_mem_enter_scope();
}

/// 関数スコープ終了
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_exit() {
    tl_mem_exit_scope();
}

/// バッファ取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_get_buffer(_index: i64) -> *mut c_void {
    std::ptr::null_mut()
}

/// メモリ解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// ログアロケーション（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_log_alloc(_ptr: *const c_void, _size: i64, _file: *const std::os::raw::c_char, _line: u32) {
    // スタブ
}

/// ログ解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_log_free(_ptr: *mut c_void, _file: *const std::os::raw::c_char, _line: u32) {
    // スタブ
}

/// テンソル登録（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_tensor(_ptr: *mut c_void) {
    // スタブ
}

/// 構造体登録（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct(_ptr: *mut c_void) {
    // スタブ
}

/// 名前付き構造体登録（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct_named(_ptr: *mut c_void, _name: *const std::os::raw::c_char) {
    // スタブ
}

/// 登録解除（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_unregister(_ptr: *mut c_void) {
    // スタブ
}

/// テンソル取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_acquire(t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    t // そのまま返す
}

/// テンソル解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_release(t: *mut crate::OpaqueTensor) {
    if !t.is_null() {
        unsafe {
            let _ = Box::from_raw(t);
        }
    }
}

/// 参照カウント増加（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_inc_ref(_ptr: *mut c_void) {
    // スタブ
}

/// 参照カウント減少（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_dec_ref(_ptr: *mut c_void) {
    // スタブ
}

/// ポインタ取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_acquire(ptr: *mut c_void) -> *mut c_void {
    ptr
}

/// ポインタ解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_release(_ptr: *mut c_void) {
    // スタブ
}

/// 一時バッファ割り当て
#[unsafe(no_mangle)]
pub extern "C" fn tl_alloc_tmp(size: i64) -> *mut c_void {
    if size <= 0 {
        return std::ptr::null_mut();
    }
    unsafe { libc::malloc(size as usize) }
}

/// 一時バッファ解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_free_tmp(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr); }
    }
}

/// メモリトレース（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_trace_mem(_msg: *const std::os::raw::c_char) {
    // スタブ
}

/// Metal プール MB 取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_mb() -> i64 {
    // スタブ実装
    0
}

/// Metal プールカウント取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_count() -> i64 {
    // スタブ実装
    0
}

/// Metal プールバイト取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_bytes() -> i64 {
    // スタブ実装
    0
}

/// テンソル返却準備（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_prepare_return(t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    t // そのまま返す
}

/// 登録テンソル（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_register_tensor(_t: *mut crate::OpaqueTensor) {
    // スタブ
}
