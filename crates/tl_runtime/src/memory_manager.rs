//! memory_manager スタブモジュール
//!
//! memory_ffi からの re-export でコンパイラの互換性を維持。

use std::ffi::c_void;

// memory_ffi からの re-export
pub use crate::memory_ffi::{
    tl_mem_unregister,
    tl_mem_enter_scope,
    tl_mem_exit_scope,
    tl_mem_function_enter,
    tl_mem_function_exit,
    tl_mem_get_buffer,
    tl_get_pool_count,
    tl_get_refcount_count,
    tl_get_scope_depth,
    tl_get_metal_pool_mb,
    tl_get_metal_pool_count,
    tl_get_metal_pool_bytes,
    tl_mem_register_struct,
    tl_mem_register_struct_named,
    tl_mem_register_tensor,
    tl_tensor_acquire,
    tl_tensor_release_safe,
    tl_tensor_finalize,
    // ptr 関数も memory_ffi から re-export
    tl_ptr_acquire,
    tl_ptr_release,
    tl_ptr_inc_ref,
    tl_ptr_dec_ref,
};

/// HashMap 用のメモリプール（スタブ）
pub struct HashMapPool;

/// テンソルプール（スタブ）
pub struct TensorPool;

/// スコープマネージャ（スタブ）
pub struct ScopeManager;

/// メモリマネージャ初期化（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_mem_manager_init() {
    // スタブ
}

/// メモリマネージャシャットダウン（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_manager_shutdown() {
    // スタブ
}

/// メモリ割り当て（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig (usize) -> void*
pub extern "C" fn tl_mem_alloc(size: usize) -> *mut c_void {
    if size == 0 {
        return std::ptr::null_mut();
    }
    unsafe { libc::malloc(size) }
}

/// メモリ解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_dealloc(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { libc::free(ptr); }
    }
}

/// メモリ統計取得（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> i64
pub extern "C" fn tl_mem_stats_get_allocated() -> i64 {
    0
}

/// メモリ統計リセット（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_stats_reset() {
    // スタブ
}
