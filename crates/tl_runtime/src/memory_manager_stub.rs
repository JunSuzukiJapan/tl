//! Memory Manager Stub
//!
//! Candle 版 memory_manager の最小スタブ実装

use std::ffi::c_void;

/// スコープ深度を取得（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> i64
pub extern "C" fn tl_get_scope_depth() -> i64 {
    0
}

/// プールカウントを取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_pool_count() -> i64 {
    0
}

/// 参照カウント数を取得（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> i64
pub extern "C" fn tl_get_refcount_count() -> i64 {
    0
}

/// 関数スコープ開始（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_enter() {
    // スタブ - 何もしない
}

/// 関数スコープ終了（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_mem_function_exit() {
    // スタブ - 何もしない
}

/// バッファ取得（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_get_buffer(_index: i64) -> *mut c_void {
    std::ptr::null_mut()
}

/// テンソルメタ情報の登録（ダミー）
pub fn register_tensor_meta_global(_ptr: *mut crate::OpaqueTensor, _meta: AllocationMeta) {
    // スタブ
}

/// アロケーションメタ情報
pub struct AllocationMeta {
    pub ctx: String,
    pub bytes: usize,
    pub dtype: String,
    pub elems: usize,
    pub shape: String,
    pub device: String,
    pub loc_file: String,
    pub loc_line: u32,
    pub pooled: bool,
}
