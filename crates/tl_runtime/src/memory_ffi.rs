//! Memory 関連の FFI 関数

use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;
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
    tl_cpu::ffi::tl_cpu_enter_scope();
}

/// 関数スコープ終了
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_exit() {
    tl_cpu::ffi::tl_cpu_exit_scope();
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

/// テンソル取得（RC 操作なし — テンソルは所有権ベースで管理）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_acquire(t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    t
}

/// テンソル解放（条件付きデータクリア）
/// 構造体自体は残すので二重呼び出しでも安全。
/// autograd を持つテンソルは backward + detach クリーンアップに委ねる。
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_release_safe(t: *mut crate::OpaqueTensor) {
    if t.is_null() { return; }
    // cpu-backend-release fix: Enforce return to pool for ALL tensors on CPU.
    // Casting MetalTensor to CpuTensor is unsafe but necessary if we are running CPU backend logic via tl_runtime symbols.
    // Ideally we should detect backend type, but for now we assume CPU if this path is hit in N-Queens.
    let cpu_tensor = t as *mut tl_cpu::tensor::CpuTensor;
    
    tl_cpu::ffi::tl_cpu_tensor_return_to_pool(cpu_tensor);
}

/// テンソルファイナライズ（No-op: exit_scope で一括処理）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_finalize(_t: *mut crate::OpaqueTensor) {
    // No-op
}


/// グローバル参照カウントマップ（構造体用）
static REF_COUNTS: std::sync::LazyLock<Mutex<HashMap<usize, usize>>> = 
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// 参照カウント増加
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_inc_ref(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    let key = ptr as usize;
    if let Ok(mut counts) = REF_COUNTS.lock() {
        *counts.entry(key).or_insert(1) += 1;
    }
}

/// 参照カウント減少
/// 戻り値: カウントが0になった場合は true (解放すべき)
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
    }
    false
}

/// ポインタ取得（参照カウント増加）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_acquire(ptr: *mut c_void) -> *mut c_void {
    if ptr.is_null() { return ptr; }
    tl_ptr_inc_ref(ptr);
    ptr
}

/// ポインタ解放（参照カウント減少、0になったら解放）
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_release(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    if tl_ptr_dec_ref(ptr) {
        unsafe { libc::free(ptr); }
    }
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

/// テンソル昇格（スコープから除外）（V4.5）
/// Metal: No-op (Persistent Pool)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_promote(_t: *mut crate::OpaqueTensor) {
    // No-op
}

/// テンソル登録（スコープに追加）（V4.5）
/// Metal: No-op
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_register(_t: *mut crate::OpaqueTensor) {
    // No-op
}
