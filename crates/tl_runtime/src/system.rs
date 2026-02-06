//! System 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::io::Write;

/// スリープ（ミリ秒）
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_sleep(ms: i64) {
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }
}

/// 現在時刻（Unix タイムスタンプ、秒）
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 標準入力から行を読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_read_line() -> *mut StringStruct {
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(_) => {
            let trimmed = input.trim_end();
            let c_str = CString::new(trimmed).unwrap_or_else(|_| CString::new("").unwrap());
            let ptr = c_str.into_raw();
            unsafe {
                let len = libc::strlen(ptr) as i64;
                let layout = std::alloc::Layout::new::<StringStruct>();
                let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
                (*struct_ptr).ptr = ptr;
                (*struct_ptr).len = len;
                struct_ptr
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// プロンプト表示して入力を読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_prompt(prompt: *const c_char) -> *mut StringStruct {
    if !prompt.is_null() {
        let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy() };
        print!("{}", prompt_str);
        let _ = std::io::stdout().flush();
    }
    tl_read_line()
}

/// デバイス設定（Metal のみなのでスタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_set_device(_device_id: i64) {
    // Metal バックエンドでは単一デバイスのため何もしない
}

/// VarBuilder 関連（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get(_name: *mut StringStruct) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder not yet supported in Metal backend");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get_from_tensor(_tensor: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder not yet supported in Metal backend");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_grad(_name: *mut StringStruct) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder gradients not yet supported in Metal backend");
    std::ptr::null_mut()
}

/// パラメータ関連（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_add_parameter(_name: *mut StringStruct, _t: *mut crate::OpaqueTensor) {
    // スタブ
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_register_parameter(_t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    // スタブ - そのまま返す
    _t
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_save_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter saving not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_load_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter loading not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_update_all_params(_lr: f64) {
    eprintln!("Warning: Parameter update not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_clear_grads() {
    eprintln!("Warning: Gradient clearing not yet supported in Metal backend");
}

/// GGUF ロード（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_gguf_load(_path: *mut StringStruct) -> *mut crate::tensor_map::OpaqueTensorMap {
    eprintln!("Warning: GGUF loading not yet supported in Metal backend");
    std::ptr::null_mut()
}

/// QTensor 関連（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_free(_ptr: usize) {
    // スタブ
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_matmul(_input: *mut crate::OpaqueTensor, _weight: usize) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: Quantized matmul not yet supported in Metal backend");
    std::ptr::null_mut()
}

/// KV Cache 関連（スタブ）
pub struct OpaqueKVCache {
    pub k: Option<*mut crate::OpaqueTensor>,
    pub v: Option<*mut crate::OpaqueTensor>,
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_new() -> *mut OpaqueKVCache {
    Box::into_raw(Box::new(OpaqueKVCache { k: None, v: None }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_free(cache: *mut OpaqueKVCache) {
    if !cache.is_null() {
        unsafe {
            let _ = Box::from_raw(cache);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_k(cache: *mut OpaqueKVCache) -> *mut crate::OpaqueTensor {
    unsafe {
        if cache.is_null() {
            return std::ptr::null_mut();
        }
        (*cache).k.unwrap_or(std::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_v(cache: *mut OpaqueKVCache) -> *mut crate::OpaqueTensor {
    unsafe {
        if cache.is_null() {
            return std::ptr::null_mut();
        }
        (*cache).v.unwrap_or(std::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_update(
    cache: *mut OpaqueKVCache,
    k: *mut crate::OpaqueTensor,
    v: *mut crate::OpaqueTensor,
) {
    unsafe {
        if !cache.is_null() {
            (*cache).k = Some(k);
            (*cache).v = Some(v);
        }
    }
}

// ========== 追加 System 関数 ==========

// tl_checkpoint と tl_trace_mem は memory_ffi.rs で定義済み

// tl_hash_string は string_ffi.rs で定義済み

// tl_http_get と tl_http_download は file_io.rs で定義済み

// tl_download_file は file_io.rs で定義済み

/// Metal 同期
#[unsafe(no_mangle)]
pub extern "C" fn tl_metal_sync() {
    // Metal バックエンドの同期（現在は何もしない）
}

// tl_register_tensor は memory_ffi.rs で定義済み

/// ランタイムエラー報告
#[unsafe(no_mangle)]
pub extern "C" fn tl_report_runtime_error(msg: *mut StringStruct) {
    if !msg.is_null() {
        unsafe {
            if !(*msg).ptr.is_null() {
                let c_str = CStr::from_ptr((*msg).ptr);
                eprintln!("Runtime error: {}", c_str.to_string_lossy());
            }
        }
    }
}

/// ランタイムエラーハンドル
#[unsafe(no_mangle)]
pub extern "C" fn tl_handle_runtime_error(msg: *mut StringStruct) {
    tl_report_runtime_error(msg);
}

// tl_log_alloc と tl_log_free は memory_ffi.rs で定義済み

// tl_query は knowledge_base.rs で定義済み
