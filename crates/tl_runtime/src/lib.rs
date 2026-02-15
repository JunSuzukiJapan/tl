#![allow(clippy::not_unsafe_ptr_arg_deref)]
//! TL Runtime - Metal Backend
//!
//! Candle を使用せず、tl_metal::MetalTensor を直接使用するランタイム。
//! 機能別にモジュールを分割して管理。

// ========== Core Modules ==========
pub mod arena;
pub mod args;
pub mod error;
pub use error::{tl_report_runtime_error_loc};
pub mod device_ffi;

// ========== FFI Modules ==========
pub mod string_ffi;
pub mod print_ffi;
pub mod memory_ffi;
pub mod file_io;
pub mod tensor_map;
pub mod system;
pub mod tensor_ops_ext;
pub mod math_ffi;
pub mod tokenizer;
pub mod quantized;
pub mod gguf;
// autograd_registry は MetalTensor 統合により廃止

// ========== Stub Modules (Legacy Compatibility) ==========
pub mod stdlib;
pub mod llm;
pub mod knowledge_base;
pub mod memory_manager;
pub mod registry;
pub mod logic;
pub mod checkpoint;
pub mod context;

// ========== Re-exports ==========
pub use string_ffi::StringStruct;
pub use tensor_map::OpaqueTensorMap;
pub use tokenizer::OpaqueTokenizer;
pub use system::OpaqueKVCache;
pub use file_io::PathStruct;

// ========== Memory Manager Exports ==========
pub use memory_ffi::{
    tl_get_pool_count, tl_get_refcount_count, tl_get_scope_depth, 
    tl_mem_function_enter, tl_mem_function_exit, tl_mem_get_buffer,
    tl_mem_enter_scope, tl_mem_exit_scope,
    tl_get_metal_pool_mb, tl_get_metal_pool_count, tl_get_metal_pool_bytes,
    tl_tensor_promote, tl_tensor_register,
};

// Metal FFI: ffi_ops の関数は device_impl.rs 経由で IDevice から呼ばれる
// 直接 re-export は不要
pub use tl_metal::ffi::{
    tl_metal_clone, tl_metal_data, tl_metal_numel, tl_metal_release,
};

// ========== TensorMap Exports ==========
pub use tensor_map::{
    tl_tensor_map_new, tl_tensor_map_free,
    tl_tensor_map_insert, tl_tensor_map_get,
    tl_tensor_map_load, tl_tensor_map_save,
    tl_tensor_map_get_quantized,
};

// ========== Print Exports ==========
pub use print_ffi::{
    tl_print_string, tl_display_string,
    tl_print_i64, tl_print_f64,
    tl_display_i64, tl_display_f64,
};

// ========== String Exports ==========
pub use string_ffi::{
    tl_string_new, tl_string_len, tl_string_concat,
    tl_string_eq, tl_string_contains,
    tl_string_from_int, tl_string_to_i64,
    tl_string_from_char, tl_string_char_at,
};

// ========== TensorOps Extended Exports ==========


// ========== Runtime IO / Legacy Exports ==========
pub use tensor_ops_ext::{
    tl_tensor_save, tl_tensor_load, tl_get_memory_bytes, tl_get_memory_mb,
    tl_image_load_grayscale, tl_image_width, tl_image_height,
};

// ========== Print Ext Exports ==========
pub use print_ffi::{
    tl_tensor_print_1, tl_tensor_print_2, tl_tensor_print_3,
    tl_tensor_display, tl_string_print,
    tl_print_i32, tl_print_f32, tl_display_i32, tl_display_f32,
    tl_print_char, tl_display_char,
};

// ========== System Exports ==========
pub use system::{
    tl_system_sleep, tl_system_time, tl_read_line, tl_prompt,
    tl_set_device, tl_varbuilder_get, tl_varbuilder_get_from_tensor, tl_varbuilder_grad,
    tl_add_parameter, tl_register_parameter, tl_save_all_params, tl_load_all_params,
    tl_update_all_params, tl_clear_grads,
    tl_qtensor_free, tl_qtensor_matmul,
    tl_metal_sync, tl_report_runtime_error, tl_handle_runtime_error,
};

// ========== Memory Exports ==========
pub use memory_ffi::{tl_ptr_acquire, tl_tensor_release_safe, tl_tensor_finalize, tl_tensor_prepare_return, tl_register_tensor, tl_trace_mem};

// ========== Checkpoint Exports ==========
pub use checkpoint::{
    tl_checkpoint_save, tl_checkpoint_load,
};

// ========== File Exports ==========
pub use file_io::{
    tl_read_file, tl_write_file, tl_file_exists,
};

// ========== Math FFI Exports ==========
pub use math_ffi::{
    // i64 math functions
    tl_i64_abs, tl_i64_pow, tl_i64_signum,
    tl_i64_div_euclid, tl_i64_rem_euclid,
    tl_i64_is_positive, tl_i64_is_negative,
    // i32 math functions
    tl_i32_abs, tl_i32_pow, tl_i32_signum,
    tl_i32_div_euclid, tl_i32_rem_euclid,
    tl_i32_is_positive, tl_i32_is_negative,
    // f64 math functions
    tl_f64_abs, tl_f64_acos, tl_f64_acosh, tl_f64_asin, tl_f64_asinh, tl_f64_atan, tl_f64_atan2, tl_f64_atanh,
    tl_f64_cbrt, tl_f64_ceil, tl_f64_copysign, tl_f64_cos, tl_f64_cosh, tl_f64_exp, tl_f64_exp2, tl_f64_exp_m1,
    tl_f64_floor, tl_f64_fract, tl_f64_hypot, tl_f64_ln, tl_f64_ln_1p, tl_f64_log, tl_f64_log10, tl_f64_log2,
    tl_f64_powf, tl_f64_powi, tl_f64_recip, tl_f64_round, tl_f64_signum, tl_f64_sin, tl_f64_sinh,
    tl_f64_sqrt, tl_f64_tan, tl_f64_tanh, tl_f64_to_degrees, tl_f64_to_radians, tl_f64_trunc,
    // f32 math functions
    tl_f32_abs, tl_f32_acos, tl_f32_acosh, tl_f32_asin, tl_f32_asinh, tl_f32_atan, tl_f32_atan2, tl_f32_atanh,
    tl_f32_cbrt, tl_f32_ceil, tl_f32_copysign, tl_f32_cos, tl_f32_cosh, tl_f32_exp, tl_f32_exp2, tl_f32_exp_m1,
    tl_f32_floor, tl_f32_fract, tl_f32_hypot, tl_f32_ln, tl_f32_ln_1p, tl_f32_log, tl_f32_log10, tl_f32_log2,
    tl_f32_powf, tl_f32_powi, tl_f32_recip, tl_f32_round, tl_f32_signum, tl_f32_sin, tl_f32_sinh,
    tl_f32_sqrt, tl_f32_tan, tl_f32_tanh, tl_f32_to_degrees, tl_f32_to_radians, tl_f32_trunc,
};

// ========== Print Additional Exports ==========
pub use print_ffi::{
    tl_print_bool, tl_print_ptr, tl_display_bool,
};

// ========== Memory Additional Exports ==========
pub use memory_ffi::{
    tl_log_alloc, tl_log_free,
    tl_mem_free, tl_free_tmp, tl_alloc_tmp,
};

// ========== File Additional Exports ==========
pub use file_io::{tl_file_exists_i64, tl_download_file};

/// ランタイムリンクを強制する関数
/// コンパイラがランタイムシンボルを確実にリンクするために呼び出す
#[inline(never)]
pub fn force_link() {
    // 何もしない - リンクを強制するだけ
}

// use std::ffi::c_void;
use std::sync::OnceLock;

// Metal バックエンド
use tl_metal::MetalTensor;

/// OpaqueTensor は MetalTensor のエイリアス
pub type OpaqueTensor = MetalTensor;

// ========== ユーティリティ ==========

#[allow(dead_code)]
pub(crate) fn mem_log_enabled() -> bool {
    std::env::var("TL_MEM_DEBUG").is_ok()
}

#[allow(dead_code)]
fn mem_trace_enabled() -> bool {
    static MEM_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
    *MEM_TRACE_ENABLED.get_or_init(|| std::env::var("TL_MEM_TRACE").is_ok())
}

/// MetalTensor を Arc<UnsafeCell> でポインタに変換するヘルパー
/// tl_metal_release (Arc::from_raw) と対をなす。
/// 重要: Box::into_raw(Box::new(MetalTensor)) を直接使ってはいけない。
/// Box と Arc はメモリレイアウトが異なるため、ヒープ破損を引き起こす。
pub fn make_metal_tensor(t: MetalTensor) -> *mut OpaqueTensor {
    let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(t));
    std::sync::Arc::into_raw(arc) as *mut OpaqueTensor
}


#[allow(dead_code)]
fn return_ptr_or_null(
    res: std::thread::Result<Result<*mut OpaqueTensor, crate::error::RuntimeError>>,
) -> *mut OpaqueTensor {
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            crate::error::set_last_error(e.to_string(), e.code());
            std::ptr::null_mut()
        }
        Err(_) => {
            crate::error::set_last_error(
                "Panic caught in runtime".to_string(),
                crate::error::RuntimeErrorCode::InternalError,
            );
            std::ptr::null_mut()
        }
    }
}

// ========== 初期化 ==========

/// ランタイム初期化
#[unsafe(no_mangle)]
pub extern "C" fn tl_runtime_init() {
    println!("Runtime device initialized: Metal");
}

/// ランタイムシャットダウン
#[unsafe(no_mangle)]
pub extern "C" fn tl_runtime_shutdown() {
    // Metal リソースのクリーンアップ
}

/// エラーハンドリング
#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn tl_amend_error_loc(
    code: u32,
    msg: String,
    file: Option<String>,
    line: u32,
    col: u32,
) {
    let code_enum: crate::error::RuntimeErrorCode = code.into();
    let loc_str = match file {
        Some(f) => format!("{}:{}:{}", f, line, col),
        None => format!("unknown:{}:{}", line, col),
    };
    eprintln!("\n[Runtime Error] Code: {:?} ({})", code_enum, code);
    eprintln!("  Message: {}", msg);
    eprintln!("  Location: {}", loc_str);
    std::process::exit(1);
}

// ========== GGUF Exports ==========
pub use gguf::tl_gguf_load;





#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    crate::print_ffi::tl_tensor_print_1(t);
}
