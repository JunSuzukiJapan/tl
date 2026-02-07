//! TL Runtime - Metal Backend
//!
//! Candle を使用せず、tl_metal::MetalTensor を直接使用するランタイム。
//! 機能別にモジュールを分割して管理。

// ========== Core Modules ==========
pub mod arena;
pub mod args;
pub mod error;

// ========== FFI Modules ==========
pub mod string_ffi;
pub mod print_ffi;
pub mod memory_ffi;
pub mod file_io;
pub mod tensor_map;
pub mod tokenizer;
pub mod system;
pub mod tensor_ops_ext;
pub mod math_ffi;

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
pub use tensor_ops_ext::{
    // Type conversion
    tl_tensor_to_f32, tl_tensor_to_i64, tl_tensor_to_device,
    // Activation functions
    tl_tensor_tan, tl_tensor_sigmoid, tl_tensor_gelu, tl_tensor_silu,
    // Reduction with dimension
    tl_tensor_max_dim, tl_tensor_min_dim, tl_tensor_mean_dim, tl_tensor_sum_dim,
    // Convolution
    tl_tensor_conv2d,
    // NN Layers (GPU accelerated)
    tl_tensor_batch_norm, tl_tensor_layer_norm,
    tl_tensor_max_pool2d, tl_tensor_avg_pool2d, tl_tensor_dropout,
    // Embedding
    tl_tensor_embedding, tl_tensor_cross_entropy,
    // Normalization
    tl_tensor_rms_norm,
    // Misc
    tl_tensor_repeat_interleave, tl_tensor_sample, tl_tensor_scale, tl_tensor_clamp,
    // Device/Grad
    tl_tensor_device_id, tl_tensor_backward, tl_tensor_grad, tl_tensor_detach, tl_tensor_enable_grad,
    // RoPE
    tl_tensor_rope_new_cos, tl_tensor_rope_new_sin, tl_tensor_apply_rope,
    // Mask
    tl_tensor_new_causal_mask,
    // 追加テンソル関数
    tl_tensor_sqrt, tl_tensor_pow, tl_tensor_tril,
    tl_tensor_get, tl_tensor_item,
    tl_tensor_get_f32_md, tl_tensor_get_i64_md, tl_tensor_set_f32_md,
    tl_tensor_from_vec_u8, tl_tensor_from_u8_labels, tl_tensor_from_i64_array,
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
    tl_gguf_load, tl_qtensor_free, tl_qtensor_matmul,
    tl_metal_sync, tl_report_runtime_error, tl_handle_runtime_error,
};

// ========== Memory Exports ==========
pub use memory_ffi::{
    tl_tensor_acquire, tl_tensor_release, tl_tensor_prepare_return,
    tl_trace_mem, tl_register_tensor,
};

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

use std::ffi::c_void;
use std::sync::OnceLock;

// Metal バックエンド
use tl_metal::{MetalTensor, DType};

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

/// テンソルをヒープに配置して返す
fn make_tensor(t: MetalTensor) -> *mut OpaqueTensor {
    Box::into_raw(Box::new(t))
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

// ========== テンソル作成 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new(
    data: *const f32,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    
    let tensor = MetalTensor::from_slice(data_slice, shape_slice, DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new_i64(
    data: *const i64,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    
    let tensor = MetalTensor::from_slice(&f32_data, shape_slice, DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_zeros(rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::zeros(shape_slice, DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_ones(rank: usize, shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::ones(shape_slice, DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_randn(rank: usize, shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::randn(shape_slice, DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_randn_debug(
    rank: usize, 
    shape: *const usize, 
    _seed: u64,
    _req_grad: bool,
) -> *mut OpaqueTensor {
    tl_tensor_randn(rank, shape, _req_grad)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_i64(data: *const i64, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let tensor = MetalTensor::from_slice(&f32_data, &[len], DType::F32);
    make_tensor(tensor)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_u8(data: *const u8, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let tensor = MetalTensor::from_slice(&f32_data, &[len], DType::F32);
    make_tensor(tensor)
}

// ========== テンソル解放 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        unsafe {
            let _ = Box::from_raw(t);
        }
    }
}

// ========== テンソル情報取得 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_dim(t: *mut OpaqueTensor, dim: usize) -> usize {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    tensor.shape().get(dim).cloned().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_len(t: *mut OpaqueTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    tensor.shape().iter().product()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_shape(t: *mut OpaqueTensor, out: *mut usize) -> usize {
    if t.is_null() || out.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    for (i, &dim) in shape.iter().enumerate() {
        unsafe { *out.add(i) = dim; }
    }
    shape.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_shape(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let shape: Vec<f32> = tensor.shape().iter().map(|&d| d as f32).collect();
    let result = MetalTensor::from_slice(&shape, &[shape.len()], DType::F32);
    make_tensor(result)
}

// ========== テンソル要素アクセス ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_f32(t: *mut OpaqueTensor, indices: *const usize, _rank: usize) -> f32 {
    if t.is_null() || indices.is_null() {
        return 0.0;
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let shape = tensor.shape();
    
    // フラットインデックスを計算
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, shape.len()) };
    let mut flat_idx = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat_idx += idx_slice[i] * stride;
        stride *= shape[i];
    }
    
    data.get(flat_idx).cloned().unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_i64(t: *mut OpaqueTensor, indices: *const usize, rank: usize) -> i64 {
    tl_tensor_get_f32(t, indices, rank) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_set_f32(
    t: *mut OpaqueTensor,
    indices: *const usize,
    _rank: usize,
    value: f32,
) {
    if t.is_null() || indices.is_null() {
        return;
    }
    let tensor = unsafe { &mut *t };
    let mut data: Vec<f32> = tensor.to_vec();
    let shape = tensor.shape().to_vec();
    
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, shape.len()) };
    let mut flat_idx = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat_idx += idx_slice[i] * stride;
        stride *= shape[i];
    }
    
    if flat_idx < data.len() {
        data[flat_idx] = value;
        *tensor = MetalTensor::from_slice(&data, &shape, DType::F32);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_item_i64(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    data.first().map(|&f| f as i64).unwrap_or(0)
}

// ========== 基本テンソル演算 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::add_impl(ta, tb);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::sub_impl(ta, tb);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::mul_impl(ta, tb);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::div_impl(ta, tb);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rem(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.rem_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::neg_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_abs(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    make_tensor(tensor.abs_impl())
}

// ========== スカラー演算 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::add_scalar_impl(tensor, s as f32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::mul_scalar_impl(tensor, s as f32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::div_scalar_impl(tensor, s as f32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_pow_scalar(t: *mut OpaqueTensor, exp: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let result_data: Vec<f32> = data.iter().map(|&x| x.powf(exp as f32)).collect();
    let result = MetalTensor::from_slice(&result_data, tensor.shape(), DType::F32);
    make_tensor(result)
}

// ========== インプレース演算 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() {
        return;
    }
    unsafe {
        let result = MetalTensor::add_impl(&*a, &*b);
        *a = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() {
        return;
    }
    unsafe {
        let result = MetalTensor::sub_impl(&*a, &*b);
        *a = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() {
        return;
    }
    unsafe {
        let result = MetalTensor::mul_impl(&*a, &*b);
        *a = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() {
        return;
    }
    unsafe {
        let result = MetalTensor::div_impl(&*a, &*b);
        *a = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mod_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() {
        return;
    }
    unsafe {
        let result = (*a).rem_impl(&*b);
        *a = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let result = MetalTensor::add_scalar_impl(&*t, s);
        *t = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let result = MetalTensor::add_scalar_impl(&*t, -s);
        *t = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let result = MetalTensor::mul_scalar_impl(&*t, s);
        *t = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let result = MetalTensor::div_scalar_impl(&*t, s);
        *t = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mod_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let data: Vec<f32> = (*t).to_vec();
        let result_data: Vec<f32> = data.iter().map(|&x| x % s).collect();
        *t = MetalTensor::from_slice(&result_data, (*t).shape(), DType::F32);
    }
}

// ========== 比較演算 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_eq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.eq_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_neq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.ne_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_lt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.lt_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_le(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.le_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_gt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.gt_impl(tb))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_ge(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.ge_impl(tb))
}

// ========== 数学関数 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::exp_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::log_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::sin_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::cos_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tanh(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let result_data: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
    let result = MetalTensor::from_slice(&result_data, tensor.shape(), DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::relu_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::softmax_impl(tensor, dim as i32);
    make_tensor(result)
}

// ========== Reduction ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated sumall
    let sum_val = MetalTensor::sumall_impl(tensor);
    let result = MetalTensor::from_slice(&[sum_val], &[1], DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated mean_all
    let mean_val = MetalTensor::mean_all_impl(tensor);
    let result = MetalTensor::from_slice(&[mean_val], &[1], DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let result = MetalTensor::from_slice(&[max_val], &[1], DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let result = MetalTensor::from_slice(&[min_val], &[1], DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmax(t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated argmax
    let max_idx = MetalTensor::argmax_all_impl(tensor);
    let result = MetalTensor::from_slice(&[max_idx as f32], &[1], DType::F32);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmin(t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated argmin
    let min_idx = MetalTensor::argmin_all_impl(tensor);
    let result = MetalTensor::from_slice(&[min_idx as f32], &[1], DType::F32);
    make_tensor(result)
}

// ========== Shape 操作 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape(
    t: *mut OpaqueTensor,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    if t.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let new_shape = unsafe { std::slice::from_raw_parts(shape, rank) };
    let result = MetalTensor::reshape_impl(tensor, new_shape);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_new(
    t: *mut OpaqueTensor,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    tl_tensor_reshape(t, rank, shape)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_dims(
    t: *mut OpaqueTensor,
    dim1: i64,
    dim2: i64,
    dim3: i64,
    dim4: i64,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let mut new_shape = Vec::new();
    if dim1 > 0 { new_shape.push(dim1 as usize); }
    if dim2 > 0 { new_shape.push(dim2 as usize); }
    if dim3 > 0 { new_shape.push(dim3 as usize); }
    if dim4 > 0 { new_shape.push(dim4 as usize); }
    
    let result = MetalTensor::reshape_impl(tensor, &new_shape);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_transpose(t: *mut OpaqueTensor, dim0: usize, dim1: usize) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::transpose_impl(tensor, dim0, dim1);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_squeeze(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::squeeze_impl(tensor, dim as usize);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_unsqueeze(t: *mut OpaqueTensor, dim: usize) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::unsqueeze_impl(tensor, dim);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::contiguous_impl(tensor);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_narrow(
    t: *mut OpaqueTensor,
    dim: usize,
    start: usize,
    len: usize,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::narrow_impl(tensor, dim, start, len);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_slice(
    t: *mut OpaqueTensor,
    dim: usize,
    start: usize,
    end: usize,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let len = end.saturating_sub(start);
    let result = MetalTensor::narrow_impl(tensor, dim, start, len);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    // GPU accelerated cat
    let result = MetalTensor::cat(&[ta, tb], dim as usize);
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_i64(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_tensor_cat(a, b, dim)
}

// ========== Clone/Print ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let cloned = tensor.clone();
    make_tensor(cloned)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print(t: *mut OpaqueTensor) {
    if t.is_null() {
        println!("Tensor[null]");
        return;
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let shape = tensor.shape();
    println!("Tensor(shape={:?}, data={:?})", shape, data);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_replace_data(dst: *mut OpaqueTensor, src: *mut OpaqueTensor) {
    if dst.is_null() || src.is_null() {
        return;
    }
    unsafe {
        *dst = (*src).clone();
    }
}

// ========== MatMul ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_matmul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::matmul_impl(ta, tb);
    make_tensor(result)
}

// ========== I/O ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_save(t: *mut OpaqueTensor, path: *mut StringStruct) {
    unsafe {
        if t.is_null() || path.is_null() || (*path).ptr.is_null() {
            return;
        }
        let tensor = &*t;
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = file_io::expand_path(&path_str);
        
        // 簡易実装: バイナリ形式で保存
        let data: Vec<f32> = tensor.to_vec();
        let shape = tensor.shape().to_vec();
        
        // バイナリフォーマット: [rank: u64][shape: u64 * rank][data: f32 * numel]
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for &dim in &shape {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for &val in &data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        
        match std::fs::write(&path_buf, &bytes) {
            Ok(_) => println!("Saved tensor to {:?}", path_buf),
            Err(e) => eprintln!("Failed to save tensor: {}", e),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_load(path: *mut StringStruct) -> *mut OpaqueTensor {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = file_io::expand_path(&path_str);
        
        match std::fs::read(&path_buf) {
            Ok(bytes) => {
                if bytes.len() < 8 {
                    eprintln!("Invalid tensor file: too short");
                    return std::ptr::null_mut();
                }
                
                // バイナリフォーマット: [rank: u64][shape: u64 * rank][data: f32 * numel]
                let rank = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
                let mut offset = 8;
                
                let mut shape = Vec::with_capacity(rank);
                for _ in 0..rank {
                    if offset + 8 > bytes.len() {
                        eprintln!("Invalid tensor file: truncated shape");
                        return std::ptr::null_mut();
                    }
                    let dim = u64::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3], bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]]) as usize;
                    shape.push(dim);
                    offset += 8;
                }
                
                let numel: usize = shape.iter().product();
                let mut data = Vec::with_capacity(numel);
                for _ in 0..numel {
                    if offset + 4 > bytes.len() {
                        eprintln!("Invalid tensor file: truncated data");
                        return std::ptr::null_mut();
                    }
                    let val = f32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
                    data.push(val);
                    offset += 4;
                }
                
                let tensor = MetalTensor::from_slice(&data, &shape, DType::F32);
                println!("Loaded tensor from {:?}", path_buf);
                make_tensor(tensor)
            }
            Err(e) => {
                eprintln!("Failed to read file: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

// ========== Memory Size ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_mb() -> i64 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(kb) = s.trim().parse::<i64>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

// ========== Compatibility Re-exports ==========
// tl_print_f64, tl_print_i64, tl_display_f64, tl_display_i64 は print_ffi モジュールで定義

// Vec helper functions for compatibility
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_void_len(ptr: *mut c_void) -> usize {
    if ptr.is_null() { return 0; }
    unsafe {
        let vec = &*(ptr as *mut Vec<*mut c_void>);
        vec.len()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_void_get(ptr: *mut c_void, idx: usize) -> *mut c_void {
    if ptr.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let vec = &*(ptr as *mut Vec<*mut c_void>);
        vec.get(idx).cloned().unwrap_or(std::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_void_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut Vec<*mut c_void>);
        }
    }
}

// Vec<u8> support
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_new() -> *mut Vec<u8> {
    Box::into_raw(Box::new(Vec::<u8>::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_len(ptr: *mut Vec<u8>) -> usize {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_get(ptr: *mut Vec<u8>, idx: usize) -> u8 {
    if ptr.is_null() { return 0; }
    unsafe {
        let vec = &*ptr;
        vec.get(idx).cloned().unwrap_or(0)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_push(ptr: *mut Vec<u8>, val: u8) {
    if !ptr.is_null() {
        unsafe { (*ptr).push(val); }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_free(ptr: *mut Vec<u8>) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}
