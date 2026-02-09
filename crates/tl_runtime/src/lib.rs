#![allow(clippy::not_unsafe_ptr_arg_deref)]
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
pub mod system;
pub mod tensor_ops_ext;
pub mod math_ffi;
pub mod tokenizer;
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
    tl_tensor_tan, tl_tensor_sigmoid, tl_tensor_gelu, tl_tensor_silu, tl_tensor_tanh,
    tl_tensor_exp, tl_tensor_log, tl_tensor_sin, tl_tensor_cos, tl_tensor_relu,
    tl_tensor_softmax,
    // Reduction with dimension
    tl_tensor_max_dim, tl_tensor_min_dim, tl_tensor_mean_dim, tl_tensor_sum_dim,
    // Global reduction and argmin/max
    tl_tensor_max, tl_tensor_min, tl_tensor_mean, tl_tensor_sum, tl_tensor_argmin,
    // Matrix Ops
    tl_tensor_matmul, tl_tensor_transpose,
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
    tl_tensor_replace_data, tl_tensor_contiguous, tl_tensor_print, tl_tensor_slice,
    tl_tensor_reshape, // Added reshape
    // Device/Grad
    tl_tensor_device_id, tl_tensor_backward, tl_tensor_grad, tl_tensor_detach, tl_tensor_enable_grad,
    // Rope
    tl_tensor_rope_new_cos, tl_tensor_rope_new_sin, tl_tensor_apply_rope,
    // Mask
    tl_tensor_new_causal_mask,
    // 追加テンソル関数
    tl_tensor_sqrt, tl_tensor_pow, tl_tensor_sub_scalar, tl_tensor_tril,
    tl_tensor_cat, // Added export
    tl_tensor_get, tl_tensor_item,
    tl_tensor_get_f32_md, tl_tensor_get_i64_md, tl_tensor_set_f32_md,
    tl_tensor_from_vec_u8, tl_tensor_from_u8_labels, tl_tensor_from_i64_array,
    // Legacy / Image / Compiler Compat / IO
    tl_tensor_cat_i64, tl_tensor_narrow, tl_tensor_reshape_dims,
    tl_tensor_argmax, tl_tensor_reshape_new,
    tl_image_load_grayscale, tl_image_width, tl_image_height,
    tl_tensor_save, tl_tensor_load, tl_get_memory_mb,
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
pub extern "C" fn tl_tensor_zeros(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::zeros(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_ones(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::ones(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::randn(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
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
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use tl_metal::autograd::ops::AddBackward;
        let t = unsafe { &mut *ptr };
        t.set_grad_fn(Box::new(AddBackward { a, b }));
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::sub_impl(ta, tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use tl_metal::autograd::ops::SubBackward;
        let t = unsafe { &mut *ptr };
        t.set_grad_fn(Box::new(SubBackward { a, b }));
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::mul_impl(ta, tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use tl_metal::autograd::ops::MulBackward;
        let t = unsafe { &mut *ptr };
        t.set_grad_fn(Box::new(MulBackward {
            a, b,
            a_data: ta.shallow_clone(),
            b_data: tb.shallow_clone(),
        }));
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = MetalTensor::div_impl(ta, tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use tl_metal::autograd::ops::DivBackward;
        let t = unsafe { &mut *ptr };
        t.set_grad_fn(Box::new(DivBackward {
            a, b,
            a_data: ta.shallow_clone(),
            b_data: tb.shallow_clone(),
        }));
    }
    ptr
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
pub extern "C" fn tl_tensor_pow_scalar(t: *mut OpaqueTensor, exp: f32) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.pow_scalar_impl(exp);
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use tl_metal::autograd::ops::PowBackward;
        let exp_tensor = MetalTensor::from_slice(&[exp], &[1], DType::F32);
        let result_clone = unsafe { &*ptr }.shallow_clone();
        let rt = unsafe { &mut *ptr };
        rt.set_grad_fn(Box::new(PowBackward {
            a: t,
            a_data: tensor.shallow_clone(),
            b_data: exp_tensor,
            output: result_clone,
        }));
    }
    ptr
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
        let result = (*t).fmod_scalar_impl(s);
        *t = result;
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
