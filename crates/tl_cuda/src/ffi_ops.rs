//! CUDA Backend FFI Operations
//!
//! tl_metal/src/ffi_ops.rs と同等の実装（V5.0 Arc ベースメモリ管理）。

use crate::autograd::ops::*;
use crate::tensor::{CudaTensor, TensorRef};
use crate::DType;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tl_backend::BackendResult;

type OpaqueTensor = CudaTensor;

// === デバッグカウンタ ===
pub static MAKE_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static ACQUIRE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// デバッグ: カウンタをリセット
#[no_mangle]
pub extern "C" fn tl_cuda_debug_reset_counters() {
    MAKE_COUNT.swap(0, Ordering::SeqCst);
    RELEASE_COUNT.swap(0, Ordering::SeqCst);
    ACQUIRE_COUNT.swap(0, Ordering::SeqCst);
}

/// 内部ヘルパー: CudaTensor を Arc で包んでポインタを返す（V5.0 メモリ管理）
pub fn make_tensor(t: CudaTensor) -> *mut OpaqueTensor {
    MAKE_COUNT.fetch_add(1, Ordering::Relaxed);
    let arc = Arc::new(UnsafeCell::new(t));
    Arc::into_raw(arc) as *mut OpaqueTensor
}

/// Arc RC-1: raw pointer から Arc を復元し、drop で RC を減らす。
pub fn release_if_live(t: *mut OpaqueTensor) {
    if t.is_null() {
        return;
    }
    RELEASE_COUNT.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let _ = Arc::from_raw(t as *const UnsafeCell<CudaTensor>);
    }
}

/// Arc RC+1: raw pointer の参照カウントを 1 増やす。
pub fn acquire_tensor(t: *mut OpaqueTensor) {
    if t.is_null() {
        return;
    }
    ACQUIRE_COUNT.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<CudaTensor>);
        let cloned = arc.clone(); // RC+1
        let _ = Arc::into_raw(arc); // 元のポインタを維持
        std::mem::forget(cloned); // RC+1 を維持（drop しない）
    }
}

/// BackendResult を安全にポインタに変換するヘルパー
fn make_result(result: BackendResult<CudaTensor>) -> *mut OpaqueTensor {
    match result {
        Ok(t) => make_tensor(t),
        Err(e) => {
            eprintln!("CUDA Backend FFI Error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// raw ポインタから CudaTensor への不変参照取得
unsafe fn get<'a>(t: *mut OpaqueTensor) -> &'a CudaTensor {
    let cell = &*(t as *const std::cell::UnsafeCell<CudaTensor>);
    &*cell.get()
}

/// raw ポインタから CudaTensor への可変参照取得
unsafe fn get_mut<'a>(t: *mut OpaqueTensor) -> &'a mut CudaTensor {
    let cell = &*(t as *const std::cell::UnsafeCell<CudaTensor>);
    &mut *cell.get()
}

/// raw ポインタから TensorRef (Arc<UnsafeCell<CudaTensor>>) を取得
/// 元の Arc の参照カウントを増やして共有する
unsafe fn tensor_ref_from_ptr(t: *mut OpaqueTensor) -> TensorRef {
    let ptr = t as *const UnsafeCell<CudaTensor>;
    Arc::increment_strong_count(ptr);
    Arc::from_raw(ptr)
}

/// autograd: requires_grad チェック付き set_grad_fn ヘルパー (単項)
unsafe fn set_grad_unary(
    result: &mut CudaTensor,
    input: *mut OpaqueTensor,
    f: impl FnOnce(TensorRef) -> Box<dyn crate::autograd::GradFn>,
) {
    if get(input).requires_grad() {
        let input_ref = tensor_ref_from_ptr(input);
        result.set_grad_fn(f(input_ref));
    }
}

/// autograd: requires_grad チェック付き set_grad_fn ヘルパー (二項)
unsafe fn set_grad_binary(
    result: &mut CudaTensor,
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    f: impl FnOnce(TensorRef, TensorRef) -> Box<dyn crate::autograd::GradFn>,
) {
    if get(a).requires_grad() || get(b).requires_grad() {
        let a_ref = tensor_ref_from_ptr(a);
        let b_ref = tensor_ref_from_ptr(b);
        result.set_grad_fn(f(a_ref, b_ref));
    }
}

// ========== テンソル作成 ==========

#[no_mangle]
pub fn tl_cuda_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let elem_count: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, elem_count) };
    make_tensor(CudaTensor::from_slice(data_slice, shape_slice, DType::F32))
}

#[no_mangle]
pub fn tl_cuda_new_i64(data: *const i64, rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let elem_count: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, elem_count) };
    make_tensor(CudaTensor::from_slice(data_slice, shape_slice, DType::I64))
}

#[no_mangle]
pub fn tl_cuda_zeros(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::slice::from_raw_parts(shape, rank) };
    let mut t = CudaTensor::zeros(s, DType::F32);
    if req_grad {
        t.enable_grad();
    }
    make_tensor(t)
}

#[no_mangle]
pub fn tl_cuda_ones(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::slice::from_raw_parts(shape, rank) };
    let mut t = CudaTensor::ones(s, DType::F32);
    if req_grad {
        t.enable_grad();
    }
    make_tensor(t)
}

#[no_mangle]
pub fn tl_cuda_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::slice::from_raw_parts(shape, rank) };
    let mut t = CudaTensor::randn(s, DType::F32);
    if req_grad {
        t.enable_grad();
    }
    make_tensor(t)
}

#[no_mangle]
pub fn tl_cuda_randn_debug(
    rank: usize,
    shape: *const usize,
    _seed: u64,
    req_grad: bool,
) -> *mut OpaqueTensor {
    tl_cuda_randn(rank, shape, req_grad)
}

#[no_mangle]
pub fn tl_cuda_from_i64(data: *const i64, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::slice::from_raw_parts(data, len) };
    make_tensor(CudaTensor::from_slice(s, &[len], DType::I64))
}

#[no_mangle]
pub fn tl_cuda_from_u8(data: *const u8, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::slice::from_raw_parts(data, len) };
    make_tensor(CudaTensor::from_slice(s, &[len], DType::U8))
}

#[no_mangle]
pub fn tl_cuda_from_vec_u8(data: *mut std::ffi::c_void, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let len = len as usize;
    let s = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
    make_tensor(CudaTensor::from_slice(s, &[len], DType::U8))
}

#[no_mangle]
pub fn tl_cuda_from_u8_labels(data: *const u8, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let len = len as usize;
    let s = unsafe { std::slice::from_raw_parts(data, len) };
    let i64_data: Vec<i64> = s.iter().map(|&x| x as i64).collect();
    make_tensor(CudaTensor::from_slice(&i64_data, &[len], DType::I64))
}

#[no_mangle]
pub fn tl_cuda_from_i64_array(data: *const i64, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let len = len as usize;
    let s = unsafe { std::slice::from_raw_parts(data, len) };
    make_tensor(CudaTensor::from_slice(s, &[len], DType::I64))
}

// ========== テンソル解放 ==========
#[no_mangle]
pub fn tl_cuda_free(t: *mut OpaqueTensor) {
    release_if_live(t);
}

// ========== テンソル情報取得 ==========
#[no_mangle]
pub fn tl_cuda_dim(t: *mut OpaqueTensor, dim: usize) -> usize {
    unsafe { get(t).shape()[dim] }
}
#[no_mangle]
pub fn tl_cuda_len(t: *mut OpaqueTensor) -> usize {
    unsafe { get(t).elem_count() }
}
#[no_mangle]
pub fn tl_cuda_shape(t: *mut OpaqueTensor, dim: usize) -> i64 {
    unsafe { get(t).shape()[dim] as i64 }
}
#[no_mangle]
pub fn tl_cuda_get_shape(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let shape: Vec<f32> = get(t).shape().iter().map(|&x| x as f32).collect();
        let len = shape.len();
        make_tensor(CudaTensor::from_slice(&shape, &[len], DType::F32))
    }
}

// ========== テンソルデータアクセス ==========
#[no_mangle]
pub fn tl_cuda_get_f32(t: *mut OpaqueTensor, idx: usize) -> f32 {
    unsafe { get(t).to_vec::<f32>()[idx] }
}
#[no_mangle]
pub fn tl_cuda_get_i64(t: *mut OpaqueTensor, idx: usize) -> i64 {
    unsafe { get(t).to_vec::<i64>()[idx] }
}
#[no_mangle]
pub fn tl_cuda_set_f32(t: *mut OpaqueTensor, idx: usize, val: f32) {
    unsafe {
        let tensor = get_mut(t);
        let mut data = tensor.to_vec::<f32>();
        data[idx] = val;
        *tensor = CudaTensor::from_slice(&data, tensor.shape(), tensor.dtype());
    }
}
#[no_mangle]
pub fn tl_cuda_item(t: *mut OpaqueTensor) -> f32 {
    unsafe { get(t).to_vec::<f32>()[0] }
}
#[no_mangle]
pub fn tl_cuda_item_i64(t: *mut OpaqueTensor) -> i64 {
    unsafe { get(t).to_vec::<i64>()[0] }
}
#[no_mangle]
pub fn tl_cuda_get(t: *mut OpaqueTensor, idx: i64) -> f32 {
    unsafe { get(t).to_vec::<f32>()[idx as usize] }
}
#[no_mangle]
pub fn tl_cuda_get_f32_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> f32 {
    unsafe {
        let tensor = get(t);
        let cols = tensor.shape()[1];
        tensor.to_vec::<f32>()[idx0 as usize * cols + idx1 as usize]
    }
}
#[no_mangle]
pub fn tl_cuda_get_i64_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> i64 {
    unsafe {
        let tensor = get(t);
        let cols = tensor.shape()[1];
        tensor.to_vec::<i64>()[idx0 as usize * cols + idx1 as usize]
    }
}
#[no_mangle]
pub fn tl_cuda_set_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
    value: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let tensor = get(t);
        let idx_slice = std::slice::from_raw_parts(indices, rank);
        let shape = tensor.shape();
        let mut flat = 0usize;
        let mut stride = 1usize;
        for d in (0..rank).rev() {
            flat += idx_slice[d] as usize * stride;
            stride *= shape[d];
        }
        let mut data = tensor.to_vec::<f32>();
        data[flat] = value;
        make_tensor(CudaTensor::from_slice(&data, shape, tensor.dtype()))
    }
}

// ========== 基本演算 (autograd 対応) ==========
macro_rules! ffi_binary_op {
    ($name:ident, $impl_fn:ident, $backward:ident) => {
        #[no_mangle]
        pub fn $name(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
            unsafe {
                let mut result = match get(a).$impl_fn(get(b)) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("CUDA FFI Error: {}", e);
                        return std::ptr::null_mut();
                    }
                };
                set_grad_binary(&mut result, a, b, |ar, br| {
                    Box::new($backward { a: ar, b: br })
                });
                make_tensor(result)
            }
        }
    };
}

macro_rules! ffi_unary_op {
    ($name:ident, $impl_fn:ident, $backward:ident, input) => {
        #[no_mangle]
        pub fn $name(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
            unsafe {
                let mut result = match get(t).$impl_fn() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("CUDA FFI Error: {}", e);
                        return std::ptr::null_mut();
                    }
                };
                set_grad_unary(&mut result, t, |tr| Box::new($backward { input: tr }));
                make_tensor(result)
            }
        }
    };
    ($name:ident, $impl_fn:ident, $backward:ident, output) => {
        #[no_mangle]
        pub fn $name(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
            unsafe {
                let mut result = match get(t).$impl_fn() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("CUDA FFI Error: {}", e);
                        return std::ptr::null_mut();
                    }
                };
                if get(t).requires_grad() {
                    let input_ref = tensor_ref_from_ptr(t);
                    let out_ref = Arc::new(UnsafeCell::new(result.shallow_clone()));
                    result.set_grad_fn(Box::new($backward {
                        input: input_ref,
                        output: out_ref.clone(),
                    }));
                }
                make_tensor(result)
            }
        }
    };
}

ffi_binary_op!(tl_cuda_add, add_impl, AddBackward);
ffi_binary_op!(tl_cuda_sub, sub_impl, SubBackward);
ffi_binary_op!(tl_cuda_mul, mul_impl, MulBackward);
ffi_binary_op!(tl_cuda_div, div_impl, DivBackward);
ffi_binary_op!(tl_cuda_pow, pow_impl, PowBackward);
ffi_binary_op!(tl_cuda_matmul, matmul_impl, MatmulBackward);

#[no_mangle]
pub fn tl_cuda_rem(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).rem_impl(get(b))) }
}

ffi_unary_op!(tl_cuda_neg, neg_impl, NegBackward, input);
ffi_unary_op!(tl_cuda_abs, abs_impl, AbsBackward, input);

// ========== 活性化関数 (autograd 対応) ==========
ffi_unary_op!(tl_cuda_relu, relu_impl, ReluBackward, input);
ffi_unary_op!(tl_cuda_sigmoid, sigmoid_impl, SigmoidBackward, output);
ffi_unary_op!(tl_cuda_tanh, tanh_impl, TanhBackward, output);
ffi_unary_op!(tl_cuda_gelu, gelu_impl, GeluBackward, input);
ffi_unary_op!(tl_cuda_silu, silu_impl, SiluBackward, input);
ffi_unary_op!(tl_cuda_exp, exp_impl, ExpBackward, output);
ffi_unary_op!(tl_cuda_log, log_impl, LogBackward, input);
#[no_mangle]
pub fn tl_cuda_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).sin_impl()) }
}
#[no_mangle]
pub fn tl_cuda_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).cos_impl()) }
}
#[no_mangle]
pub fn tl_cuda_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).tan_impl()) }
}
ffi_unary_op!(tl_cuda_sqrt, sqrt_impl, SqrtBackward, output);

// ========== スカラー演算 (autograd 対応) ==========
#[no_mangle]
pub fn tl_cuda_add_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).add_scalar_impl(s as f32) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(AddScalarBackward { input: tr })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_mul_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    unsafe {
        let s = s as f32;
        let mut result = match get(t).mul_scalar_impl(s) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(MulScalarBackward {
                input: tr,
                scalar: s,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_sub_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).add_scalar_impl(-(s as f32)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(SubScalarBackward { input: tr })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_div_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    unsafe {
        let s = s as f32;
        let mut result = match get(t).div_scalar_impl(s) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(DivScalarBackward {
                input: tr,
                scalar: s,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_pow_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    unsafe {
        let s = s as f32;
        let mut result = match get(t).pow_scalar_impl(s) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(PowScalarBackward {
                input: tr,
                scalar: s,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_scale(t: *mut OpaqueTensor, s: f32) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).scale_impl(s) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(ScaleBackward {
                input: tr,
                scalar: s,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_clamp(t: *mut OpaqueTensor, min: f64, max: f64) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).clamp_impl(min as f32, max as f32)) }
}

// ========== リダクション (autograd 対応) ==========
#[no_mangle]
pub fn tl_cuda_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).sum_all_tensor_impl() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| Box::new(SumallBackward { input: tr }));
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).mean_all_tensor_impl() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| Box::new(MeanAllBackward { input: tr }));
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).max_all_impl()) }
}
#[no_mangle]
pub fn tl_cuda_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).min_all_impl()) }
}
#[no_mangle]
pub fn tl_cuda_sum_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).sum_impl(dim as i32) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| Box::new(SumDimBackward { input: tr }));
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_mean_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    unsafe {
        let dim_usize = dim;
        let mut result = match get(t).mean_impl(dim as i32) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(MeanDimBackward {
                input: tr,
                dim: dim_usize,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_max_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).max_impl(dim as i32)) }
}
#[no_mangle]
pub fn tl_cuda_min_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).min_impl(dim as i32)) }
}
#[no_mangle]
pub fn tl_cuda_argmax(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).argmax_impl(dim as i32)) }
}
#[no_mangle]
pub fn tl_cuda_argmin(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).argmin_impl(dim as i32)) }
}
#[no_mangle]
pub fn tl_cuda_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    unsafe {
        let mut result = match get(t).softmax_impl(dim as i32) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        if get(t).requires_grad() {
            let input_ref = tensor_ref_from_ptr(t);
            let out_ref = Arc::new(UnsafeCell::new(result.shallow_clone()));
            result.set_grad_fn(Box::new(SoftmaxBackward {
                input: input_ref,
                output: out_ref,
                dim: dim as i32,
            }));
        }
        make_tensor(result)
    }
}

// ========== 型変換 ==========
#[no_mangle]
pub fn tl_cuda_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).to_dtype(DType::F32)) }
}
#[no_mangle]
pub fn tl_cuda_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).to_dtype(DType::I64)) }
}
#[no_mangle]
pub fn tl_cuda_to_device(t: *mut OpaqueTensor, _device_id: i32) -> *mut OpaqueTensor {
    // 単一 GPU のため、データをそのままコピー
    unsafe { make_result(get(t).clone_data()) }
}

// ========== 形状操作 ==========
fn parse_dims(dims: *const i64, num_dims: usize) -> Vec<usize> {
    let raw = unsafe { std::slice::from_raw_parts(dims, num_dims) };
    raw.iter().map(|&d| d as usize).collect()
}

#[no_mangle]
pub fn tl_cuda_reshape(
    t: *mut OpaqueTensor,
    dims: *const i64,
    num_dims: usize,
) -> *mut OpaqueTensor {
    let shape = parse_dims(dims, num_dims);
    unsafe {
        let mut result = match get(t).reshape_impl(&shape) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| Box::new(ReshapeBackward { input: tr }));
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_reshape_new(
    t: *mut OpaqueTensor,
    dims: *const i64,
    num_dims: usize,
) -> *mut OpaqueTensor {
    tl_cuda_reshape(t, dims, num_dims)
}
#[no_mangle]
pub fn tl_cuda_reshape_dims(
    t: *mut OpaqueTensor,
    dims: *const i64,
    num_dims: usize,
) -> *mut OpaqueTensor {
    tl_cuda_reshape(t, dims, num_dims)
}
#[no_mangle]
pub fn tl_cuda_transpose(t: *mut OpaqueTensor, dim0: usize, dim1: usize) -> *mut OpaqueTensor {
    unsafe {
        let d0 = dim0;
        let d1 = dim1;
        let mut result = match get(t).transpose_impl(dim0, dim1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("CUDA FFI Error: {}", e);
                return std::ptr::null_mut();
            }
        };
        set_grad_unary(&mut result, t, |tr| {
            Box::new(TransposeBackward {
                input: tr,
                dim0: d0,
                dim1: d1,
            })
        });
        make_tensor(result)
    }
}
#[no_mangle]
pub fn tl_cuda_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).contiguous_impl()) }
}
#[no_mangle]
pub fn tl_cuda_narrow(
    t: *mut OpaqueTensor,
    dim: usize,
    start: usize,
    len: usize,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).narrow_impl(dim, start, len)) }
}
#[no_mangle]
pub fn tl_cuda_slice(
    t: *mut OpaqueTensor,
    dim: usize,
    start: usize,
    len: usize,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).slice_impl(dim, start, len)) }
}
#[no_mangle]
pub fn tl_cuda_cat(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).cat_impl(get(b), dim as usize)) }
}
#[no_mangle]
pub fn tl_cuda_cat_i64(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_cuda_cat(a, b, dim)
}
#[no_mangle]
pub fn tl_cuda_tril(t: *mut OpaqueTensor, diagonal: i64) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).tril_impl(diagonal as i32)) }
}
#[no_mangle]
pub fn tl_cuda_repeat_interleave(
    t: *mut OpaqueTensor,
    repeats: usize,
    dim: usize,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(t).repeat_interleave_impl(repeats, dim)) }
}
#[no_mangle]
pub fn tl_cuda_sample(t: *mut OpaqueTensor, _temp: f32, _top_p: f32) -> *mut OpaqueTensor {
    // 簡易実装: argmax で最も確率の高いトークンを返す
    unsafe { make_result(get(t).argmax_all_impl()) }
}

// ========== NN ==========
#[no_mangle]
pub fn tl_cuda_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    _bias: *mut OpaqueTensor,
    stride: usize,
    padding: usize,
    _dilation: usize,
    _groups: usize,
) -> *mut OpaqueTensor {
    unsafe {
        make_result(get(input).conv2d_impl(get(weight), (stride, stride), (padding, padding)))
    }
}
#[no_mangle]
pub fn tl_cuda_batch_norm(
    input: *mut OpaqueTensor,
    running_mean: *mut OpaqueTensor,
    running_var: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    _training: bool,
    _momentum: f64,
    eps: f64,
) -> *mut OpaqueTensor {
    unsafe {
        make_result(get(input).batch_norm_impl(
            get(weight),
            get(bias),
            get(running_mean),
            get(running_var),
            eps as f32,
        ))
    }
}
#[no_mangle]
pub fn tl_cuda_layer_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(input).layer_norm_impl(get(weight), get(bias), eps as f32)) }
}
#[no_mangle]
pub fn tl_cuda_max_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(input).max_pool2d_impl((kernel_size, kernel_size), (stride, stride))) }
}
#[no_mangle]
pub fn tl_cuda_avg_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(input).avg_pool2d_impl((kernel_size, kernel_size), (stride, stride))) }
}
#[no_mangle]
pub fn tl_cuda_dropout(input: *mut OpaqueTensor, p: f64, training: bool) -> *mut OpaqueTensor {
    unsafe { make_result(get(input).dropout_impl(p as f32, training)) }
}
#[no_mangle]
pub fn tl_cuda_embedding(
    weight: *mut OpaqueTensor,
    indices: *mut OpaqueTensor,
    _padding_idx: i64,
    _scale_grad_by_freq: bool,
    _sparse: bool,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(weight).embedding_impl(get(indices))) }
}
#[no_mangle]
pub fn tl_cuda_cross_entropy(
    logits: *mut OpaqueTensor,
    labels: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe { make_result(get(logits).cross_entropy_impl(get(labels))) }
}
#[no_mangle]
pub fn tl_cuda_rms_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let normalized = match get(input).rms_norm_impl(eps) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("rms_norm_impl error: {}", e);
                return std::ptr::null_mut();
            }
        };
        if weight.is_null() {
            return make_tensor(normalized);
        }
        // weight (scale) を element-wise 乗算
        match normalized.mul_impl(get(weight)) {
            Ok(res) => make_tensor(res),
            Err(e) => {
                eprintln!("rms_norm weight mul error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

// ========== 比較演算 ==========
#[no_mangle]
pub fn tl_cuda_eq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).eq_impl(get(b))) }
}
#[no_mangle]
pub fn tl_cuda_neq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).ne_impl(get(b))) }
}
#[no_mangle]
pub fn tl_cuda_lt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).lt_impl(get(b))) }
}
#[no_mangle]
pub fn tl_cuda_le(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).le_impl(get(b))) }
}
#[no_mangle]
pub fn tl_cuda_gt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).gt_impl(get(b))) }
}
#[no_mangle]
pub fn tl_cuda_ge(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_result(get(a).ge_impl(get(b))) }
}

// ========== インプレース演算 ==========
#[no_mangle]
pub fn tl_cuda_add_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let result = get(a).add_impl(get(b));
        if let Ok(r) = result {
            *get_mut(a) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_sub_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let result = get(a).sub_impl(get(b));
        if let Ok(r) = result {
            *get_mut(a) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_mul_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let result = get(a).mul_impl(get(b));
        if let Ok(r) = result {
            *get_mut(a) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_div_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let result = get(a).div_impl(get(b));
        if let Ok(r) = result {
            *get_mut(a) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_mod_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let result = get(a).rem_impl(get(b));
        if let Ok(r) = result {
            *get_mut(a) = r;
        }
    }
}

// ========== スカラー In-place 演算 ==========
#[no_mangle]
pub fn tl_cuda_add_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    unsafe {
        let result = get(t).add_scalar_impl(s);
        if let Ok(r) = result {
            *get_mut(t) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_sub_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    unsafe {
        let result = get(t).add_scalar_impl(-s);
        if let Ok(r) = result {
            *get_mut(t) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_mul_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    unsafe {
        let result = get(t).mul_scalar_impl(s);
        if let Ok(r) = result {
            *get_mut(t) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_div_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    unsafe {
        let result = get(t).div_scalar_impl(s);
        if let Ok(r) = result {
            *get_mut(t) = r;
        }
    }
}
#[no_mangle]
pub fn tl_cuda_mod_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    unsafe {
        // fmod: データを直接操作
        let data: Vec<f32> = get(t).to_vec::<f32>().iter().map(|&x| x % s).collect();
        *get_mut(t) = CudaTensor::from_slice(&data, get(t).shape(), get(t).dtype());
    }
}

// ========== RoPE ==========
#[no_mangle]
pub fn tl_cuda_rope_new_cos(dim: usize, seq_len: usize, freq_base: f32) -> *mut OpaqueTensor {
    match CudaTensor::rope_cos_sin_impl(seq_len, dim, freq_base) {
        Ok((cos, _sin)) => make_tensor(cos),
        Err(e) => {
            eprintln!("rope_new_cos error: {}", e);
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub fn tl_cuda_rope_new_sin(dim: usize, seq_len: usize, freq_base: f32) -> *mut OpaqueTensor {
    match CudaTensor::rope_cos_sin_impl(seq_len, dim, freq_base) {
        Ok((_cos, sin)) => make_tensor(sin),
        Err(e) => {
            eprintln!("rope_new_sin error: {}", e);
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub fn tl_cuda_apply_rope(
    q: *mut OpaqueTensor,
    k: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
    pos: usize,
) {
    unsafe {
        if let Ok(new_q) = get(q).apply_rope_impl(get(cos), get(sin), pos) {
            *get_mut(q) = new_q;
        }
        if let Ok(new_k) = get(k).apply_rope_impl(get(cos), get(sin), pos) {
            *get_mut(k) = new_k;
        }
    }
}

// ========== Mask ==========
#[no_mangle]
pub fn tl_cuda_new_causal_mask(size: usize) -> *mut OpaqueTensor {
    make_result(CudaTensor::causal_mask_impl(size))
}

// ========== Device/Grad ==========
#[no_mangle]
pub fn tl_cuda_device_id(_t: *mut OpaqueTensor) -> i32 {
    0
}

#[no_mangle]
pub fn tl_cuda_backward(t: *mut OpaqueTensor) {
    unsafe {
        let _ = get_mut(t).backward();
    }
}
#[no_mangle]
pub fn tl_cuda_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        match get(t).get_grad() {
            Some(g) => make_tensor(g),
            None => {
                // 勾配なし → 入力と同じ shape のゼロテンソルを返す（CPU 版と同じ挙動）
                let shape = get(t).shape().to_vec();
                let zeros = crate::tensor::CudaTensor::zeros(&shape, crate::DType::F32);
                make_tensor(zeros)
            }
        }
    }
}
#[no_mangle]
pub fn tl_cuda_detach(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe { make_tensor(get(t).detach()) }
}
#[no_mangle]
pub fn tl_cuda_enable_grad(t: *mut OpaqueTensor) {
    unsafe {
        get_mut(t).enable_grad();
    }
}
#[no_mangle]
pub fn tl_cuda_replace_data(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    unsafe {
        let new_data = get(b)
            .clone_data()
            .unwrap_or_else(|_| CudaTensor::zeros(&[0], DType::F32));
        *get_mut(a) = new_data;
    }
}
