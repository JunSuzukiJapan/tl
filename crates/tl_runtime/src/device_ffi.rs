//! デバイス統一 FFI ラッパー
//!
//! IDevice トレイトを通じて、CPU/GPU の切替を一元管理する。
//! `builtins.rs` からはこのモジュールの `tl_device_*` 関数をマッピングすればよい。

use std::ffi::c_void;
use std::sync::OnceLock;
use tl_backend::{BackendResult, IDevice};
use tl_cpu::device_impl::CpuDevice;
#[cfg(target_os = "linux")]
use tl_cuda::device_impl::CudaDeviceImpl;
#[cfg(target_os = "macos")]
use tl_metal::device_impl::MetalDeviceImpl;

/// TL_DEVICE 環境変数のキャッシュ
#[inline]
pub(crate) fn is_cpu() -> bool {
    static IS_CPU: OnceLock<bool> = OnceLock::new();
    *IS_CPU.get_or_init(|| std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu"))
}

/// FFI 境界でエラー時に返す安全な値を定義するトレイト
pub trait FFISafeReturn {
    fn error_value() -> Self;
}

impl FFISafeReturn for *mut c_void {
    #[inline(always)]
    fn error_value() -> Self {
        std::ptr::null_mut()
    }
}
impl FFISafeReturn for *const f32 {
    #[inline(always)]
    fn error_value() -> Self {
        std::ptr::null()
    }
}
impl FFISafeReturn for () {
    #[inline(always)]
    fn error_value() -> Self {
        ()
    }
}
impl FFISafeReturn for f32 {
    #[inline(always)]
    fn error_value() -> Self {
        f32::NAN
    }
}
impl FFISafeReturn for f64 {
    #[inline(always)]
    fn error_value() -> Self {
        f64::NAN
    }
}
impl FFISafeReturn for i64 {
    #[inline(always)]
    fn error_value() -> Self {
        -1
    }
}
impl FFISafeReturn for i32 {
    #[inline(always)]
    fn error_value() -> Self {
        -1
    }
}
impl FFISafeReturn for usize {
    #[inline(always)]
    fn error_value() -> Self {
        usize::MAX
    }
}
impl FFISafeReturn for bool {
    #[inline(always)]
    fn error_value() -> Self {
        false
    }
}

/// 汎用ディスパッチ: CPU/GPU を切り替えてクロージャを実行し、エラーをハンドリングする
#[inline]
fn dispatch<F, R>(f: F) -> R
where
    F: FnOnce(&dyn IDevice) -> BackendResult<R>,
    R: FFISafeReturn,
{
    let device: &dyn IDevice = if is_cpu() {
        &CpuDevice
    } else {
        #[cfg(target_os = "macos")]
        {
            &MetalDeviceImpl
        }
        #[cfg(target_os = "linux")]
        {
            &CudaDeviceImpl
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            &CpuDevice
        }
    };

    match f(device) {
        Ok(val) => val,
        Err(e) => {
            // エラーを TLS に設定
            crate::error::set_backend_error(e);
            R::error_value()
        }
    }
}

// ========== テンソル作成 ==========
/// @ffi_sig (f32*, usize, usize*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_new(
    data: *const f32,
    rank: usize,
    shape: *const usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_new(data, rank, shape))
}
/// @ffi_sig (i64*, usize, usize*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_new_i64(
    data: *const i64,
    rank: usize,
    shape: *const usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_new_i64(data, rank, shape))
}
/// @ffi_sig (i64*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_from_i64_array(data: *const i64, len: i64) -> *mut c_void {
    dispatch(|d| d.tensor_from_i64_array(data, len))
}
/// @ffi_sig (usize, usize*, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_zeros(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_zeros(rank, shape, req_grad))
}
/// @ffi_sig (usize, usize*, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_ones(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_ones(rank, shape, req_grad))
}
/// @ffi_sig (usize, usize*, u64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_randn_debug(
    rank: usize,
    shape: *const usize,
    seed: u64,
    req_grad: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_randn_debug(rank, shape, seed, req_grad))
}
/// @ffi_sig (usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_new_causal_mask(size: usize) -> *mut c_void {
    dispatch(|d| d.tensor_new_causal_mask(size))
}
/// @ffi_sig (void*, i64, i64*, i64) -> Tensor*
/// data は Vec<u8> のバッファポインタ, offset はバイトオフセット,
/// shape_ptr は shape 配列 (*const i64), rank は shape の要素数
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_from_vec_u8(data: *mut c_void, offset: i64, shape_ptr: *const i64, rank: i64) -> *mut c_void {
    dispatch(|d| d.tensor_from_vec_u8(data, offset, shape_ptr, rank))
}
/// @ffi_sig (u8*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_from_u8_labels(data: *const u8, len: i64) -> *mut c_void {
    dispatch(|d| d.tensor_from_u8_labels(data, len))
}
/// @ffi_sig (usize, usize*, f32, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_full(
    rank: usize,
    shape: *const usize,
    value: f32,
    req_grad: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_full(rank, shape, value, req_grad))
}
/// @ffi_sig (usize, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_eye(n: usize, req_grad: bool) -> *mut c_void {
    dispatch(|d| d.tensor_eye(n, req_grad))
}
/// @ffi_sig (f64, f64, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_arange(start: f64, end: f64, step: f64) -> *mut c_void {
    dispatch(|d| d.tensor_arange(start, end, step))
}
/// @ffi_sig (f64, f64, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_linspace(start: f64, end: f64, steps: usize) -> *mut c_void {
    dispatch(|d| d.tensor_linspace(start, end, steps))
}
/// @ffi_sig (usize, usize*, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rand(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_rand(rank, shape, req_grad))
}
/// @ffi_sig (Tensor*) -> Tensor*
/// 入力テンソルと同じ形状の一様分布乱数テンソルを生成
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rand_like(t: *mut c_void) -> *mut c_void {
    let shape_t = tl_device_tensor_get_shape(t);
    if shape_t.is_null() {
        return std::ptr::null_mut();
    }
    let rank = dispatch(|d| d.tensor_numel(shape_t)) as usize;
    let data_ptr = dispatch(|d| d.tensor_data(shape_t));
    if data_ptr.is_null() || rank == 0 {
        return std::ptr::null_mut();
    }
    let dims_f32 = unsafe { std::slice::from_raw_parts(data_ptr, rank) };
    let dims_usize: Vec<usize> = dims_f32.iter().map(|&x| x as usize).collect();
    let result = dispatch(|d| d.tensor_rand(rank, dims_usize.as_ptr(), false));
    dispatch(|d| d.tensor_free(shape_t));
    result
}
/// @ffi_sig (Tensor*) -> Tensor*
/// 入力テンソルと同じ形状の正規分布乱数テンソルを生成
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_randn_like(t: *mut c_void) -> *mut c_void {
    let shape_t = tl_device_tensor_get_shape(t);
    if shape_t.is_null() {
        return std::ptr::null_mut();
    }
    let rank = dispatch(|d| d.tensor_numel(shape_t)) as usize;
    let data_ptr = dispatch(|d| d.tensor_data(shape_t));
    if data_ptr.is_null() || rank == 0 {
        return std::ptr::null_mut();
    }
    let dims_f32 = unsafe { std::slice::from_raw_parts(data_ptr, rank) };
    let dims_usize: Vec<usize> = dims_f32.iter().map(|&x| x as usize).collect();
    let result = dispatch(|d| d.tensor_randn_debug(rank, dims_usize.as_ptr(), 0, false));
    dispatch(|d| d.tensor_free(shape_t));
    result
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*) -> Tensor*
/// condition テンソルに基づいて x, y を選択
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_where_cond(
    cond: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
) -> *mut c_void {
    dispatch(|d| d.tensor_where_cond(cond, x, y))
}
/// @ffi_sig (Tensor*, Tensor*, f64) -> Tensor*
/// mask > 0 の位置を value で置換
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_masked_fill_scalar(
    t: *mut c_void,
    mask: *mut c_void,
    value: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_masked_fill_scalar(t, mask, value))
}
/// @ffi_sig (Tensor*, i64, bool) -> Tensor*
/// var with dim (reduce_generic pattern)
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_var_dim(
    t: *mut c_void,
    dim: i64,
    _keepdim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_var(t, dim as i32))
}
/// @ffi_sig (Tensor*, i64, bool) -> Tensor*
/// std with dim (reduce_generic pattern)
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_std_dim(
    t: *mut c_void,
    dim: i64,
    _keepdim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_std(t, dim as i32))
}
/// @ffi_sig (Tensor*, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_var(t: *mut c_void, dim: i32) -> *mut c_void {
    dispatch(|d| d.tensor_var(t, dim))
}
/// @ffi_sig (Tensor*, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_std(t: *mut c_void, dim: i32) -> *mut c_void {
    dispatch(|d| d.tensor_std(t, dim))
}
/// @ffi_sig (Tensor*, i64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_prod_dim(
    t: *mut c_void,
    dim: i64,
    _keepdim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_prod(t, dim as i32))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_prod(t: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_prod(t, -1))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_fill_(t: *mut c_void, value: f32) {
    dispatch(|d| d.tensor_fill_(t, value))
}
/// @ffi_sig (Tensor*, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cumsum(t: *mut c_void, dim: i32) -> *mut c_void {
    dispatch(|d| d.tensor_cumsum(t, dim))
}
/// @ffi_sig (Tensor*, f32, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_norm(t: *mut c_void, p: f32, dim: i32) -> *mut c_void {
    dispatch(|d| d.tensor_norm(t, p, dim))
}
/// @ffi_sig (Tensor*, i64, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_topk(t: *mut c_void, k: i64, dim: i32) -> *mut c_void {
    dispatch(|d| d.tensor_topk(t, k as usize, dim))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_logical_and(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_logical_and(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_logical_or(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_logical_or(a, b))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_logical_not(t: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_logical_not(t))
}
/// @ffi_sig (Tensor*, *const i64, i64) -> Tensor*
/// expand / broadcast_to: shape は dims 配列
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_expand(
    t: *mut c_void,
    dims: *const i64,
    rank: i64,
) -> *mut c_void {
    let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(dims, rank as usize) }
        .iter()
        .map(|&d| d as usize)
        .collect();
    dispatch(|d| d.tensor_broadcast_to(t, &shape))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
/// expand / broadcast_to: shape は Tensor<i64> (e.g. [3, 3] literal)
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_expand_new(
    t: *mut c_void,
    shape_tensor: *mut c_void,
) -> *mut c_void {
    if shape_tensor.is_null() {
        return t;
    }
    // Read shape from shape tensor (contains i64 or f32 dims)
    let rank = dispatch(|d| d.tensor_numel(shape_tensor)) as usize;
    let data_ptr = dispatch(|d| d.tensor_data(shape_tensor));
    if data_ptr.is_null() || rank == 0 {
        return t;
    }
    let dims_f32 = unsafe { std::slice::from_raw_parts(data_ptr, rank) };
    let shape: Vec<usize> = dims_f32.iter().map(|&x| x as usize).collect();
    dispatch(|d| d.tensor_broadcast_to(t, &shape))
}
/// @ffi_sig (Tensor*, Tensor*, i64) -> Tensor*
/// stack: 2つのテンソルを新しい次元で結合
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_stack(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void {
    dispatch(|d| d.tensor_stack(a, b, dim))
}
/// @ffi_sig (Tensor*, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_leaky_relu(t: *mut c_void, slope: f32) -> *mut c_void {
    dispatch(|d| d.tensor_leaky_relu(t, slope))
}
/// @ffi_sig (Tensor*, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_elu(t: *mut c_void, alpha: f32) -> *mut c_void {
    dispatch(|d| d.tensor_elu(t, alpha))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mish(t: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_mish(t))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mse_loss(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_mse_loss(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_l1_loss(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_l1_loss(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_bce_loss(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_bce_loss(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_nll_loss(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_nll_loss(a, b))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_linear(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
) -> *mut c_void {
    dispatch(|d| d.tensor_linear(input, weight, bias))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_hardswish(t: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_hardswish(t))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_hardsigmoid(t: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_hardsigmoid(t))
}
/// @ffi_sig (Tensor*, i64, Tensor*, Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_group_norm(
    input: *mut c_void,
    num_groups: i64,
    weight: *mut c_void,
    bias: *mut c_void,
    eps: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_group_norm(input, num_groups, weight, bias, eps))
}
/// @ffi_sig (Tensor*, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_adaptive_avg_pool2d(
    input: *mut c_void,
    output_h: i64,
    output_w: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_adaptive_avg_pool2d(input, output_h, output_w))
}
/// @ffi_sig (Tensor*, i64, i64, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_pad(
    input: *mut c_void,
    pad_left: i64,
    pad_right: i64,
    value: f32,
) -> *mut c_void {
    dispatch(|d| d.tensor_pad(input, pad_left, pad_right, value))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_instance_norm(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    eps: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_instance_norm(input, weight, bias, eps))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_conv1d(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    stride: i64,
    padding: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_conv1d(input, weight, bias, stride, padding))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_kl_div_loss(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_kl_div_loss(a, b))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_conv_transpose2d(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    stride: i64,
    padding: i64,
    output_padding: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_conv_transpose2d(input, weight, bias, stride, padding, output_padding))
}
/// @ffi_sig (Tensor*, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_interpolate(
    input: *mut c_void,
    output_h: i64,
    output_w: i64,
    mode: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_interpolate(input, output_h, output_w, mode))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sdpa(
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    mask: *mut c_void,
) -> *mut c_void {
    dispatch(|d| d.tensor_scaled_dot_product_attention(q, k, v, mask))
}
/// @ffi_sig (Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_top_k_sample(logits: *mut c_void, k: i64) -> *mut c_void {
    dispatch(|d| d.tensor_top_k_sample(logits, k))
}
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_top_p_sample(logits: *mut c_void, p: f64) -> *mut c_void {
    dispatch(|d| d.tensor_top_p_sample(logits, p))
}
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_temperature_scale(
    logits: *mut c_void,
    temperature: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_temperature_scale(logits, temperature))
}
/// @ffi_sig (Tensor*, Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_repetition_penalty(
    logits: *mut c_void,
    tokens: *mut c_void,
    penalty: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_repetition_penalty(logits, tokens, penalty))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_dot(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_dot(a, b))
}
/// @ffi_sig (Tensor*) -> Tensor*
/// 入力テンソルと同じ形状のゼロテンソルを生成
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_zeros_like(t: *mut c_void) -> *mut c_void {
    // get shape tensor, extract dims, call zeros
    let shape_t = tl_device_tensor_get_shape(t);
    if shape_t.is_null() {
        return std::ptr::null_mut();
    }
    let rank = dispatch(|d| d.tensor_numel(shape_t)) as usize;
    let data_ptr = dispatch(|d| d.tensor_data(shape_t));
    if data_ptr.is_null() || rank == 0 {
        return std::ptr::null_mut();
    }
    // shape tensor contains f32 dims → convert to usize
    let dims_f32 = unsafe { std::slice::from_raw_parts(data_ptr, rank) };
    let dims_usize: Vec<usize> = dims_f32.iter().map(|&x| x as usize).collect();
    let result = dispatch(|d| d.tensor_zeros(rank, dims_usize.as_ptr(), false));
    dispatch(|d| d.tensor_free(shape_t));
    result
}
/// @ffi_sig (Tensor*) -> Tensor*
/// 入力テンソルと同じ形状の全1テンソルを生成
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_ones_like(t: *mut c_void) -> *mut c_void {
    let shape_t = tl_device_tensor_get_shape(t);
    if shape_t.is_null() {
        return std::ptr::null_mut();
    }
    let rank = dispatch(|d| d.tensor_numel(shape_t)) as usize;
    let data_ptr = dispatch(|d| d.tensor_data(shape_t));
    if data_ptr.is_null() || rank == 0 {
        return std::ptr::null_mut();
    }
    let dims_f32 = unsafe { std::slice::from_raw_parts(data_ptr, rank) };
    let dims_usize: Vec<usize> = dims_f32.iter().map(|&x| x as usize).collect();
    let result = dispatch(|d| d.tensor_ones(rank, dims_usize.as_ptr(), false));
    dispatch(|d| d.tensor_free(shape_t));
    result
}
/// @ffi_sig (void*, void*) -> Tensor*
/// Vec<f32> と Vec<i64>(shape) からテンソルを生成
/// data_vec, shape_vec はそれぞれ TL の Vec<f32>, Vec<i64> 構造体ポインタ
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_from_vec_f32(
    data_vec: *mut c_void,
    shape_vec: *mut c_void,
) -> *mut c_void {
    if data_vec.is_null() || shape_vec.is_null() {
        return std::ptr::null_mut();
    }
    // JitVec layout: { ptr: *mut T, cap: i64, len: i64 }
    // Read shape vec
    #[repr(C)]
    struct JitVecI64 {
        ptr: *const i64,
        cap: i64,
        len: i64,
    }
    #[repr(C)]
    struct JitVecF32 {
        ptr: *const f32,
        cap: i64,
        len: i64,
    }

    let shape_jv = unsafe { &*(shape_vec as *const JitVecI64) };
    let data_jv = unsafe { &*(data_vec as *const JitVecF32) };

    let rank = shape_jv.len as usize;
    if rank == 0 || shape_jv.ptr.is_null() || data_jv.ptr.is_null() {
        return std::ptr::null_mut();
    }

    // Convert shape i64 -> usize
    let shape_slice = unsafe { std::slice::from_raw_parts(shape_jv.ptr, rank) };
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    dispatch(|d| d.tensor_new(data_jv.ptr, rank, shape_usize.as_ptr()))
}

/// @ffi_sig (Tensor*) -> Vec<f32>*
/// テンソルの f32 データを TL の Vec<f32> 構造体として返す
/// JitVec layout: { ptr: *mut f32, cap: i64, len: i64 }
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_to_vec_f32(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data: Vec<f32> = read_runtime_tensor_to_f32_vec(t);
    let len = data.len() as i64;
    let cap = data.capacity() as i64;
    let ptr = data.as_ptr() as *mut f32;
    std::mem::forget(data); // Vec の所有権を放棄（JitVec が管理）

    // Allocate JitVec struct on heap: { ptr: *mut f32, cap: i64, len: i64 }
    #[repr(C)]
    struct JitVecF32 {
        ptr: *mut f32,
        cap: i64,
        len: i64,
    }
    let jv = Box::new(JitVecF32 { ptr, cap, len });
    Box::into_raw(jv) as *mut c_void
}

// ========== メモリ管理 ==========
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_clone(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_clone(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_shallow_clone(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_shallow_clone(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_free(a: *mut c_void) {
    dispatch(|d| d.tensor_free(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_release(a: *mut c_void) {
    dispatch(|d| d.tensor_release(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_acquire(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_acquire(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_release_safe(a: *mut c_void) {
    dispatch(|d| d.tensor_release_safe(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_promote(a: *mut c_void) {
    dispatch(|d| d.tensor_promote(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_register(a: *mut c_void) {
    dispatch(|d| d.tensor_register(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_prepare_return(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_prepare_return(a))
}

// ========== テンソル情報 ==========
/// @ffi_sig (Tensor*) -> usize
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_len(a: *mut c_void) -> usize {
    dispatch(|d| d.tensor_len(a))
}
/// @ffi_sig (Tensor*, usize) -> usize
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_dim(a: *mut c_void, dim: usize) -> usize {
    dispatch(|d| d.tensor_dim(a, dim))
}
/// @ffi_sig (Tensor*) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_numel(a: *mut c_void) -> i64 {
    dispatch(|d| d.tensor_numel(a))
}
/// @ffi_sig (Tensor*) -> f32*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_data(a: *mut c_void) -> *const f32 {
    dispatch(|d| d.tensor_data(a))
}
/// @ffi_sig (Tensor*) -> i32
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_device_id(a: *mut c_void) -> i32 {
    dispatch(|d| d.tensor_device_id(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
/// shape テンソルを返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_get_shape(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_get_shape(a))
}

// ========== 要素アクセス ==========
/// @ffi_sig (Tensor*, i64) -> f32
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_get(a: *mut c_void, idx: i64) -> f32 {
    dispatch(|d| d.tensor_get(a, idx))
}
/// @ffi_sig (Tensor*, i64*, i64) -> f32
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_get_f32_md(
    a: *mut c_void,
    indices: *const i64,
    rank: i64,
) -> f32 {
    dispatch(|d| d.tensor_get_f32_md(a, indices, rank))
}
/// @ffi_sig (Tensor*, i64*, i64) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_get_i64_md(
    a: *mut c_void,
    indices: *const i64,
    rank: i64,
) -> i64 {
    dispatch(|d| d.tensor_get_i64_md(a, indices, rank))
}
/// @ffi_sig (Tensor*, i64*, usize, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_set_f32_md(
    a: *mut c_void,
    indices: *const i64,
    rank: usize,
    value: f32,
) -> *mut c_void {
    dispatch(|d| d.tensor_set_f32_md(a, indices, rank, value))
}
/// @ffi_sig (Tensor*) -> f32
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_item(a: *mut c_void) -> f32 {
    dispatch(|d| d.tensor_item(a))
}
/// @ffi_sig (Tensor*) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_item_i64(a: *mut c_void) -> i64 {
    dispatch(|d| d.tensor_item_i64(a))
}

// ========== 二項演算 ==========
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_add(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_add(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sub(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_sub(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mul(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_mul(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_div(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_div(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rem(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_rem(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_matmul(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_matmul(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_pow(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_pow(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cross_entropy(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_cross_entropy(a, b))
}

// ========== 単項演算 ==========
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_neg(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_neg(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_abs(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_abs(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_contiguous(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_contiguous(a))
}

// ========== 比較演算 ==========
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_eq(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_eq(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_neq(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_neq(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_gt(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_gt(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_lt(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_lt(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_ge(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_ge(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_le(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_le(a, b))
}

// ========== スカラー演算 ==========
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_add_scalar(a: *mut c_void, s: f64) -> *mut c_void {
    dispatch(|d| d.tensor_add_scalar(a, s))
}
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sub_scalar(a: *mut c_void, s: f64) -> *mut c_void {
    dispatch(|d| d.tensor_sub_scalar(a, s))
}
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mul_scalar(a: *mut c_void, s: f64) -> *mut c_void {
    dispatch(|d| d.tensor_mul_scalar(a, s))
}
/// @ffi_sig (Tensor*, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_div_scalar(a: *mut c_void, s: f64) -> *mut c_void {
    dispatch(|d| d.tensor_div_scalar(a, s))
}
/// @ffi_sig (Tensor*, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_pow_scalar(a: *mut c_void, exp: f32) -> *mut c_void {
    dispatch(|d| d.tensor_pow_scalar(a, exp))
}
/// @ffi_sig (Tensor*, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_scale(a: *mut c_void, s: f32) -> *mut c_void {
    dispatch(|d| d.tensor_scale(a, s))
}

// ========== Autograd 拡張 ==========
/// @ffi_sig (Tensor*, bool) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_set_requires_grad(t: *mut c_void, req_grad: bool) {
    let _ = dispatch(|d| {
        let _ = d.tensor_set_requires_grad(t, req_grad);
        Ok(std::ptr::null_mut())
    });
}

/// @ffi_sig (Tensor*, f64, f64) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_clip_grad_value(t: *mut c_void, min: f64, max: f64) {
    let _ = dispatch(|d| {
        let _ = d.tensor_clip_grad_value(t, min, max);
        Ok(std::ptr::null_mut())
    });
}

/// @ffi_sig (Tensor*, f64, f64) -> f64
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_clip_grad_norm(
    t: *mut c_void,
    max_norm: f64,
    norm_type: f64,
) -> f64 {
    let mut norm = 0.0f64;
    let _ = dispatch(|d| {
        if let Ok(n) = d.tensor_clip_grad_norm(t, max_norm, norm_type) {
            norm = n;
        }
        Ok(std::ptr::null_mut())
    });
    norm
}

// ========== インプレース演算 ==========
/// @ffi_sig (Tensor*, Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_add_assign(a: *mut c_void, b: *mut c_void) {
    dispatch(|d| d.tensor_add_assign(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sub_assign(a: *mut c_void, b: *mut c_void) {
    dispatch(|d| d.tensor_sub_assign(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mul_assign(a: *mut c_void, b: *mut c_void) {
    dispatch(|d| d.tensor_mul_assign(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_div_assign(a: *mut c_void, b: *mut c_void) {
    dispatch(|d| d.tensor_div_assign(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mod_assign(a: *mut c_void, b: *mut c_void) {
    dispatch(|d| d.tensor_mod_assign(a, b))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_add_assign_scalar_f32(a: *mut c_void, s: f32) {
    dispatch(|d| d.tensor_add_assign_scalar_f32(a, s))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sub_assign_scalar_f32(a: *mut c_void, s: f32) {
    dispatch(|d| d.tensor_sub_assign_scalar_f32(a, s))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mul_assign_scalar_f32(a: *mut c_void, s: f32) {
    dispatch(|d| d.tensor_mul_assign_scalar_f32(a, s))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_div_assign_scalar_f32(a: *mut c_void, s: f32) {
    dispatch(|d| d.tensor_div_assign_scalar_f32(a, s))
}
/// @ffi_sig (Tensor*, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mod_assign_scalar_f32(a: *mut c_void, s: f32) {
    dispatch(|d| d.tensor_mod_assign_scalar_f32(a, s))
}

// ========== 数学・活性化関数 ==========
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_exp(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_exp(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_log(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_log(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sqrt(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_sqrt(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sin(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_sin(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cos(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_cos(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_tan(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_tan(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_tanh(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_tanh(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sigmoid(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_sigmoid(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_relu(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_relu(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_gelu(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_gelu(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_silu(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_silu(a))
}

// ========== Reduction ==========
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sum(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_sum(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mean(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_mean(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_max(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_max(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_min(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_min(a))
}
/// @ffi_sig (Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_softmax(a: *mut c_void, dim: i64) -> *mut c_void {
    dispatch(|d| d.tensor_softmax(a, dim))
}
/// @ffi_sig (Tensor*, usize, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_max_dim(
    a: *mut c_void,
    dim: usize,
    keep_dim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_max_dim(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, usize, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_min_dim(
    a: *mut c_void,
    dim: usize,
    keep_dim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_min_dim(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, usize, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_mean_dim(
    a: *mut c_void,
    dim: usize,
    keep_dim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_mean_dim(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, usize, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sum_dim(
    a: *mut c_void,
    dim: usize,
    keep_dim: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_sum_dim(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, i64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_argmax(a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void {
    dispatch(|d| d.tensor_argmax(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, i64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_argmin(a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void {
    dispatch(|d| d.tensor_argmin(a, dim, keep_dim))
}
/// @ffi_sig (Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_tril(a: *mut c_void, diagonal: i64) -> *mut c_void {
    dispatch(|d| d.tensor_tril(a, diagonal))
}
/// @ffi_sig (Tensor*, f64, f64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_clamp(a: *mut c_void, min: f64, max: f64) -> *mut c_void {
    dispatch(|d| d.tensor_clamp(a, min, max))
}
/// @ffi_sig (Tensor*, f32, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_sample(a: *mut c_void, temp: f32, top_p: f32) -> *mut c_void {
    dispatch(|d| d.tensor_sample(a, temp, top_p))
}

// ========== Autograd ==========
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_backward(a: *mut c_void) {
    dispatch(|d| d.tensor_backward(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_grad(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_grad(a))
}
/// @ffi_sig (Tensor*, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_detach(a: *mut c_void, req_grad: bool) -> *mut c_void {
    dispatch(|d| d.tensor_detach(a, req_grad))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_enable_grad(a: *mut c_void) {
    dispatch(|d| d.tensor_enable_grad(a))
}
/// @ffi_sig () -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_clear_grads() {
    dispatch(|d| d.clear_grads())
}

// ========== 形状操作 ==========
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
/// s は shape テンソル
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_reshape_new(a: *mut c_void, s: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_reshape_new(a, s))
}
/// @ffi_sig (Tensor*, i64*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_reshape_dims(
    a: *mut c_void,
    dims_ptr: *const i64,
    rank: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_reshape_dims(a, dims_ptr, rank))
}
/// @ffi_sig (Tensor*, usize, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_transpose(
    a: *mut c_void,
    dim0: usize,
    dim1: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_transpose(a, dim0, dim1))
}
/// @ffi_sig (Tensor*, i64, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_slice(
    a: *mut c_void,
    dim: i64,
    start: i64,
    end: i64,
    step: i64,
) -> *mut c_void {
    // Convert (dim, start, end, step) to (dim, start, len)
    // step は現在1のみ対応
    let len = if step == 1 {
        end - start
    } else {
        // step != 1: (end - start + step - 1) / step
        (end - start + step - 1) / step
    };
    dispatch(|d| d.tensor_slice(a, dim, start, len))
}
/// @ffi_sig (Tensor*, usize, usize, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_narrow(
    a: *mut c_void,
    dim: usize,
    start: usize,
    len: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_narrow(a, dim, start, len))
}
/// @ffi_sig (Tensor*, Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cat(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void {
    dispatch(|d| d.tensor_cat(a, b, dim))
}
/// @ffi_sig (Tensor*, Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cat_i64(
    a: *mut c_void,
    b: *mut c_void,
    dim: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_cat_i64(a, b, dim))
}
/// @ffi_sig (Tensor*, Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cat2(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void {
    dispatch(|d| d.tensor_cat2(a, b, dim))
}
/// @ffi_sig (Tensor*, Tensor*, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_cat_4d(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void {
    dispatch(|d| d.tensor_cat_4d(a, b, dim))
}
/// @ffi_sig (Tensor*, Tensor*) -> void
/// dst のデータを src のデータで置換
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_replace_data(dst: *mut c_void, src: *mut c_void) {
    dispatch(|d| d.tensor_replace_data(dst, src))
}
/// @ffi_sig (Tensor*, usize, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_repeat_interleave(
    a: *mut c_void,
    repeats: usize,
    dim: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_repeat_interleave(a, repeats, dim))
}
/// @ffi_sig (Tensor*, i32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_to_device(a: *mut c_void, device_id: i32) -> *mut c_void {
    dispatch(|d| d.tensor_to_device(a, device_id))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_to_f32(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_to_f32(a))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_to_i64(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_to_i64(a))
}
/// @ffi_sig (Tensor*, Tensor*, i64, bool, bool) -> Tensor*
/// w: 重みテンソル, idx: インデックステンソル
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_embedding(
    w: *mut c_void,
    idx: *mut c_void,
    pad: i64,
    sg: bool,
    sp: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_embedding(w, idx, pad, sg, sp))
}

// ========== LLM ==========
/// @ffi_sig (Tensor*, Tensor*, f32) -> Tensor*
/// a: 入力テンソル, w: 重みテンソル
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rms_norm(
    a: *mut c_void,
    w: *mut c_void,
    eps: f32,
) -> *mut c_void {
    dispatch(|d| d.tensor_rms_norm(a, w, eps))
}
/// @ffi_sig (usize, usize, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rope_new_cos(
    dim: usize,
    seq_len: usize,
    base: f32,
) -> *mut c_void {
    dispatch(|d| d.tensor_rope_new_cos(dim, seq_len, base))
}
/// @ffi_sig (usize, usize, f32) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_rope_new_sin(
    dim: usize,
    seq_len: usize,
    base: f32,
) -> *mut c_void {
    dispatch(|d| d.tensor_rope_new_sin(dim, seq_len, base))
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*) -> Tensor*
/// a: 入力テンソル, cos: cosキャッシュ, sin: sinキャッシュ
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_apply_rope(
    a: *mut c_void,
    cos: *mut c_void,
    sin: *mut c_void,
) -> *mut c_void {
    dispatch(|d| d.tensor_apply_rope(a, cos, sin))
}

// ========== IO / Print ==========
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_print(a: *mut c_void) {
    dispatch(|d| d.tensor_print(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_display(a: *mut c_void) {
    dispatch(|d| d.tensor_display(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_print_1(a: *mut c_void) {
    dispatch(|d| d.tensor_print_1(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_print_2(a: *mut c_void) {
    dispatch(|d| d.tensor_print_2(a))
}
/// @ffi_sig (Tensor*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_print_3(a: *mut c_void) {
    dispatch(|d| d.tensor_print_3(a))
}
/// @ffi_sig (Tensor*, i8*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_save(a: *mut c_void, path: *const i8) {
    dispatch(|d| d.tensor_save(a, path))
}
/// @ffi_sig (i8*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_load(path: *const i8) -> *mut c_void {
    dispatch(|d| d.tensor_load(path))
}

// ========== NN ==========
/// @ffi_sig (Tensor*, Tensor*, Tensor*, usize, usize, usize, usize) -> Tensor*
/// input, weight, bias テンソル + stride, padding, dilation, groups
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_conv2d(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_conv2d(input, weight, bias, stride, padding, dilation, groups))
}

/// Simple conv2d wrapper: 4-arg (input, weight, padding, stride) → 7-arg full conv2d
/// codegen は (input, weight, padding: i64, stride: i64) を渡すため、
/// bias=null, dilation=1, groups=1 を補完して完全なシグネチャに変換する
#[unsafe(no_mangle)]
pub extern "C" fn tl_conv2d_simple_wrapper(
    input: *mut c_void,
    weight: *mut c_void,
    padding: i64,
    stride: i64,
) -> *mut c_void {
    tl_device_tensor_conv2d(
        input,
        weight,
        std::ptr::null_mut(), // bias = null
        stride as usize,
        padding as usize,
        1, // dilation = 1
        1, // groups = 1
    )
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, Tensor*, Tensor*, bool, f64, f64) -> Tensor*
/// input, running_mean, running_var, weight, bias + training, momentum, eps
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_batch_norm(
    input: *mut c_void,
    running_mean: *mut c_void,
    running_var: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    training: bool,
    momentum: f64,
    eps: f64,
) -> *mut c_void {
    dispatch(|d| {
        d.tensor_batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            training,
            momentum,
            eps,
        )
    })
}
/// @ffi_sig (Tensor*, Tensor*, Tensor*, f64) -> Tensor*
/// input, weight, bias + eps
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_layer_norm(
    input: *mut c_void,
    weight: *mut c_void,
    bias: *mut c_void,
    eps: f64,
) -> *mut c_void {
    dispatch(|d| d.tensor_layer_norm(input, weight, bias, eps))
}
/// @ffi_sig (Tensor*, f64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_dropout(
    input: *mut c_void,
    p: f64,
    training: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_dropout(input, p, training))
}
/// @ffi_sig (Tensor*, f64, bool) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_dropout2d(
    input: *mut c_void,
    p: f64,
    training: bool,
) -> *mut c_void {
    dispatch(|d| d.tensor_dropout2d(input, p, training))
}
/// @ffi_sig (Tensor*, usize, usize, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_max_pool2d(
    input: *mut c_void,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_max_pool2d(input, kernel_size, stride, padding))
}
/// @ffi_sig (Tensor*, usize, usize, usize) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_avg_pool2d(
    input: *mut c_void,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> *mut c_void {
    dispatch(|d| d.tensor_avg_pool2d(input, kernel_size, stride, padding))
}

// ========== CPU 専用 (device_ffi 経由) ==========
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_transpose_2d(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_transpose_2d(a))
}
/// @ffi_sig (Tensor*, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_reshape_2d(a: *mut c_void, d0: i64, d1: i64) -> *mut c_void {
    dispatch(|d| d.tensor_reshape_2d(a, d0, d1))
}
/// @ffi_sig (Tensor*, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_reshape_3d_to_2d(
    a: *mut c_void,
    d0: i64,
    d1: i64,
) -> *mut c_void {
    dispatch(|d| d.tensor_reshape_3d_to_2d(a, d0, d1))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_matmul_4d(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_matmul_4d(a, b))
}
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_add_4d(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_add_4d(a, b))
}
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_silu_4d(a: *mut c_void) -> *mut c_void {
    dispatch(|d| d.tensor_silu_4d(a))
}

// Helper: Convert tensor shape to Vec<i64> for JIT
#[repr(C)]
struct JitVec {
    ptr: *mut i64,
    cap: i64,
    len: i64,
}

/// @ffi_sig (Tensor*) -> void*
/// テンソルの shape を JitVec (Vec<i64>) に変換して返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_shape_vec(a: *mut c_void) -> *mut c_void {
    // 1. Get shape as Tensor (F32)
    let shape_tensor = tl_device_tensor_get_shape(a);
    if shape_tensor.is_null() {
        return std::ptr::null_mut();
    }

    // 2. Get data and len
    let len_i64 = tl_device_tensor_numel(shape_tensor);
    let len = len_i64 as usize;
    let data_ptr = tl_device_tensor_data(shape_tensor);

    // 3. Convert to Vec<i64>
    let mut vec_i64 = Vec::with_capacity(len);
    if !data_ptr.is_null() {
        unsafe {
            let slice = std::slice::from_raw_parts(data_ptr, len);
            for &val in slice {
                vec_i64.push(val as i64);
            }
        }
    }

    // 4. Free shape tensor
    tl_device_tensor_free(shape_tensor);

    // 5. Construct JitVec matching struct Vec<T> { ptr, cap, len }
    let cap = vec_i64.capacity();
    let len = vec_i64.len();
    let ptr = vec_i64.as_mut_ptr();
    std::mem::forget(vec_i64);

    let jit_vec = JitVec {
        ptr,
        cap: cap as i64,
        len: len as i64,
    };

    let boxed = Box::new(jit_vec);
    Box::into_raw(boxed) as *mut c_void
}

// ========== ランタイムヘルパー（GPU 直接参照を集約） ==========

/// テンソル作成ヘルパー: CPU/GPU 両対応
/// 各モジュールがバックエンドを直接呼ぶ代わりにこの関数を使う
pub fn create_runtime_tensor_f32(data: &[f32], shape: &[usize]) -> *mut c_void {
    if is_cpu() {
        tl_cpu::ffi::tl_cpu_tensor_new(data.as_ptr(), shape.len(), shape.as_ptr()) as *mut c_void
    } else {
        #[cfg(target_os = "macos")]
        {
            tl_metal::ffi_ops::tl_metal_new(data.as_ptr(), shape.len(), shape.as_ptr())
                as *mut c_void
        }
        #[cfg(target_os = "linux")]
        {
            tl_cuda::ffi_ops::tl_cuda_new(data.as_ptr(), shape.len(), shape.as_ptr()) as *mut c_void
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            tl_cpu::ffi::tl_cpu_tensor_new(data.as_ptr(), shape.len(), shape.as_ptr())
                as *mut c_void
        }
    }
}

/// テンソル読み取りヘルパー: CPU/GPU 両対応
/// テンソルポインタから f32 データを Vec<f32> として読み出す
pub fn read_runtime_tensor_to_f32_vec(t: *mut c_void) -> Vec<f32> {
    if t.is_null() {
        return Vec::new();
    }
    if is_cpu() {
        let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
        tensor.data_f32().to_vec()
    } else {
        #[cfg(target_os = "macos")]
        {
            let tensor = unsafe { &*(t as *const std::cell::UnsafeCell<tl_metal::MetalTensor>) };
            unsafe { (*tensor.get()).to_vec::<f32>() }
        }
        #[cfg(target_os = "linux")]
        {
            let tensor =
                unsafe { &*(t as *const std::cell::UnsafeCell<tl_cuda::tensor::CudaTensor>) };
            unsafe { (*tensor.get()).to_vec::<f32>() }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
            tensor.data_f32().to_vec()
        }
    }
}

/// テンソル読み取りヘルパー (i64): CPU/GPU 両対応
/// テンソルポインタから i64 データを Vec<i64> として読み出す
pub fn read_runtime_tensor_to_i64_vec(t: *mut c_void) -> Vec<i64> {
    if t.is_null() {
        return Vec::new();
    }
    if is_cpu() {
        let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
        // CPU テンソルは f32 で格納されるため、f32 → i64 変換
        tensor.data_f32().iter().map(|&v| v as i64).collect()
    } else {
        #[cfg(target_os = "macos")]
        {
            let tensor = unsafe { &*(t as *const std::cell::UnsafeCell<tl_metal::MetalTensor>) };
            unsafe { (*tensor.get()).to_vec::<i64>() }
        }
        #[cfg(target_os = "linux")]
        {
            let tensor =
                unsafe { &*(t as *const std::cell::UnsafeCell<tl_cuda::tensor::CudaTensor>) };
            unsafe { (*tensor.get()).to_vec::<i64>() }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
            tensor.data_f32().iter().map(|&v| v as i64).collect()
        }
    }
}

/// テンソル shape 読み取りヘルパー: CPU/GPU 両対応
/// OpaqueTensor ポインタから shape を Vec<usize> として読み出す
pub fn read_runtime_tensor_shape(t: *mut c_void) -> Vec<usize> {
    if t.is_null() {
        return Vec::new();
    }
    if is_cpu() {
        let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
        tensor.shape().to_vec()
    } else {
        #[cfg(target_os = "macos")]
        {
            let tensor = unsafe { &*(t as *const std::cell::UnsafeCell<tl_metal::MetalTensor>) };
            unsafe { (*tensor.get()).shape().to_vec() }
        }
        #[cfg(target_os = "linux")]
        {
            let tensor =
                unsafe { &*(t as *const std::cell::UnsafeCell<tl_cuda::tensor::CudaTensor>) };
            unsafe { (*tensor.get()).shape().to_vec() }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let tensor = unsafe { &*(t as *mut tl_cpu::CpuTensor<f32>) };
            tensor.shape().to_vec()
        }
    }
}

/// U8 テンソル作成ヘルパー
pub fn create_runtime_tensor_u8(data: &[u8], shape: &[usize]) -> *mut c_void {
    if is_cpu() {
        // CPU: u8 データを f32 に変換して作成
        let f32_data: Vec<f32> = data.iter().map(|&b| b as f32).collect();
        tl_cpu::ffi::tl_cpu_tensor_new(f32_data.as_ptr(), shape.len(), shape.as_ptr())
            as *mut c_void
    } else {
        #[cfg(target_os = "macos")]
        {
            let tensor = tl_metal::MetalTensor::from_slice(data, shape, tl_metal::DType::U8);
            crate::make_tensor(tensor) as *mut c_void
        }
        #[cfg(target_os = "linux")]
        {
            let tensor = tl_cuda::CudaTensor::from_slice(data, shape, tl_cuda::DType::U8);
            crate::make_tensor(tensor) as *mut c_void
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let f32_data: Vec<f32> = data.iter().map(|&b| b as f32).collect();
            tl_cpu::ffi::tl_cpu_tensor_new(f32_data.as_ptr(), shape.len(), shape.as_ptr())
                as *mut c_void
        }
    }
}

/// ゼロテンソル作成ヘルパー
pub fn create_runtime_zeros(shape: &[usize]) -> *mut c_void {
    if is_cpu() {
        let data = vec![0.0f32; shape.iter().product()];
        tl_cpu::ffi::tl_cpu_tensor_new(data.as_ptr(), shape.len(), shape.as_ptr()) as *mut c_void
    } else {
        #[cfg(target_os = "macos")]
        {
            let tensor = tl_metal::MetalTensor::zeros(shape, tl_metal::DType::F32);
            crate::make_tensor(tensor) as *mut c_void
        }
        #[cfg(target_os = "linux")]
        {
            let tensor = tl_cuda::CudaTensor::zeros(shape, tl_cuda::DType::F32);
            crate::make_tensor(tensor) as *mut c_void
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let data = vec![0.0f32; shape.iter().product()];
            tl_cpu::ffi::tl_cpu_tensor_new(data.as_ptr(), shape.len(), shape.as_ptr())
                as *mut c_void
        }
    }
}

/// テンソル解放ヘルパー: CPU/GPU 両対応
pub fn release_runtime_tensor(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    if is_cpu() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut tl_cpu::CpuTensor<f32>);
        }
    } else {
        #[cfg(target_os = "macos")]
        {
            tl_metal::ffi::tl_metal_release(ptr as *mut tl_metal::MetalTensor);
        }
        #[cfg(target_os = "linux")]
        {
            tl_cuda::ffi::tl_cuda_release(ptr as *mut tl_cuda::CudaTensor);
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            unsafe {
                let _ = Box::from_raw(ptr as *mut tl_cpu::CpuTensor<f32>);
            }
        }
    }
}

/// GPU Sync ヘルパー
pub fn runtime_gpu_sync() {
    if !is_cpu() {
        #[cfg(target_os = "macos")]
        {
            if let Some(device) = tl_metal::device::try_get_device() {
                let cmd = device.command_queue().new_command_buffer();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
        #[cfg(target_os = "linux")]
        {
            unsafe extern "C" {
                fn cudaDeviceSynchronize() -> i32;
            }
            unsafe {
                cudaDeviceSynchronize();
            }
        }
    }
}

// ========== CUDA 融合 Q4_K / Q6_K matmul ==========

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn tl_cuda_mul_mv_q4_k(input: *mut c_void, w_raw: *mut c_void, n: i64, k: i64) -> *mut c_void;
    fn tl_cuda_mul_mv_q6_k(input: *mut c_void, w_raw: *mut c_void, n: i64, k: i64) -> *mut c_void;
}

pub fn cuda_mul_mv_q4_k(input: *mut c_void, w_raw: *mut c_void, n: i64, k: i64) -> *mut c_void {
    #[cfg(target_os = "linux")]
    unsafe {
        return tl_cuda_mul_mv_q4_k(input, w_raw, n, k);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (input, w_raw, n, k);
        std::ptr::null_mut()
    }
}

pub fn cuda_mul_mv_q6_k(input: *mut c_void, w_raw: *mut c_void, n: i64, k: i64) -> *mut c_void {
    #[cfg(target_os = "linux")]
    unsafe {
        return tl_cuda_mul_mv_q6_k(input, w_raw, n, k);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (input, w_raw, n, k);
        std::ptr::null_mut()
    }
}

// ========== chunk / split (narrow ラッパー) ==========

/// t.chunk(num_chunks, dim, index) → narrow(dim, chunk_size*index, chunk_size)
/// @ffi_sig (Tensor*, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_chunk(
    t: *mut c_void,
    num_chunks: i64,
    dim: i64,
    index: i64,
) -> *mut c_void {
    if t.is_null() || num_chunks <= 0 {
        return std::ptr::null_mut();
    }
    // dim 方向のサイズを取得
    let dim_size = tl_device_tensor_dim(t, dim as usize) as i64;
    let chunk_size = (dim_size + num_chunks - 1) / num_chunks; // ceil division
    let start = chunk_size * index;
    let len = std::cmp::min(chunk_size, dim_size - start);
    if start >= dim_size || len <= 0 {
        return std::ptr::null_mut();
    }
    dispatch(|d| d.tensor_narrow(t, dim as usize, start as usize, len as usize))
}

/// t.split(split_size, dim, index) → narrow(dim, split_size*index, min(split_size, remaining))
/// @ffi_sig (Tensor*, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_device_tensor_split(
    t: *mut c_void,
    split_size: i64,
    dim: i64,
    index: i64,
) -> *mut c_void {
    if t.is_null() || split_size <= 0 {
        return std::ptr::null_mut();
    }
    let dim_size = tl_device_tensor_dim(t, dim as usize) as i64;
    let start = split_size * index;
    let len = std::cmp::min(split_size, dim_size - start);
    if start >= dim_size || len <= 0 {
        return std::ptr::null_mut();
    }
    dispatch(|d| d.tensor_narrow(t, dim as usize, start as usize, len as usize))
}

// ========== 線形代数 ==========

/// 逆行列 (CPU): ガウス・ジョルダン消去法
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_inverse(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!(
            "Error: inverse requires square matrix, got {} elements",
            n_sq
        );
        return std::ptr::null_mut();
    }
    // 拡大行列 [A | I]
    let mut aug = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    // ガウス・ジョルダン
    for col in 0..n {
        // ピボット選択
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            eprintln!("Error: singular matrix in inverse");
            return std::ptr::null_mut();
        }
        // 行交換
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }
        // 正規化
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        // 消去
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    // 逆行列を抽出
    let mut result = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    let shape = [n, n];
    create_runtime_tensor_f32(&result, &shape)
}

/// 行列式 (CPU): LU分解ベース
/// @ffi_sig (Tensor*) -> Tensor* (scalar)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_det(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: det requires square matrix");
        return std::ptr::null_mut();
    }
    let mut mat = data.clone();
    let mut det: f32 = 1.0;
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return create_runtime_tensor_f32(&[0.0], &[1]);
        }
        if max_row != col {
            for j in 0..n {
                mat.swap(col * n + j, max_row * n + j);
            }
            det = -det;
        }
        det *= mat[col * n + col];
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for j in col..n {
                mat[row * n + j] -= factor * mat[col * n + j];
            }
        }
    }
    create_runtime_tensor_f32(&[det], &[1])
}

/// 連立方程式 Ax=b (CPU): LU分解ベース
/// @ffi_sig (Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_solve(a: *mut c_void, b: *mut c_void) -> *mut c_void {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let a_data = read_runtime_tensor_to_f32_vec(a);
    let b_data = read_runtime_tensor_to_f32_vec(b);
    let n_sq = a_data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || b_data.len() != n {
        eprintln!("Error: solve requires square A and compatible b");
        return std::ptr::null_mut();
    }
    // 拡大行列 [A | b]
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_data[i * n + j];
        }
        aug[i * (n + 1) + n] = b_data[i];
    }
    // 前進消去
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            eprintln!("Error: singular matrix in solve");
            return std::ptr::null_mut();
        }
        if max_row != col {
            for j in 0..(n + 1) {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for j in col..(n + 1) {
            aug[col * (n + 1) + j] /= pivot;
        }
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col];
            for j in col..(n + 1) {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }
    // 後退代入
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        x[i] = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            x[i] -= aug[i * (n + 1) + j] * x[j];
        }
    }
    create_runtime_tensor_f32(&x, &[n])
}

/// SVD: U成分 (CPU: Householder二重対角化 + 暗黙的QRシフト)
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_svd_u(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: svd_u requires square matrix");
        return std::ptr::null_mut();
    }
    let (u, _, _) = svd_decompose(&data, n);
    create_runtime_tensor_f32(&u, &[n, n])
}

/// SVD: S成分 (特異値ベクトル)
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_svd_s(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: svd_s requires square matrix");
        return std::ptr::null_mut();
    }
    let (_, s, _) = svd_decompose(&data, n);
    create_runtime_tensor_f32(&s, &[n])
}

/// SVD: V成分
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_svd_v(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: svd_v requires square matrix");
        return std::ptr::null_mut();
    }
    let (_, _, v) = svd_decompose(&data, n);
    create_runtime_tensor_f32(&v, &[n, n])
}

/// 固有値 (CPU: QR法)
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_eig_values(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: eig_values requires square matrix");
        return std::ptr::null_mut();
    }
    let (values, _) = eigen_decompose(&data, n);
    create_runtime_tensor_f32(&values, &[n])
}

/// 固有ベクトル (CPU: QR法 + 逆反復法)
/// @ffi_sig (Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_eig_vectors(t: *mut c_void) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let data = read_runtime_tensor_to_f32_vec(t);
    let n_sq = data.len();
    let n = (n_sq as f64).sqrt() as usize;
    if n * n != n_sq || n == 0 {
        eprintln!("Error: eig_vectors requires square matrix");
        return std::ptr::null_mut();
    }
    let (_, vectors) = eigen_decompose(&data, n);
    create_runtime_tensor_f32(&vectors, &[n, n])
}

// ========== SVD 自前実装 ==========

/// SVD分解: A = U * diag(S) * V^T
/// Jacobi rotation法による自前実装 (改造しやすい構造)
fn svd_decompose(data: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // A^T * A の固有値分解から V と特異値を求める
    // U = A * V * diag(1/sigma)
    let mut ata = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += data[k * n + i] * data[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // A^T*A の固有値分解（対称行列なので Jacobi 法が使える）
    let (eigenvalues, v) = symmetric_eigen_jacobi(&ata, n);

    // 特異値 = sqrt(固有値)、降順ソート
    let mut sv_indices: Vec<(f32, usize)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &val)| (if val > 0.0 { val.sqrt() } else { 0.0 }, i))
        .collect();
    sv_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = vec![0.0f32; n];
    let mut v_sorted = vec![0.0f32; n * n];
    for (new_idx, &(sigma, old_idx)) in sv_indices.iter().enumerate() {
        s[new_idx] = sigma;
        for row in 0..n {
            v_sorted[row * n + new_idx] = v[row * n + old_idx];
        }
    }

    // U = A * V * diag(1/sigma)
    let mut u = vec![0.0f32; n * n];
    for j in 0..n {
        if s[j] > 1e-10 {
            let inv_sigma = 1.0 / s[j];
            for i in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += data[i * n + k] * v_sorted[k * n + j];
                }
                u[i * n + j] = sum * inv_sigma;
            }
        } else {
            // ゼロ特異値: 単位ベクトルで埋める
            u[j * n + j] = 1.0;
        }
    }

    (u, s, v_sorted)
}

/// 対称行列の固有値分解 (Jacobi回転法)
/// (固有値ベクトル, 固有ベクトル行列) を返す
fn symmetric_eigen_jacobi(mat: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = mat.to_vec();
    // 固有ベクトル行列を単位行列で初期化
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = 1e-10f32;

    for _ in 0..max_iter {
        // 最大の非対角要素を見つける
        let mut max_val = 0.0f32;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Jacobi 回転角の計算
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // A' = G^T * A * G (Givens 回転)
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = cos_t * aip + sin_t * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -sin_t * aip + cos_t * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = cos_t * cos_t * app + 2.0 * cos_t * sin_t * apq + sin_t * sin_t * aqq;
        new_a[q * n + q] = sin_t * sin_t * app - 2.0 * cos_t * sin_t * apq + cos_t * cos_t * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // 固有ベクトルの更新: V' = V * G
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = cos_t * vip + sin_t * viq;
            v[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

// ========== 固有値分解 自前実装 ==========

/// 一般行列の固有値分解 (QR法 + Hessenberg前処理)
/// (固有値ベクトル, 固有ベクトル行列) を返す
fn eigen_decompose(data: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    // 対称行列チェック
    let mut is_symmetric = true;
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            if (data[i * n + j] - data[j * n + i]).abs() > 1e-6 {
                is_symmetric = false;
                break 'outer;
            }
        }
    }

    if is_symmetric {
        return symmetric_eigen_jacobi(data, n);
    }

    // 非対称行列: QR反復法
    let mut h = data.to_vec();

    // Hessenberg 化 (Householder変換)
    let mut q_accum = vec![0.0f32; n * n];
    for i in 0..n {
        q_accum[i * n + i] = 1.0;
    }

    for k in 0..(n.saturating_sub(2)) {
        // k列目の(k+1)行以降からHouseholderベクトルを構成
        let mut x = vec![0.0f32; n - k - 1];
        for i in 0..x.len() {
            x[i] = h[(k + 1 + i) * n + k];
        }

        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_x < 1e-12 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let norm_v: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_v < 1e-12 {
            continue;
        }
        for v in x.iter_mut() {
            *v /= norm_v;
        }

        // H = H - 2 * v * (v^T * H)  (左から)
        let m = x.len();
        for j in 0..n {
            let mut dot = 0.0f32;
            for i in 0..m {
                dot += x[i] * h[(k + 1 + i) * n + j];
            }
            for i in 0..m {
                h[(k + 1 + i) * n + j] -= 2.0 * x[i] * dot;
            }
        }

        // H = H - 2 * (H * v) * v^T  (右から)
        for i in 0..n {
            let mut dot = 0.0f32;
            for j in 0..m {
                dot += h[i * n + (k + 1 + j)] * x[j];
            }
            for j in 0..m {
                h[i * n + (k + 1 + j)] -= 2.0 * dot * x[j];
            }
        }

        // Q の累積
        for i in 0..n {
            let mut dot = 0.0f32;
            for j in 0..m {
                dot += q_accum[i * n + (k + 1 + j)] * x[j];
            }
            for j in 0..m {
                q_accum[i * n + (k + 1 + j)] -= 2.0 * dot * x[j];
            }
        }
    }

    // QR反復 (Wilkinson シフト)
    let max_iter = 200 * n;
    for _ in 0..max_iter {
        // 収束判定: 最下部の副対角要素
        let sub_diag = if n >= 2 { h[(n - 1) * n + (n - 2)].abs() } else { 0.0 };
        if sub_diag < 1e-10 {
            break;
        }

        // Wilkinson シフト
        let shift = h[(n - 1) * n + (n - 1)];

        // シフト適用
        for i in 0..n {
            h[i * n + i] -= shift;
        }

        // QR分解 (Givens回転)
        let mut g_list: Vec<(usize, f32, f32)> = Vec::new();
        for i in 0..(n - 1) {
            let a_val = h[i * n + i];
            let b_val = h[(i + 1) * n + i];
            let r = (a_val * a_val + b_val * b_val).sqrt();
            if r < 1e-15 {
                continue;
            }
            let c = a_val / r;
            let s = b_val / r;
            g_list.push((i, c, s));

            // 左から Givens 回転
            for j in 0..n {
                let t1 = h[i * n + j];
                let t2 = h[(i + 1) * n + j];
                h[i * n + j] = c * t1 + s * t2;
                h[(i + 1) * n + j] = -s * t1 + c * t2;
            }
        }

        // 右から Givens 回転 (R * Q)
        for &(i, c, s) in &g_list {
            for j in 0..n {
                let t1 = h[j * n + i];
                let t2 = h[j * n + (i + 1)];
                h[j * n + i] = c * t1 + s * t2;
                h[j * n + (i + 1)] = -s * t1 + c * t2;
            }
        }

        // Q の累積
        for &(i, c, s) in &g_list {
            for j in 0..n {
                let t1 = q_accum[j * n + i];
                let t2 = q_accum[j * n + (i + 1)];
                q_accum[j * n + i] = c * t1 + s * t2;
                q_accum[j * n + (i + 1)] = -s * t1 + c * t2;
            }
        }

        // シフト解除
        for i in 0..n {
            h[i * n + i] += shift;
        }
    }

    let eigenvalues: Vec<f32> = (0..n).map(|i| h[i * n + i]).collect();
    (eigenvalues, q_accum)
}

#[unsafe(no_mangle)]
/// @ffi_sig () -> void
/// メモリプールとOSアロケータの未使用キャッシュを強制解放する — バックグラウンドで自動的に実行
pub extern "C" fn tl_device_mem_purge() {
    // 1. CPUメモリ(malloc)・ゾーンアロケータのプレッシャー解放
    crate::tensor_ops_ext::tl_mem_purge();

    // 2. GPUバックエンド特有のプールパージ
    #[cfg(target_os = "macos")]
    {
        tl_metal::pool_purge();
    }
}
