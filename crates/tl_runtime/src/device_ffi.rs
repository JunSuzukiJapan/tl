//! デバイス統一 FFI ラッパー
//!
//! IDevice トレイトを通じて、CPU/GPU の切替を一元管理する。
//! `builtins.rs` からはこのモジュールの `tl_device_*` 関数をマッピングすればよい。

use tl_backend::IDevice;
use tl_cpu::device_impl::CpuDevice;
use tl_metal::device_impl::MetalDeviceImpl;
use std::ffi::c_void;
use std::sync::OnceLock;

/// TL_DEVICE 環境変数のキャッシュ
#[inline]
fn is_cpu() -> bool {
    static IS_CPU: OnceLock<bool> = OnceLock::new();
    *IS_CPU.get_or_init(|| {
        std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu")
    })
}

/// 汎用ディスパッチ: CPU/GPU を切り替えてクロージャを実行
#[inline]
fn dispatch<F, R>(f: F) -> R
where
    F: FnOnce(&dyn IDevice) -> R,
{
    if is_cpu() {
        f(&CpuDevice)
    } else {
        f(&MetalDeviceImpl)
    }
}

// ========== テンソル作成 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut c_void { dispatch(|d| d.tensor_new(data, rank, shape)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_new_i64(data: *const i64, rank: usize, shape: *const usize) -> *mut c_void { dispatch(|d| d.tensor_new_i64(data, rank, shape)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_from_i64_array(data: *const i64, len: i64) -> *mut c_void { dispatch(|d| d.tensor_from_i64_array(data, len)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_zeros(rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { dispatch(|d| d.tensor_zeros(rank, shape, req_grad)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_ones(rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { dispatch(|d| d.tensor_ones(rank, shape, req_grad)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_randn_debug(rank: usize, shape: *const usize, seed: u64, req_grad: bool) -> *mut c_void { dispatch(|d| d.tensor_randn_debug(rank, shape, seed, req_grad)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_new_causal_mask(size: usize) -> *mut c_void { dispatch(|d| d.tensor_new_causal_mask(size)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_from_vec_u8(data: *mut c_void, len: i64) -> *mut c_void { dispatch(|d| d.tensor_from_vec_u8(data, len)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_from_u8_labels(data: *const u8, len: i64) -> *mut c_void { dispatch(|d| d.tensor_from_u8_labels(data, len)) }

// ========== メモリ管理 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_clone(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_clone(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_shallow_clone(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_shallow_clone(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_free(a: *mut c_void) { dispatch(|d| d.tensor_free(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_release(a: *mut c_void) { dispatch(|d| d.tensor_release(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_acquire(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_acquire(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_release_safe(a: *mut c_void) { dispatch(|d| d.tensor_release_safe(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_promote(a: *mut c_void) { dispatch(|d| d.tensor_promote(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_register(a: *mut c_void) { dispatch(|d| d.tensor_register(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_prepare_return(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_prepare_return(a)) }

// ========== テンソル情報 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_len(a: *mut c_void) -> usize { dispatch(|d| d.tensor_len(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_dim(a: *mut c_void, dim: usize) -> usize { dispatch(|d| d.tensor_dim(a, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_numel(a: *mut c_void) -> i64 { dispatch(|d| d.tensor_numel(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_data(a: *mut c_void) -> *const f32 { dispatch(|d| d.tensor_data(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_device_id(a: *mut c_void) -> i32 { dispatch(|d| d.tensor_device_id(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_get_shape(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_get_shape(a)) }

// ========== 要素アクセス ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_get(a: *mut c_void, idx: i64) -> f32 { dispatch(|d| d.tensor_get(a, idx)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_get_f32_md(a: *mut c_void, indices: *const i64, rank: i64) -> f32 { dispatch(|d| d.tensor_get_f32_md(a, indices, rank)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_get_i64_md(a: *mut c_void, indices: *const i64, rank: i64) -> i64 { dispatch(|d| d.tensor_get_i64_md(a, indices, rank)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_set_f32_md(a: *mut c_void, indices: *const i64, rank: usize, value: f32) -> *mut c_void { dispatch(|d| d.tensor_set_f32_md(a, indices, rank, value)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_item(a: *mut c_void) -> f32 { dispatch(|d| d.tensor_item(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_item_i64(a: *mut c_void) -> i64 { dispatch(|d| d.tensor_item_i64(a)) }

// ========== 二項演算 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_add(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_add(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sub(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_sub(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mul(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_mul(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_div(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_div(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_rem(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_rem(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_matmul(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_matmul(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_pow(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_pow(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cross_entropy(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_cross_entropy(a, b)) }

// ========== 単項演算 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_neg(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_neg(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_abs(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_abs(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_contiguous(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_contiguous(a)) }

// ========== 比較演算 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_eq(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_eq(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_neq(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_neq(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_gt(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_gt(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_lt(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_lt(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_ge(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_ge(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_le(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_le(a, b)) }

// ========== スカラー演算 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_add_scalar(a: *mut c_void, s: f64) -> *mut c_void { dispatch(|d| d.tensor_add_scalar(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sub_scalar(a: *mut c_void, s: f64) -> *mut c_void { dispatch(|d| d.tensor_sub_scalar(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mul_scalar(a: *mut c_void, s: f64) -> *mut c_void { dispatch(|d| d.tensor_mul_scalar(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_div_scalar(a: *mut c_void, s: f64) -> *mut c_void { dispatch(|d| d.tensor_div_scalar(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_pow_scalar(a: *mut c_void, exp: f32) -> *mut c_void { dispatch(|d| d.tensor_pow_scalar(a, exp)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_scale(a: *mut c_void, s: f32) -> *mut c_void { dispatch(|d| d.tensor_scale(a, s)) }

// ========== インプレース演算 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_add_assign(a: *mut c_void, b: *mut c_void) { dispatch(|d| d.tensor_add_assign(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sub_assign(a: *mut c_void, b: *mut c_void) { 
    eprintln!("[DEBUG device_ffi] tl_device_tensor_sub_assign a={:?} b={:?}", a, b);
    dispatch(|d| d.tensor_sub_assign(a, b)) 
}
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mul_assign(a: *mut c_void, b: *mut c_void) { dispatch(|d| d.tensor_mul_assign(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_div_assign(a: *mut c_void, b: *mut c_void) { dispatch(|d| d.tensor_div_assign(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mod_assign(a: *mut c_void, b: *mut c_void) { dispatch(|d| d.tensor_mod_assign(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_add_assign_scalar_f32(a: *mut c_void, s: f32) { dispatch(|d| d.tensor_add_assign_scalar_f32(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sub_assign_scalar_f32(a: *mut c_void, s: f32) { dispatch(|d| d.tensor_sub_assign_scalar_f32(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mul_assign_scalar_f32(a: *mut c_void, s: f32) { dispatch(|d| d.tensor_mul_assign_scalar_f32(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_div_assign_scalar_f32(a: *mut c_void, s: f32) { dispatch(|d| d.tensor_div_assign_scalar_f32(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mod_assign_scalar_f32(a: *mut c_void, s: f32) { dispatch(|d| d.tensor_mod_assign_scalar_f32(a, s)) }

// ========== 数学・活性化関数 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_exp(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_exp(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_log(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_log(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sqrt(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_sqrt(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sin(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_sin(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cos(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_cos(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_tan(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_tan(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_tanh(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_tanh(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sigmoid(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_sigmoid(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_relu(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_relu(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_gelu(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_gelu(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_silu(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_silu(a)) }

// ========== Reduction ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sum(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_sum(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mean(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_mean(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_max(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_max(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_min(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_min(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_softmax(a: *mut c_void, dim: i64) -> *mut c_void { dispatch(|d| d.tensor_softmax(a, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_max_dim(a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_max_dim(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_min_dim(a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_min_dim(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_mean_dim(a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_mean_dim(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sum_dim(a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_sum_dim(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_argmax(a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_argmax(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_argmin(a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { dispatch(|d| d.tensor_argmin(a, dim, keep_dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_tril(a: *mut c_void, diagonal: i64) -> *mut c_void { dispatch(|d| d.tensor_tril(a, diagonal)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_clamp(a: *mut c_void, min: f64, max: f64) -> *mut c_void { dispatch(|d| d.tensor_clamp(a, min, max)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_sample(a: *mut c_void, temp: f32, top_p: f32) -> *mut c_void { dispatch(|d| d.tensor_sample(a, temp, top_p)) }

// ========== Autograd ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_backward(a: *mut c_void) { dispatch(|d| d.tensor_backward(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_grad(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_grad(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_detach(a: *mut c_void, req_grad: bool) -> *mut c_void { dispatch(|d| d.tensor_detach(a, req_grad)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_enable_grad(a: *mut c_void) { dispatch(|d| d.tensor_enable_grad(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_clear_grads() { dispatch(|d| d.clear_grads()) }

// ========== 形状操作 ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_reshape_new(a: *mut c_void, s: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_reshape_new(a, s)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_reshape_dims(a: *mut c_void, d1: i64, d2: i64, d3: i64, d4: i64) -> *mut c_void { dispatch(|d| d.tensor_reshape_dims(a, d1, d2, d3, d4)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_transpose(a: *mut c_void, dim0: usize, dim1: usize) -> *mut c_void { dispatch(|d| d.tensor_transpose(a, dim0, dim1)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_slice(a: *mut c_void, dim: i64, start: i64, len: i64) -> *mut c_void { dispatch(|d| d.tensor_slice(a, dim, start, len)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_narrow(a: *mut c_void, dim: usize, start: usize, len: usize) -> *mut c_void { dispatch(|d| d.tensor_narrow(a, dim, start, len)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cat(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { dispatch(|d| d.tensor_cat(a, b, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cat_i64(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { dispatch(|d| d.tensor_cat_i64(a, b, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cat2(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { dispatch(|d| d.tensor_cat2(a, b, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_cat_4d(a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { dispatch(|d| d.tensor_cat_4d(a, b, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_replace_data(dst: *mut c_void, src: *mut c_void) { dispatch(|d| d.tensor_replace_data(dst, src)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_repeat_interleave(a: *mut c_void, repeats: usize, dim: usize) -> *mut c_void { dispatch(|d| d.tensor_repeat_interleave(a, repeats, dim)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_to_device(a: *mut c_void, device_id: i32) -> *mut c_void { dispatch(|d| d.tensor_to_device(a, device_id)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_to_f32(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_to_f32(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_to_i64(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_to_i64(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_embedding(w: *mut c_void, idx: *mut c_void, pad: i64, sg: bool, sp: bool) -> *mut c_void { dispatch(|d| d.tensor_embedding(w, idx, pad, sg, sp)) }

// ========== LLM ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_rms_norm(a: *mut c_void, w: *mut c_void, eps: f32) -> *mut c_void { dispatch(|d| d.tensor_rms_norm(a, w, eps)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_rope_new_cos(dim: usize, seq_len: usize, base: f32) -> *mut c_void { dispatch(|d| d.tensor_rope_new_cos(dim, seq_len, base)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_rope_new_sin(dim: usize, seq_len: usize, base: f32) -> *mut c_void { dispatch(|d| d.tensor_rope_new_sin(dim, seq_len, base)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_apply_rope(a: *mut c_void, cos: *mut c_void, sin: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_apply_rope(a, cos, sin)) }

// ========== IO / Print ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_print(a: *mut c_void) { dispatch(|d| d.tensor_print(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_display(a: *mut c_void) { dispatch(|d| d.tensor_display(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_print_1(a: *mut c_void) { dispatch(|d| d.tensor_print_1(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_print_2(a: *mut c_void) { dispatch(|d| d.tensor_print_2(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_print_3(a: *mut c_void) { dispatch(|d| d.tensor_print_3(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_save(a: *mut c_void, path: *const i8) { dispatch(|d| d.tensor_save(a, path)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_load(path: *const i8) -> *mut c_void { dispatch(|d| d.tensor_load(path)) }

// ========== NN ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_conv2d(input: *mut c_void, weight: *mut c_void, bias: *mut c_void, stride: usize, padding: usize, dilation: usize, groups: usize) -> *mut c_void { dispatch(|d| d.tensor_conv2d(input, weight, bias, stride, padding, dilation, groups)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_batch_norm(input: *mut c_void, running_mean: *mut c_void, running_var: *mut c_void, weight: *mut c_void, bias: *mut c_void, training: bool, momentum: f64, eps: f64) -> *mut c_void { dispatch(|d| d.tensor_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_layer_norm(input: *mut c_void, weight: *mut c_void, bias: *mut c_void, eps: f64) -> *mut c_void { dispatch(|d| d.tensor_layer_norm(input, weight, bias, eps)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_dropout(input: *mut c_void, p: f64, training: bool) -> *mut c_void { dispatch(|d| d.tensor_dropout(input, p, training)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_max_pool2d(input: *mut c_void, kernel_size: usize, stride: usize, padding: usize) -> *mut c_void { dispatch(|d| d.tensor_max_pool2d(input, kernel_size, stride, padding)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_avg_pool2d(input: *mut c_void, kernel_size: usize, stride: usize, padding: usize) -> *mut c_void { dispatch(|d| d.tensor_avg_pool2d(input, kernel_size, stride, padding)) }

// ========== CPU 専用 (device_ffi 経由) ==========
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_transpose_2d(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_transpose_2d(a)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_reshape_2d(a: *mut c_void, d0: i64, d1: i64) -> *mut c_void { dispatch(|d| d.tensor_reshape_2d(a, d0, d1)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_reshape_3d_to_2d(a: *mut c_void, d0: i64, d1: i64) -> *mut c_void { dispatch(|d| d.tensor_reshape_3d_to_2d(a, d0, d1)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_matmul_4d(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_matmul_4d(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_add_4d(a: *mut c_void, b: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_add_4d(a, b)) }
#[unsafe(no_mangle)] pub extern "C" fn tl_device_tensor_silu_4d(a: *mut c_void) -> *mut c_void { dispatch(|d| d.tensor_silu_4d(a)) }

// Helper: Convert tensor shape to Vec<i64> for JIT
#[repr(C)]
struct JitVec {
    ptr: *mut i64,
    cap: i64,
    len: i64,
}

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
