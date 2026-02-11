//! CpuDevice: IDevice トレイトの CPU 実装
//!
//! 既存の `ffi::tl_cpu_tensor_*` 関数に委譲する。
//! 全メソッドは void* と CpuTensor のキャストで橋渡しする。

use tl_backend::IDevice;
use crate::ffi;
use crate::tensor::CpuTensor;
use std::ffi::c_void;

/// CPU デバイス (ゼロサイズ型)
pub struct CpuDevice;

/// void* → *mut CpuTensor キャスト
#[inline(always)]
fn t(p: *mut c_void) -> *mut CpuTensor { p as *mut CpuTensor }

/// *mut CpuTensor → void* キャスト
#[inline(always)]
fn v(p: *mut CpuTensor) -> *mut c_void { p as *mut c_void }

impl IDevice for CpuDevice {
    // ========== テンソル作成 ==========
    #[inline] fn tensor_new(&self, data: *const f32, rank: usize, shape: *const usize) -> *mut c_void { v(ffi::tl_cpu_tensor_new(data, rank, shape)) }
    #[inline] fn tensor_new_i64(&self, data: *const i64, rank: usize, shape: *const usize) -> *mut c_void { v(ffi::tl_cpu_tensor_new_i64(data, rank, shape)) }
    #[inline] fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_from_i64(data, len)) }
    #[inline] fn tensor_zeros(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_zeros(rank, shape, req_grad)) }
    #[inline] fn tensor_ones(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_ones(rank, shape, req_grad)) }
    #[inline] fn tensor_randn_debug(&self, rank: usize, shape: *const usize, seed: u64, req_grad: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_randn_debug(rank, shape, seed, req_grad)) }
    #[inline] fn tensor_new_causal_mask(&self, size: usize) -> *mut c_void { v(ffi::tl_cpu_tensor_new_causal_mask(size)) }

    // ========== メモリ管理 ==========
    #[inline] fn tensor_clone(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_clone(t(a))) }
    #[inline] fn tensor_free(&self, a: *mut c_void) { ffi::tl_cpu_tensor_free(t(a)) }
    #[inline] fn tensor_release(&self, a: *mut c_void) { ffi::tl_cpu_tensor_release(t(a)) }
    #[inline] fn tensor_acquire(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_acquire(t(a))) }
    #[inline] fn tensor_release_safe(&self, a: *mut c_void) { ffi::tl_cpu_tensor_release(t(a)) }
    #[inline] fn tensor_promote(&self, a: *mut c_void) { ffi::tl_cpu_tensor_promote(t(a)) }
    #[inline] fn tensor_register(&self, a: *mut c_void) { ffi::tl_cpu_tensor_register(t(a)) }
    #[inline] fn tensor_prepare_return(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_prepare_return(t(a))) }

    // ========== テンソル情報 ==========
    #[inline] fn tensor_len(&self, a: *mut c_void) -> usize { ffi::tl_cpu_tensor_len(t(a)) }
    #[inline] fn tensor_dim(&self, a: *mut c_void, dim: usize) -> usize { ffi::tl_cpu_tensor_dim(t(a), dim) }
    #[inline] fn tensor_numel(&self, a: *mut c_void) -> i64 { ffi::tl_cpu_tensor_numel(t(a)) }
    #[inline] fn tensor_data(&self, a: *mut c_void) -> *const f32 { ffi::tl_cpu_tensor_data(t(a)) }
    #[inline] fn tensor_device_id(&self, a: *mut c_void) -> i32 { ffi::tl_cpu_tensor_device_id(t(a)) }
    #[inline] fn tensor_get_shape(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_get_shape(t(a))) }

    // ========== 要素アクセス ==========
    #[inline] fn tensor_get(&self, a: *mut c_void, idx: i64) -> f32 { ffi::tl_cpu_tensor_get(t(a), idx) }
    #[inline] fn tensor_get_f32_md(&self, a: *mut c_void, indices: *const i64, rank: i64) -> f32 { ffi::tl_cpu_tensor_get_f32_md(t(a), indices, rank) }
    #[inline] fn tensor_get_i64_md(&self, a: *mut c_void, indices: *const i64, rank: i64) -> i64 { ffi::tl_cpu_tensor_get_i64_md(t(a), indices, rank) }
    #[inline] fn tensor_set_f32_md(&self, a: *mut c_void, indices: *const i64, rank: usize, value: f32) -> *mut c_void { v(ffi::tl_cpu_tensor_set_f32_md(t(a), indices, rank, value)) }
    #[inline] fn tensor_item(&self, a: *mut c_void) -> f32 { ffi::tl_cpu_tensor_item(t(a)) }
    #[inline] fn tensor_item_i64(&self, a: *mut c_void) -> i64 { ffi::tl_cpu_tensor_item_i64(t(a)) }

    // ========== 二項演算 ==========
    #[inline] fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_add(t(a), t(b))) }
    #[inline] fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sub(t(a), t(b))) }
    #[inline] fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_mul(t(a), t(b))) }
    #[inline] fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_div(t(a), t(b))) }
    #[inline] fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_rem(t(a), t(b))) }
    #[inline] fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_matmul(t(a), t(b))) }
    #[inline] fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_pow(t(a), t(b))) }
    #[inline] fn tensor_cross_entropy(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_cross_entropy(t(a), t(b))) }

    // ========== 単項演算 ==========
    #[inline] fn tensor_neg(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_neg(t(a))) }
    #[inline] fn tensor_abs(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_abs(t(a))) }
    #[inline] fn tensor_contiguous(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_contiguous(t(a))) }

    // ========== 比較演算 ==========
    #[inline] fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_eq(t(a), t(b))) }
    #[inline] fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_neq(t(a), t(b))) }
    #[inline] fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_gt(t(a), t(b))) }
    #[inline] fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_lt(t(a), t(b))) }
    #[inline] fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_ge(t(a), t(b))) }
    #[inline] fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_le(t(a), t(b))) }

    // ========== スカラー演算 ==========
    #[inline] fn tensor_add_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_add_scalar(t(a), s)) }
    #[inline] fn tensor_sub_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_sub_scalar(t(a), s)) }
    #[inline] fn tensor_mul_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_mul_scalar(t(a), s)) }
    #[inline] fn tensor_div_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_div_scalar(t(a), s)) }
    #[inline] fn tensor_pow_scalar(&self, a: *mut c_void, exp: f32) -> *mut c_void { v(ffi::tl_cpu_tensor_pow_scalar(t(a), exp)) }
    #[inline] fn tensor_scale(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_scale(t(a), s)) }

    // ========== インプレース演算 ==========
    #[inline] fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void) { ffi::tl_cpu_tensor_add_assign(t(a), t(b)) }
    #[inline] fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void) { ffi::tl_cpu_tensor_sub_assign(t(a), t(b)) }
    #[inline] fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void) { ffi::tl_cpu_tensor_mul_assign(t(a), t(b)) }
    #[inline] fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void) { ffi::tl_cpu_tensor_div_assign(t(a), t(b)) }
    #[inline] fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void) { ffi::tl_cpu_tensor_mod_assign(t(a), t(b)) }
    #[inline] fn tensor_add_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi::tl_cpu_tensor_add_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_sub_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi::tl_cpu_tensor_sub_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_mul_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi::tl_cpu_tensor_mul_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_div_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi::tl_cpu_tensor_div_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_mod_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi::tl_cpu_tensor_mod_assign_scalar_f32(t(a), s) }

    // ========== 数学・活性化関数 ==========
    #[inline] fn tensor_exp(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_exp(t(a))) }
    #[inline] fn tensor_log(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_log(t(a))) }
    #[inline] fn tensor_sqrt(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sqrt(t(a))) }
    #[inline] fn tensor_sin(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sin(t(a))) }
    #[inline] fn tensor_cos(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_cos(t(a))) }
    #[inline] fn tensor_tan(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_tan(t(a))) }
    #[inline] fn tensor_tanh(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_tanh(t(a))) }
    #[inline] fn tensor_sigmoid(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sigmoid(t(a))) }
    #[inline] fn tensor_relu(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_relu(t(a))) }
    #[inline] fn tensor_gelu(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_gelu(t(a))) }
    #[inline] fn tensor_silu(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_silu(t(a))) }

    // ========== Reduction ==========
    #[inline] fn tensor_sum(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sum(t(a))) }
    #[inline] fn tensor_mean(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_mean(t(a))) }
    #[inline] fn tensor_max(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_max(t(a))) }
    #[inline] fn tensor_min(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_min(t(a))) }
    #[inline] fn tensor_softmax(&self, a: *mut c_void, dim: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_softmax(t(a), dim)) }
    #[inline] fn tensor_max_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_max_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_min_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_min_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_mean_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_mean_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_sum_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_sum_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_argmax(&self, a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_argmax(t(a), dim, keep_dim)) }
    #[inline] fn tensor_argmin(&self, a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_argmin(t(a), dim, keep_dim)) }
    #[inline] fn tensor_tril(&self, a: *mut c_void, diagonal: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_tril(t(a), diagonal)) }
    #[inline] fn tensor_clamp(&self, a: *mut c_void, min: f64, max: f64) -> *mut c_void { v(ffi::tl_cpu_tensor_clamp(t(a), min, max)) }
    #[inline] fn tensor_sample(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_sample(t(a))) }

    // ========== Autograd ==========
    #[inline] fn tensor_backward(&self, a: *mut c_void) { ffi::tl_cpu_tensor_backward(t(a)) }
    #[inline] fn tensor_grad(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_grad(t(a))) }
    #[inline] fn tensor_detach(&self, a: *mut c_void, req_grad: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_detach(t(a), req_grad)) }
    #[inline] fn tensor_enable_grad(&self, a: *mut c_void) { ffi::tl_cpu_tensor_enable_grad(t(a)) }
    #[inline] fn clear_grads(&self) { ffi::tl_cpu_clear_grads() }

    // ========== 形状操作 ==========
    #[inline] fn tensor_reshape_new(&self, a: *mut c_void, s: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_reshape_new(t(a), t(s))) }
    #[inline] fn tensor_reshape_dims(&self, a: *mut c_void, d1: i64, d2: i64, d3: i64, d4: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_reshape_dims(t(a), d1, d2, d3, d4)) }
    #[inline] fn tensor_transpose(&self, a: *mut c_void, dim0: usize, dim1: usize) -> *mut c_void { v(ffi::tl_cpu_tensor_transpose(t(a), dim0, dim1)) }
    #[inline] fn tensor_slice(&self, a: *mut c_void, dim: i64, start: i64, len: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_slice(t(a), dim, start, len)) }
    #[inline] fn tensor_narrow(&self, a: *mut c_void, dim: usize, start: usize, len: usize) -> *mut c_void { v(ffi::tl_cpu_tensor_narrow(t(a), dim, start, len)) }
    #[inline] fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_cat(t(a), t(b), dim)) }
    #[inline] fn tensor_cat_i64(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_cat_i64(t(a), t(b), dim)) }
    #[inline] fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_cat2(t(a), t(b), dim)) }
    #[inline] fn tensor_cat_4d(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_cat_4d(t(a), t(b), dim)) }
    #[inline] fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void) { ffi::tl_cpu_tensor_replace_data(t(dst), t(src)) }
    #[inline] fn tensor_repeat_interleave(&self, a: *mut c_void, repeats: usize, dim: usize) -> *mut c_void { v(ffi::tl_cpu_tensor_repeat_interleave(t(a), repeats, dim)) }
    #[inline] fn tensor_to_device(&self, a: *mut c_void, device_id: i32) -> *mut c_void { v(ffi::tl_cpu_tensor_to_device(t(a), device_id)) }
    #[inline] fn tensor_to_f32(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_to_f32(t(a))) }
    #[inline] fn tensor_to_i64(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_to_i64(t(a))) }
    #[inline] fn tensor_embedding(&self, w: *mut c_void, idx: *mut c_void, pad: i64, sg: bool, sp: bool) -> *mut c_void { v(ffi::tl_cpu_tensor_embedding(t(w), t(idx), pad, sg, sp)) }

    // ========== LLM ==========
    #[inline] fn tensor_rms_norm(&self, a: *mut c_void, w: *mut c_void, eps: f32) -> *mut c_void { v(ffi::tl_cpu_tensor_rms_norm(t(a), t(w), eps)) }
    #[inline] fn tensor_rope_new_cos(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void { v(ffi::tl_cpu_tensor_rope_new_cos(seq_len, dim, base)) }
    #[inline] fn tensor_rope_new_sin(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void { v(ffi::tl_cpu_tensor_rope_new_sin(seq_len, dim, base)) }
    #[inline] fn tensor_apply_rope(&self, a: *mut c_void, cos: *mut c_void, sin: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_apply_rope(t(a), t(cos), t(sin))) }

    // ========== IO / Print ==========
    #[inline] fn tensor_print(&self, a: *mut c_void) { ffi::tl_cpu_tensor_print(t(a)) }
    #[inline] fn tensor_display(&self, a: *mut c_void) { ffi::tl_cpu_tensor_print(t(a)) }
    #[inline] fn tensor_print_1(&self, a: *mut c_void) { ffi::tl_cpu_tensor_print_1(t(a)) }
    #[inline] fn tensor_print_2(&self, a: *mut c_void) { ffi::tl_cpu_tensor_print_2(t(a)) }
    #[inline] fn tensor_print_3(&self, a: *mut c_void) { ffi::tl_cpu_tensor_print_3(t(a)) }
    #[inline] fn tensor_save(&self, a: *mut c_void, path: *const i8) { ffi::tl_cpu_tensor_save(t(a), path) }
    #[inline] fn tensor_load(&self, path: *const i8) -> *mut c_void { v(ffi::tl_cpu_tensor_load(path)) }

    // ========== NN ==========
    #[inline] fn tensor_conv2d(&self, input: *mut c_void, weight: *mut c_void, _bias: *mut c_void, stride: usize, padding: usize, _dilation: usize, _groups: usize) -> *mut c_void {
        // CPU版: ffi は (input, weight, padding, stride) の4引数
        v(ffi::tl_cpu_tensor_conv2d(t(input), t(weight), padding as i64, stride as i64))
    }

    // ========== CPU 専用 ==========
    #[inline] fn tensor_transpose_2d(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_transpose_2d(t(a))) }
    #[inline] fn tensor_reshape_2d(&self, a: *mut c_void, d0: i64, d1: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_reshape_2d(t(a), d0, d1)) }
    #[inline] fn tensor_reshape_3d_to_2d(&self, a: *mut c_void, d0: i64, d1: i64) -> *mut c_void { v(ffi::tl_cpu_tensor_reshape_3d_to_2d(t(a), d0, d1)) }
    #[inline] fn tensor_matmul_4d(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_matmul_4d(t(a), t(b))) }
    #[inline] fn tensor_add_4d(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_add_4d(t(a), t(b))) }
    #[inline] fn tensor_silu_4d(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_cpu_tensor_silu_4d(t(a))) }
}
