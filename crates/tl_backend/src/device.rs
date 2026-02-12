//! IDevice トレイト定義
//!
//! CPU と GPU (Metal/CUDA) の FFI 層を統一するデバイス抽象化。
//! JIT 生成コードは void* でテンソルを扱い、IDevice の実装が
//! 適切な具象型にキャストして処理する。

use std::ffi::c_void;

/// デバイス抽象化トレイト
///
/// 各デバイス (CPU, Metal, CUDA) は、このトレイトを実装し、
/// `extern "C"` FFI 関数と同じセマンティクスのメソッドを提供する。
/// 全メソッドは void* (c_void) でテンソルを受け渡しする。
///
/// 命名規則: tensor_<op> — Rust の camelCase ではなく FFI のスネークケースに合わせる
pub trait IDevice {
    // ========== テンソル作成 ==========
    fn tensor_new(&self, data: *const f32, rank: usize, shape: *const usize) -> *mut c_void;
    fn tensor_new_i64(&self, data: *const i64, rank: usize, shape: *const usize) -> *mut c_void;
    fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> *mut c_void;
    fn tensor_zeros(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void;
    fn tensor_ones(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void;
    fn tensor_randn_debug(&self, rank: usize, shape: *const usize, seed: u64, req_grad: bool) -> *mut c_void;
    fn tensor_new_causal_mask(&self, size: usize) -> *mut c_void;
    fn tensor_from_vec_u8(&self, data: *mut c_void, len: i64) -> *mut c_void;
    fn tensor_from_u8_labels(&self, data: *const u8, len: i64) -> *mut c_void;

    // ========== メモリ管理 ==========
    fn tensor_clone(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_free(&self, t: *mut c_void);
    fn tensor_release(&self, t: *mut c_void);
    fn tensor_acquire(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_release_safe(&self, t: *mut c_void);
    fn tensor_promote(&self, t: *mut c_void);
    fn tensor_register(&self, t: *mut c_void);
    fn tensor_prepare_return(&self, t: *mut c_void) -> *mut c_void;

    // ========== テンソル情報 ==========
    fn tensor_len(&self, t: *mut c_void) -> usize;
    fn tensor_dim(&self, t: *mut c_void, dim: usize) -> usize;
    fn tensor_numel(&self, t: *mut c_void) -> i64;
    fn tensor_data(&self, t: *mut c_void) -> *const f32;
    fn tensor_device_id(&self, t: *mut c_void) -> i32;
    fn tensor_get_shape(&self, t: *mut c_void) -> *mut c_void;

    // ========== 要素アクセス ==========
    fn tensor_get(&self, t: *mut c_void, idx: i64) -> f32;
    fn tensor_get_f32_md(&self, t: *mut c_void, indices: *const i64, rank: i64) -> f32;
    fn tensor_get_i64_md(&self, t: *mut c_void, indices: *const i64, rank: i64) -> i64;
    fn tensor_set_f32_md(&self, t: *mut c_void, indices: *const i64, rank: usize, value: f32) -> *mut c_void;
    fn tensor_item(&self, t: *mut c_void) -> f32;
    fn tensor_item_i64(&self, t: *mut c_void) -> i64;

    // ========== 二項演算 ==========
    fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_cross_entropy(&self, logits: *mut c_void, labels: *mut c_void) -> *mut c_void;

    // ========== 単項演算 ==========
    fn tensor_neg(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_abs(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_contiguous(&self, t: *mut c_void) -> *mut c_void;

    // ========== 比較演算 ==========
    fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void;

    // ========== スカラー演算 ==========
    fn tensor_add_scalar(&self, t: *mut c_void, s: f64) -> *mut c_void;
    fn tensor_sub_scalar(&self, t: *mut c_void, s: f64) -> *mut c_void;
    fn tensor_mul_scalar(&self, t: *mut c_void, s: f64) -> *mut c_void;
    fn tensor_div_scalar(&self, t: *mut c_void, s: f64) -> *mut c_void;
    fn tensor_pow_scalar(&self, t: *mut c_void, exp: f32) -> *mut c_void;
    fn tensor_scale(&self, t: *mut c_void, s: f64) -> *mut c_void;

    // ========== インプレース演算 ==========
    fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void);
    fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void);
    fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void);
    fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void);
    fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void);
    fn tensor_add_assign_scalar_f32(&self, t: *mut c_void, s: f32);
    fn tensor_sub_assign_scalar_f32(&self, t: *mut c_void, s: f32);
    fn tensor_mul_assign_scalar_f32(&self, t: *mut c_void, s: f32);
    fn tensor_div_assign_scalar_f32(&self, t: *mut c_void, s: f32);
    fn tensor_mod_assign_scalar_f32(&self, t: *mut c_void, s: f32);

    // ========== 数学・活性化関数 ==========
    fn tensor_exp(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_log(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_sqrt(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_sin(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_cos(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_tan(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_tanh(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_sigmoid(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_relu(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_gelu(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_silu(&self, t: *mut c_void) -> *mut c_void;

    // ========== Reduction ==========
    fn tensor_sum(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_mean(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_max(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_min(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_softmax(&self, t: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_max_dim(&self, t: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void;
    fn tensor_min_dim(&self, t: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void;
    fn tensor_mean_dim(&self, t: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void;
    fn tensor_sum_dim(&self, t: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void;
    fn tensor_argmax(&self, t: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void;
    fn tensor_argmin(&self, t: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void;
    fn tensor_tril(&self, t: *mut c_void, diagonal: i64) -> *mut c_void;
    fn tensor_clamp(&self, t: *mut c_void, min: f64, max: f64) -> *mut c_void;
    fn tensor_sample(&self, t: *mut c_void) -> *mut c_void;

    // ========== Autograd ==========
    fn tensor_backward(&self, t: *mut c_void);
    fn tensor_grad(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_detach(&self, t: *mut c_void, req_grad: bool) -> *mut c_void;
    fn tensor_enable_grad(&self, t: *mut c_void);
    fn clear_grads(&self);

    // ========== 形状操作 ==========
    fn tensor_reshape_new(&self, t: *mut c_void, new_shape: *mut c_void) -> *mut c_void;
    fn tensor_reshape_dims(&self, t: *mut c_void, d1: i64, d2: i64, d3: i64, d4: i64) -> *mut c_void;
    fn tensor_transpose(&self, t: *mut c_void, dim0: usize, dim1: usize) -> *mut c_void;
    fn tensor_slice(&self, t: *mut c_void, dim: i64, start: i64, len: i64) -> *mut c_void;
    fn tensor_narrow(&self, t: *mut c_void, dim: usize, start: usize, len: usize) -> *mut c_void;
    fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_cat_i64(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_cat_4d(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void);
    fn tensor_repeat_interleave(&self, t: *mut c_void, repeats: usize, dim: usize) -> *mut c_void;
    fn tensor_to_device(&self, t: *mut c_void, device_id: i32) -> *mut c_void;
    fn tensor_to_f32(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_to_i64(&self, t: *mut c_void) -> *mut c_void;
    fn tensor_embedding(&self, weight: *mut c_void, indices: *mut c_void, padding_idx: i64, scale_grad: bool, sparse: bool) -> *mut c_void;

    // ========== LLM ==========
    fn tensor_rms_norm(&self, input: *mut c_void, weight: *mut c_void, eps: f32) -> *mut c_void;
    fn tensor_rope_new_cos(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void;
    fn tensor_rope_new_sin(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void;
    fn tensor_apply_rope(&self, t: *mut c_void, cos: *mut c_void, sin: *mut c_void) -> *mut c_void;

    // ========== IO / Print ==========
    fn tensor_print(&self, t: *mut c_void);
    fn tensor_display(&self, t: *mut c_void);
    fn tensor_print_1(&self, t: *mut c_void);
    fn tensor_print_2(&self, t: *mut c_void);
    fn tensor_print_3(&self, t: *mut c_void);
    fn tensor_save(&self, t: *mut c_void, path: *const i8);
    fn tensor_load(&self, path: *const i8) -> *mut c_void;

    // ========== NN ==========
    fn tensor_conv2d(&self, input: *mut c_void, weight: *mut c_void, bias: *mut c_void, stride: usize, padding: usize, dilation: usize, groups: usize) -> *mut c_void;

    // ========== CPU 専用 (デフォルト実装: GPU では呼ばれない) ==========
    fn tensor_transpose_2d(&self, t: *mut c_void) -> *mut c_void { let _ = t; std::ptr::null_mut() }
    fn tensor_reshape_2d(&self, t: *mut c_void, d0: i64, d1: i64) -> *mut c_void { let _ = (t, d0, d1); std::ptr::null_mut() }
    fn tensor_reshape_3d_to_2d(&self, t: *mut c_void, d0: i64, d1: i64) -> *mut c_void { let _ = (t, d0, d1); std::ptr::null_mut() }
    fn tensor_matmul_4d(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { let _ = (a, b); std::ptr::null_mut() }
    fn tensor_add_4d(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { let _ = (a, b); std::ptr::null_mut() }
    fn tensor_silu_4d(&self, t: *mut c_void) -> *mut c_void { let _ = t; std::ptr::null_mut() }
}
