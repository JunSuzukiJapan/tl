//! CUDA Backend FFI Operations
//!
//! tl_metal/src/ffi_ops.rs と同等のスタブ実装。

use crate::tensor::CudaTensor;
use crate::DType;
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

type OpaqueTensor = CudaTensor;

// === デバッグカウンタ ===
static MAKE_COUNT: AtomicUsize = AtomicUsize::new(0);
static RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);
static ACQUIRE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// デバッグ: カウンタをリセット
pub fn tl_cuda_debug_reset_counters() {
    MAKE_COUNT.store(0, Ordering::SeqCst);
    RELEASE_COUNT.store(0, Ordering::SeqCst);
    ACQUIRE_COUNT.store(0, Ordering::SeqCst);
}

pub fn make_tensor(_t: CudaTensor) -> *mut OpaqueTensor {
    unimplemented!("make_tensor")
}

pub fn release_if_live(_t: *mut OpaqueTensor) {
    unimplemented!("release_if_live")
}

pub fn acquire_tensor(_t: *mut OpaqueTensor) {
    unimplemented!("acquire_tensor")
}

// ========== テンソル作成 ==========

#[no_mangle] pub fn tl_cuda_new(_data: *const f32, _rank: usize, _shape: *const usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_new") }
#[no_mangle] pub fn tl_cuda_new_i64(_data: *const i64, _rank: usize, _shape: *const usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_new_i64") }
#[no_mangle] pub fn tl_cuda_zeros(_rank: usize, _shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_zeros") }
#[no_mangle] pub fn tl_cuda_ones(_rank: usize, _shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_ones") }
#[no_mangle] pub fn tl_cuda_randn(_rank: usize, _shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_randn") }
#[no_mangle] pub fn tl_cuda_randn_debug(_rank: usize, _shape: *const usize, _seed: u64, _req_grad: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_randn_debug") }
#[no_mangle] pub fn tl_cuda_from_i64(_data: *const i64, _len: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_from_i64") }
#[no_mangle] pub fn tl_cuda_from_u8(_data: *const u8, _len: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_from_u8") }
#[no_mangle] pub fn tl_cuda_from_vec_u8(_data: *mut std::ffi::c_void, _len: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_from_vec_u8") }
#[no_mangle] pub fn tl_cuda_from_u8_labels(_data: *const u8, _len: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_from_u8_labels") }
#[no_mangle] pub fn tl_cuda_from_i64_array(_data: *const i64, _len: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_from_i64_array") }

// ========== テンソル解放 ==========
#[no_mangle] pub fn tl_cuda_free(_t: *mut OpaqueTensor) { unimplemented!("tl_cuda_free") }

// ========== テンソル情報取得 ==========
#[no_mangle] pub fn tl_cuda_dim(_t: *mut OpaqueTensor, _dim: usize) -> usize { unimplemented!("tl_cuda_dim") }
#[no_mangle] pub fn tl_cuda_len(_t: *mut OpaqueTensor) -> usize { unimplemented!("tl_cuda_len") }
#[no_mangle] pub fn tl_cuda_shape(_t: *mut OpaqueTensor, _dim: usize) -> i64 { unimplemented!("tl_cuda_shape") }
#[no_mangle] pub fn tl_cuda_get_shape(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_get_shape") }

// ========== テンソルデータアクセス ==========
#[no_mangle] pub fn tl_cuda_get_f32(_t: *mut OpaqueTensor, _idx: usize) -> f32 { unimplemented!("tl_cuda_get_f32") }
#[no_mangle] pub fn tl_cuda_get_i64(_t: *mut OpaqueTensor, _idx: usize) -> i64 { unimplemented!("tl_cuda_get_i64") }
#[no_mangle] pub fn tl_cuda_set_f32(_t: *mut OpaqueTensor, _idx: usize, _val: f32) { unimplemented!("tl_cuda_set_f32") }
#[no_mangle] pub fn tl_cuda_item(_t: *mut OpaqueTensor) -> f32 { unimplemented!("tl_cuda_item") }
#[no_mangle] pub fn tl_cuda_item_i64(_t: *mut OpaqueTensor) -> i64 { unimplemented!("tl_cuda_item_i64") }
#[no_mangle] pub fn tl_cuda_get(_t: *mut OpaqueTensor, _idx: i64) -> f32 { unimplemented!("tl_cuda_get") }
#[no_mangle] pub fn tl_cuda_get_f32_md(_t: *mut OpaqueTensor, _idx0: i64, _idx1: i64) -> f32 { unimplemented!("tl_cuda_get_f32_md") }
#[no_mangle] pub fn tl_cuda_get_i64_md(_t: *mut OpaqueTensor, _idx0: i64, _idx1: i64) -> i64 { unimplemented!("tl_cuda_get_i64_md") }
#[no_mangle] pub fn tl_cuda_set_f32_md(_t: *mut OpaqueTensor, _indices: *const i64, _rank: usize, _value: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_set_f32_md") }

// ========== 基本演算 ==========
#[no_mangle] pub fn tl_cuda_add(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_add") }
#[no_mangle] pub fn tl_cuda_sub(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sub") }
#[no_mangle] pub fn tl_cuda_mul(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_mul") }
#[no_mangle] pub fn tl_cuda_div(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_div") }
#[no_mangle] pub fn tl_cuda_rem(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_rem") }
#[no_mangle] pub fn tl_cuda_neg(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_neg") }
#[no_mangle] pub fn tl_cuda_abs(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_abs") }
#[no_mangle] pub fn tl_cuda_matmul(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_matmul") }

// ========== 活性化関数 ==========
#[no_mangle] pub fn tl_cuda_relu(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_relu") }
#[no_mangle] pub fn tl_cuda_sigmoid(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sigmoid") }
#[no_mangle] pub fn tl_cuda_tanh(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_tanh") }
#[no_mangle] pub fn tl_cuda_gelu(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_gelu") }
#[no_mangle] pub fn tl_cuda_silu(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_silu") }
#[no_mangle] pub fn tl_cuda_exp(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_exp") }
#[no_mangle] pub fn tl_cuda_log(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_log") }
#[no_mangle] pub fn tl_cuda_sin(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sin") }
#[no_mangle] pub fn tl_cuda_cos(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_cos") }
#[no_mangle] pub fn tl_cuda_tan(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_tan") }
#[no_mangle] pub fn tl_cuda_sqrt(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sqrt") }

// ========== スカラー演算 ==========
#[no_mangle] pub fn tl_cuda_add_scalar(_t: *mut OpaqueTensor, _s: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_add_scalar") }
#[no_mangle] pub fn tl_cuda_mul_scalar(_t: *mut OpaqueTensor, _s: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_mul_scalar") }
#[no_mangle] pub fn tl_cuda_sub_scalar(_t: *mut OpaqueTensor, _s: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sub_scalar") }
#[no_mangle] pub fn tl_cuda_div_scalar(_t: *mut OpaqueTensor, _s: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_div_scalar") }
#[no_mangle] pub fn tl_cuda_pow_scalar(_t: *mut OpaqueTensor, _s: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_pow_scalar") }
#[no_mangle] pub fn tl_cuda_scale(_t: *mut OpaqueTensor, _s: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_scale") }
#[no_mangle] pub fn tl_cuda_clamp(_t: *mut OpaqueTensor, _min: f64, _max: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_clamp") }

// ========== リダクション ==========
#[no_mangle] pub fn tl_cuda_sum(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sum") }
#[no_mangle] pub fn tl_cuda_mean(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_mean") }
#[no_mangle] pub fn tl_cuda_max(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_max") }
#[no_mangle] pub fn tl_cuda_min(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_min") }
#[no_mangle] pub fn tl_cuda_sum_dim(_t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sum_dim") }
#[no_mangle] pub fn tl_cuda_mean_dim(_t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_mean_dim") }
#[no_mangle] pub fn tl_cuda_max_dim(_t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_max_dim") }
#[no_mangle] pub fn tl_cuda_min_dim(_t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_min_dim") }
#[no_mangle] pub fn tl_cuda_argmax(_t: *mut OpaqueTensor, _dim: i64, _keepdim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_argmax") }
#[no_mangle] pub fn tl_cuda_argmin(_t: *mut OpaqueTensor, _dim: i64, _keepdim: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_argmin") }
#[no_mangle] pub fn tl_cuda_softmax(_t: *mut OpaqueTensor, _dim: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_softmax") }

// ========== 型変換 ==========
#[no_mangle] pub fn tl_cuda_to_f32(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_to_f32") }
#[no_mangle] pub fn tl_cuda_to_i64(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_to_i64") }
#[no_mangle] pub fn tl_cuda_to_device(_t: *mut OpaqueTensor, _device_id: i32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_to_device") }

// ========== 形状操作 ==========
#[no_mangle] pub fn tl_cuda_reshape(_t: *mut OpaqueTensor, _dims: *const i64, _num_dims: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_reshape") }
#[no_mangle] pub fn tl_cuda_reshape_new(_t: *mut OpaqueTensor, _dims: *const i64, _num_dims: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_reshape_new") }
#[no_mangle] pub fn tl_cuda_reshape_dims(_t: *mut OpaqueTensor, _dims: *const i64, _num_dims: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_reshape_dims") }
#[no_mangle] pub fn tl_cuda_transpose(_t: *mut OpaqueTensor, _dim0: usize, _dim1: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_transpose") }
#[no_mangle] pub fn tl_cuda_contiguous(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_contiguous") }
#[no_mangle] pub fn tl_cuda_narrow(_t: *mut OpaqueTensor, _dim: usize, _start: usize, _len: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_narrow") }
#[no_mangle] pub fn tl_cuda_slice(_t: *mut OpaqueTensor, _dim: usize, _start: usize, _len: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_slice") }
#[no_mangle] pub fn tl_cuda_cat(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor, _dim: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_cat") }
#[no_mangle] pub fn tl_cuda_cat_i64(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor, _dim: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_cat_i64") }
#[no_mangle] pub fn tl_cuda_tril(_t: *mut OpaqueTensor, _diagonal: i64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_tril") }
#[no_mangle] pub fn tl_cuda_repeat_interleave(_t: *mut OpaqueTensor, _repeats: usize, _dim: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_repeat_interleave") }
#[no_mangle] pub fn tl_cuda_sample(_t: *mut OpaqueTensor, _temp: f32, _top_p: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_sample") }

// ========== 畳み込み / NN ==========
#[no_mangle] pub fn tl_cuda_conv2d(_input: *mut OpaqueTensor, _weight: *mut OpaqueTensor, _bias: *mut OpaqueTensor, _stride: usize, _padding: usize, _dilation: usize, _groups: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_conv2d") }
#[no_mangle] pub fn tl_cuda_batch_norm(_input: *mut OpaqueTensor, _running_mean: *mut OpaqueTensor, _running_var: *mut OpaqueTensor, _weight: *mut OpaqueTensor, _bias: *mut OpaqueTensor, _training: bool, _momentum: f64, _eps: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_batch_norm") }
#[no_mangle] pub fn tl_cuda_layer_norm(_input: *mut OpaqueTensor, _weight: *mut OpaqueTensor, _bias: *mut OpaqueTensor, _eps: f64) -> *mut OpaqueTensor { unimplemented!("tl_cuda_layer_norm") }
#[no_mangle] pub fn tl_cuda_max_pool2d(_input: *mut OpaqueTensor, _kernel_size: usize, _stride: usize, _padding: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_max_pool2d") }
#[no_mangle] pub fn tl_cuda_avg_pool2d(_input: *mut OpaqueTensor, _kernel_size: usize, _stride: usize, _padding: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_avg_pool2d") }
#[no_mangle] pub fn tl_cuda_dropout(_input: *mut OpaqueTensor, _p: f64, _training: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_dropout") }
#[no_mangle] pub fn tl_cuda_embedding(_weight: *mut OpaqueTensor, _indices: *mut OpaqueTensor, _padding_idx: i64, _scale_grad_by_freq: bool, _sparse: bool) -> *mut OpaqueTensor { unimplemented!("tl_cuda_embedding") }
#[no_mangle] pub fn tl_cuda_cross_entropy(_logits: *mut OpaqueTensor, _labels: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_cross_entropy") }
#[no_mangle] pub fn tl_cuda_rms_norm(_input: *mut OpaqueTensor, _weight: *mut OpaqueTensor, _eps: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_rms_norm") }

// ========== 比較演算 ==========
#[no_mangle] pub fn tl_cuda_eq(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_eq") }
#[no_mangle] pub fn tl_cuda_neq(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_neq") }
#[no_mangle] pub fn tl_cuda_lt(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_lt") }
#[no_mangle] pub fn tl_cuda_le(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_le") }
#[no_mangle] pub fn tl_cuda_gt(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_gt") }
#[no_mangle] pub fn tl_cuda_ge(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_ge") }

// ========== インプレース演算 ==========
#[no_mangle] pub fn tl_cuda_add_assign(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_add_assign") }
#[no_mangle] pub fn tl_cuda_sub_assign(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_sub_assign") }
#[no_mangle] pub fn tl_cuda_mul_assign(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_mul_assign") }
#[no_mangle] pub fn tl_cuda_div_assign(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_div_assign") }
#[no_mangle] pub fn tl_cuda_mod_assign(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_mod_assign") }

// ========== スカラー In-place 演算 ==========
#[no_mangle] pub fn tl_cuda_add_assign_scalar_f32(_t: *mut OpaqueTensor, _s: f32) { unimplemented!("tl_cuda_add_assign_scalar_f32") }
#[no_mangle] pub fn tl_cuda_sub_assign_scalar_f32(_t: *mut OpaqueTensor, _s: f32) { unimplemented!("tl_cuda_sub_assign_scalar_f32") }
#[no_mangle] pub fn tl_cuda_mul_assign_scalar_f32(_t: *mut OpaqueTensor, _s: f32) { unimplemented!("tl_cuda_mul_assign_scalar_f32") }
#[no_mangle] pub fn tl_cuda_div_assign_scalar_f32(_t: *mut OpaqueTensor, _s: f32) { unimplemented!("tl_cuda_div_assign_scalar_f32") }
#[no_mangle] pub fn tl_cuda_mod_assign_scalar_f32(_t: *mut OpaqueTensor, _s: f32) { unimplemented!("tl_cuda_mod_assign_scalar_f32") }

// ========== RoPE ==========
#[no_mangle] pub fn tl_cuda_rope_new_cos(_dim: usize, _seq_len: usize, _freq_base: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_rope_new_cos") }
#[no_mangle] pub fn tl_cuda_rope_new_sin(_dim: usize, _seq_len: usize, _freq_base: f32) -> *mut OpaqueTensor { unimplemented!("tl_cuda_rope_new_sin") }
#[no_mangle] pub fn tl_cuda_apply_rope(_q: *mut OpaqueTensor, _k: *mut OpaqueTensor, _cos: *mut OpaqueTensor, _sin: *mut OpaqueTensor, _pos: usize) { unimplemented!("tl_cuda_apply_rope") }

// ========== Mask ==========
#[no_mangle] pub fn tl_cuda_new_causal_mask(_size: usize) -> *mut OpaqueTensor { unimplemented!("tl_cuda_new_causal_mask") }

// ========== Device/Grad ==========
#[no_mangle] pub fn tl_cuda_device_id(_t: *mut OpaqueTensor) -> i32 { unimplemented!("tl_cuda_device_id") }
#[no_mangle] pub fn tl_cuda_backward(_t: *mut OpaqueTensor) { unimplemented!("tl_cuda_backward") }
#[no_mangle] pub fn tl_cuda_grad(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_grad") }
#[no_mangle] pub fn tl_cuda_detach(_t: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_detach") }
#[no_mangle] pub fn tl_cuda_enable_grad(_t: *mut OpaqueTensor) { unimplemented!("tl_cuda_enable_grad") }
#[no_mangle] pub fn tl_cuda_replace_data(_a: *mut OpaqueTensor, _b: *mut OpaqueTensor) { unimplemented!("tl_cuda_replace_data") }
#[no_mangle] pub fn tl_cuda_pow(_t: *mut OpaqueTensor, _exp: *mut OpaqueTensor) -> *mut OpaqueTensor { unimplemented!("tl_cuda_pow") }
