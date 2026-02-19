//! IDevice トレイトの CUDA 実装 (スタブ)
//!
//! tl_backend::IDevice の全メソッドを unimplemented!() で実装。

use tl_backend::device::{IDevice, BackendResult};
use std::ffi::c_void;

pub struct CudaDeviceImpl;

impl IDevice for CudaDeviceImpl {
    // ========== テンソル作成 ==========
    fn tensor_new(&self, _data: *const f32, _rank: usize, _shape: *const usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_new_i64(&self, _data: *const i64, _rank: usize, _shape: *const usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_from_i64_array(&self, _data: *const i64, _len: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_zeros(&self, _rank: usize, _shape: *const usize, _req_grad: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_ones(&self, _rank: usize, _shape: *const usize, _req_grad: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_randn_debug(&self, _rank: usize, _shape: *const usize, _seed: u64, _req_grad: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_new_causal_mask(&self, _size: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_from_vec_u8(&self, _data: *mut c_void, _len: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_from_u8_labels(&self, _data: *const u8, _len: i64) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== メモリ管理 ==========
    fn tensor_clone(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_shallow_clone(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_free(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_release(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_acquire(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_release_safe(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_promote(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_register(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_prepare_return(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== テンソル情報 ==========
    fn tensor_len(&self, _t: *mut c_void) -> BackendResult<usize> { unimplemented!() }
    fn tensor_dim(&self, _t: *mut c_void, _dim: usize) -> BackendResult<usize> { unimplemented!() }
    fn tensor_numel(&self, _t: *mut c_void) -> BackendResult<i64> { unimplemented!() }
    fn tensor_data(&self, _t: *mut c_void) -> BackendResult<*const f32> { unimplemented!() }
    fn tensor_device_id(&self, _t: *mut c_void) -> BackendResult<i32> { unimplemented!() }
    fn tensor_get_shape(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== 要素アクセス ==========
    fn tensor_get(&self, _t: *mut c_void, _idx: i64) -> BackendResult<f32> { unimplemented!() }
    fn tensor_get_f32_md(&self, _t: *mut c_void, _indices: *const i64, _rank: i64) -> BackendResult<f32> { unimplemented!() }
    fn tensor_get_i64_md(&self, _t: *mut c_void, _indices: *const i64, _rank: i64) -> BackendResult<i64> { unimplemented!() }
    fn tensor_set_f32_md(&self, _t: *mut c_void, _indices: *const i64, _rank: usize, _value: f32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_item(&self, _t: *mut c_void) -> BackendResult<f32> { unimplemented!() }
    fn tensor_item_i64(&self, _t: *mut c_void) -> BackendResult<i64> { unimplemented!() }

    // ========== 二項演算 ==========
    fn tensor_add(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sub(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_mul(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_div(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_rem(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_matmul(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_pow(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cross_entropy(&self, _logits: *mut c_void, _labels: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== 単項演算 ==========
    fn tensor_neg(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_abs(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_contiguous(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== 比較演算 ==========
    fn tensor_eq(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_neq(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_gt(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_lt(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_ge(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_le(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== スカラー演算 ==========
    fn tensor_add_scalar(&self, _t: *mut c_void, _s: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sub_scalar(&self, _t: *mut c_void, _s: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_mul_scalar(&self, _t: *mut c_void, _s: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_div_scalar(&self, _t: *mut c_void, _s: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_pow_scalar(&self, _t: *mut c_void, _exp: f32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_scale(&self, _t: *mut c_void, _s: f32) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== インプレース演算 ==========
    fn tensor_add_assign(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_sub_assign(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_mul_assign(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_div_assign(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_mod_assign(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_add_assign_scalar_f32(&self, _t: *mut c_void, _s: f32) -> BackendResult<()> { unimplemented!() }
    fn tensor_sub_assign_scalar_f32(&self, _t: *mut c_void, _s: f32) -> BackendResult<()> { unimplemented!() }
    fn tensor_mul_assign_scalar_f32(&self, _t: *mut c_void, _s: f32) -> BackendResult<()> { unimplemented!() }
    fn tensor_div_assign_scalar_f32(&self, _t: *mut c_void, _s: f32) -> BackendResult<()> { unimplemented!() }
    fn tensor_mod_assign_scalar_f32(&self, _t: *mut c_void, _s: f32) -> BackendResult<()> { unimplemented!() }

    // ========== 数学・活性化関数 ==========
    fn tensor_exp(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_log(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sqrt(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sin(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cos(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_tan(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_tanh(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sigmoid(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_relu(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_gelu(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_silu(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== Reduction ==========
    fn tensor_sum(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_mean(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_max(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_min(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_softmax(&self, _t: *mut c_void, _dim: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_max_dim(&self, _t: *mut c_void, _dim: usize, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_min_dim(&self, _t: *mut c_void, _dim: usize, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_mean_dim(&self, _t: *mut c_void, _dim: usize, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sum_dim(&self, _t: *mut c_void, _dim: usize, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_argmax(&self, _t: *mut c_void, _dim: i64, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_argmin(&self, _t: *mut c_void, _dim: i64, _keep_dim: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_tril(&self, _t: *mut c_void, _diagonal: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_clamp(&self, _t: *mut c_void, _min: f64, _max: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_sample(&self, _t: *mut c_void, _temp: f32, _top_p: f32) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== Autograd ==========
    fn tensor_backward(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_grad(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_detach(&self, _t: *mut c_void, _req_grad: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_enable_grad(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn clear_grads(&self) -> BackendResult<()> { unimplemented!() }

    // ========== 形状操作 ==========
    fn tensor_reshape_new(&self, _t: *mut c_void, _new_shape: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_reshape_dims(&self, _t: *mut c_void, _d1: i64, _d2: i64, _d3: i64, _d4: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_transpose(&self, _t: *mut c_void, _dim0: usize, _dim1: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_slice(&self, _t: *mut c_void, _dim: i64, _start: i64, _len: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_narrow(&self, _t: *mut c_void, _dim: usize, _start: usize, _len: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cat(&self, _a: *mut c_void, _b: *mut c_void, _dim: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cat_i64(&self, _a: *mut c_void, _b: *mut c_void, _dim: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cat2(&self, _a: *mut c_void, _b: *mut c_void, _dim: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_cat_4d(&self, _a: *mut c_void, _b: *mut c_void, _dim: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_replace_data(&self, _dst: *mut c_void, _src: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_repeat_interleave(&self, _t: *mut c_void, _repeats: usize, _dim: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_to_device(&self, _t: *mut c_void, _device_id: i32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_to_f32(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_to_i64(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_embedding(&self, _weight: *mut c_void, _indices: *mut c_void, _padding_idx: i64, _scale_grad: bool, _sparse: bool) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== LLM ==========
    fn tensor_rms_norm(&self, _input: *mut c_void, _weight: *mut c_void, _eps: f32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_rope_new_cos(&self, _dim: usize, _seq_len: usize, _base: f32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_rope_new_sin(&self, _dim: usize, _seq_len: usize, _base: f32) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_apply_rope(&self, _t: *mut c_void, _cos: *mut c_void, _sin: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== IO / Print ==========
    fn tensor_print(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_display(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_print_1(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_print_2(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_print_3(&self, _t: *mut c_void) -> BackendResult<()> { unimplemented!() }
    fn tensor_save(&self, _t: *mut c_void, _path: *const i8) -> BackendResult<()> { unimplemented!() }
    fn tensor_load(&self, _path: *const i8) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== NN ==========
    fn tensor_conv2d(&self, _input: *mut c_void, _weight: *mut c_void, _bias: *mut c_void, _stride: usize, _padding: usize, _dilation: usize, _groups: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_batch_norm(&self, _input: *mut c_void, _running_mean: *mut c_void, _running_var: *mut c_void, _weight: *mut c_void, _bias: *mut c_void, _training: bool, _momentum: f64, _eps: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_layer_norm(&self, _input: *mut c_void, _weight: *mut c_void, _bias: *mut c_void, _eps: f64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_dropout(&self, _input: *mut c_void, _p: f64, _training: bool) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_max_pool2d(&self, _input: *mut c_void, _kernel_size: usize, _stride: usize, _padding: usize) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_avg_pool2d(&self, _input: *mut c_void, _kernel_size: usize, _stride: usize, _padding: usize) -> BackendResult<*mut c_void> { unimplemented!() }

    // ========== 次元特化メソッド ==========
    fn tensor_transpose_2d(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_reshape_2d(&self, _t: *mut c_void, _d0: i64, _d1: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_reshape_3d_to_2d(&self, _t: *mut c_void, _d0: i64, _d1: i64) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_matmul_4d(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_add_4d(&self, _a: *mut c_void, _b: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
    fn tensor_silu_4d(&self, _t: *mut c_void) -> BackendResult<*mut c_void> { unimplemented!() }
}
