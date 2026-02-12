//! MetalDeviceImpl: IDevice トレイトの Metal 実装
//!
//! 既存の `ffi_ops::tl_metal_*` 関数に委譲する。
//! Metal FFI と IDevice のシグネチャ差はここで吸収する。

use tl_backend::IDevice;
use crate::ffi_ops;
use crate::ffi;
use crate::tensor::MetalTensor;
use std::ffi::c_void;

/// Metal デバイス (ゼロサイズ型)
pub struct MetalDeviceImpl;

/// void* → *mut MetalTensor
#[inline(always)]
fn t(p: *mut c_void) -> *mut MetalTensor { p as *mut MetalTensor }

/// *mut MetalTensor → void*
#[inline(always)]
fn v(p: *mut MetalTensor) -> *mut c_void { p as *mut c_void }

impl IDevice for MetalDeviceImpl {
    // ========== テンソル作成 ==========
    #[inline] fn tensor_new(&self, data: *const f32, rank: usize, shape: *const usize) -> *mut c_void { v(ffi_ops::tl_metal_new(data, rank, shape)) }
    #[inline] fn tensor_new_i64(&self, data: *const i64, rank: usize, shape: *const usize) -> *mut c_void { v(ffi_ops::tl_metal_new_i64(data, rank, shape)) }
    #[inline] fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> *mut c_void { v(ffi_ops::tl_metal_from_i64_array(data, len)) }
    #[inline] fn tensor_zeros(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { v(ffi_ops::tl_metal_zeros(rank, shape, req_grad)) }
    #[inline] fn tensor_ones(&self, rank: usize, shape: *const usize, req_grad: bool) -> *mut c_void { v(ffi_ops::tl_metal_ones(rank, shape, req_grad)) }
    #[inline] fn tensor_randn_debug(&self, rank: usize, shape: *const usize, seed: u64, req_grad: bool) -> *mut c_void { v(ffi_ops::tl_metal_randn_debug(rank, shape, seed, req_grad)) }
    #[inline] fn tensor_new_causal_mask(&self, size: usize) -> *mut c_void { v(ffi_ops::tl_metal_new_causal_mask(size)) }
    #[inline] fn tensor_from_vec_u8(&self, data: *mut c_void, len: i64) -> *mut c_void { v(ffi_ops::tl_metal_from_vec_u8(data, len)) }
    #[inline] fn tensor_from_u8_labels(&self, data: *const u8, len: i64) -> *mut c_void { v(ffi_ops::tl_metal_from_u8_labels(data, len)) }

    // ========== メモリ管理 ==========
    #[inline] fn tensor_clone(&self, a: *mut c_void) -> *mut c_void { v(ffi::tl_metal_clone(t(a))) }
    #[inline] fn tensor_free(&self, a: *mut c_void) { ffi_ops::tl_metal_free(t(a)) }
    #[inline] fn tensor_release(&self, a: *mut c_void) { ffi::tl_metal_release(t(a)) }
    #[inline] fn tensor_acquire(&self, a: *mut c_void) -> *mut c_void {
        if a.is_null() { return a; }
        // Arc<UnsafeCell<MetalTensor>> として from_raw + clone + forget
        let arc = unsafe { std::sync::Arc::from_raw(t(a) as *const std::cell::UnsafeCell<MetalTensor>) };
        let cloned = arc.clone();
        std::mem::forget(arc); // 元のポインタは生かす
        v(std::sync::Arc::into_raw(cloned) as *mut MetalTensor)
    }
    #[inline] fn tensor_release_safe(&self, a: *mut c_void) {
        if a.is_null() { return; }
        // Arc::from_raw で復元し drop。RC-1、RC=0 で MetalTensor が Drop される。
        unsafe { let _ = std::sync::Arc::from_raw(t(a) as *const std::cell::UnsafeCell<MetalTensor>); }
    }
    #[inline] fn tensor_promote(&self, _a: *mut c_void) { /* Metal: pool-based — runtime 側で処理 */ }
    #[inline] fn tensor_register(&self, _a: *mut c_void) { /* Metal: noop */ }
    #[inline] fn tensor_prepare_return(&self, a: *mut c_void) -> *mut c_void { a }

    // ========== テンソル情報 ==========
    #[inline] fn tensor_len(&self, a: *mut c_void) -> usize { ffi_ops::tl_metal_len(t(a)) }
    #[inline] fn tensor_dim(&self, a: *mut c_void, dim: usize) -> usize { ffi_ops::tl_metal_dim(t(a), dim) }
    #[inline] fn tensor_numel(&self, a: *mut c_void) -> i64 { ffi::tl_metal_numel(t(a)) }
    #[inline] fn tensor_data(&self, a: *mut c_void) -> *const f32 { ffi::tl_metal_data(t(a)) as *const f32 }
    #[inline] fn tensor_device_id(&self, a: *mut c_void) -> i32 { ffi_ops::tl_metal_device_id(t(a)) }
    #[inline] fn tensor_get_shape(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_get_shape(t(a))) }

    // ========== 要素アクセス ==========
    #[inline] fn tensor_get(&self, a: *mut c_void, idx: i64) -> f32 { ffi_ops::tl_metal_get(t(a), idx) }
    #[inline] fn tensor_get_f32_md(&self, a: *mut c_void, indices: *const i64, rank: i64) -> f32 {
        // Metal の tl_tensor_get_f32_md は (t, idx0, idx1) シグネチャ
        // IDevice では (t, *const i64, rank) — ここでアダプト
        let slice = unsafe { std::slice::from_raw_parts(indices, rank as usize) };
        if slice.len() >= 2 {
            ffi_ops::tl_metal_get_f32_md(t(a), slice[0], slice[1])
        } else if slice.len() == 1 {
            ffi_ops::tl_metal_get_f32_md(t(a), slice[0], 0)
        } else {
            0.0
        }
    }
    #[inline] fn tensor_get_i64_md(&self, a: *mut c_void, indices: *const i64, rank: i64) -> i64 {
        let slice = unsafe { std::slice::from_raw_parts(indices, rank as usize) };
        if slice.len() >= 2 {
            ffi_ops::tl_metal_get_i64_md(t(a), slice[0], slice[1])
        } else if slice.len() == 1 {
            ffi_ops::tl_metal_get_i64_md(t(a), slice[0], 0)
        } else {
            0
        }
    }
    #[inline] fn tensor_set_f32_md(&self, a: *mut c_void, indices: *const i64, rank: usize, value: f32) -> *mut c_void {
        v(ffi_ops::tl_metal_set_f32_md(t(a), indices, rank, value))
    }
    #[inline] fn tensor_item(&self, a: *mut c_void) -> f32 { ffi_ops::tl_metal_item(t(a)) }
    #[inline] fn tensor_item_i64(&self, a: *mut c_void) -> i64 { ffi_ops::tl_metal_item_i64(t(a)) }

    // ========== 二項演算 ==========
    #[inline] fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_add(t(a), t(b))) }
    #[inline] fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sub(t(a), t(b))) }
    #[inline] fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_mul(t(a), t(b))) }
    #[inline] fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_div(t(a), t(b))) }
    #[inline] fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_rem(t(a), t(b))) }
    #[inline] fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_matmul(t(a), t(b))) }
    #[inline] fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_pow(t(a), t(b))) }
    #[inline] fn tensor_cross_entropy(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_cross_entropy(t(a), t(b))) }

    // ========== 単項演算 ==========
    #[inline] fn tensor_neg(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_neg(t(a))) }
    #[inline] fn tensor_abs(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_abs(t(a))) }
    #[inline] fn tensor_contiguous(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_contiguous(t(a))) }

    // ========== 比較演算 ==========
    #[inline] fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_eq(t(a), t(b))) }
    #[inline] fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_neq(t(a), t(b))) }
    #[inline] fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_gt(t(a), t(b))) }
    #[inline] fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_lt(t(a), t(b))) }
    #[inline] fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_ge(t(a), t(b))) }
    #[inline] fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_le(t(a), t(b))) }

    // ========== スカラー演算 ==========
    #[inline] fn tensor_add_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi_ops::tl_metal_add_scalar(t(a), s)) }
    #[inline] fn tensor_sub_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi_ops::tl_metal_sub_scalar(t(a), s)) }
    #[inline] fn tensor_mul_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi_ops::tl_metal_mul_scalar(t(a), s)) }
    #[inline] fn tensor_div_scalar(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi_ops::tl_metal_div_scalar(t(a), s)) }
    #[inline] fn tensor_pow_scalar(&self, a: *mut c_void, exp: f32) -> *mut c_void { v(ffi_ops::tl_metal_pow_scalar(t(a), exp as f64)) }
    #[inline] fn tensor_scale(&self, a: *mut c_void, s: f64) -> *mut c_void { v(ffi_ops::tl_metal_scale(t(a), s)) }

    // ========== インプレース演算 ==========
    #[inline] fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void) { ffi_ops::tl_metal_add_assign(t(a), t(b)) }
    #[inline] fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void) { ffi_ops::tl_metal_sub_assign(t(a), t(b)) }
    #[inline] fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void) { ffi_ops::tl_metal_mul_assign(t(a), t(b)) }
    #[inline] fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void) { ffi_ops::tl_metal_div_assign(t(a), t(b)) }
    #[inline] fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void) { ffi_ops::tl_metal_mod_assign(t(a), t(b)) }
    #[inline] fn tensor_add_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi_ops::tl_metal_add_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_sub_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi_ops::tl_metal_sub_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_mul_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi_ops::tl_metal_mul_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_div_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi_ops::tl_metal_div_assign_scalar_f32(t(a), s) }
    #[inline] fn tensor_mod_assign_scalar_f32(&self, a: *mut c_void, s: f32) { ffi_ops::tl_metal_mod_assign_scalar_f32(t(a), s) }

    // ========== 数学・活性化関数 ==========
    #[inline] fn tensor_exp(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_exp(t(a))) }
    #[inline] fn tensor_log(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_log(t(a))) }
    #[inline] fn tensor_sqrt(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sqrt(t(a))) }
    #[inline] fn tensor_sin(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sin(t(a))) }
    #[inline] fn tensor_cos(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_cos(t(a))) }
    #[inline] fn tensor_tan(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_tan(t(a))) }
    #[inline] fn tensor_tanh(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_tanh(t(a))) }
    #[inline] fn tensor_sigmoid(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sigmoid(t(a))) }
    #[inline] fn tensor_relu(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_relu(t(a))) }
    #[inline] fn tensor_gelu(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_gelu(t(a))) }
    #[inline] fn tensor_silu(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_silu(t(a))) }

    // ========== Reduction ==========
    #[inline] fn tensor_sum(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sum(t(a))) }
    #[inline] fn tensor_mean(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_mean(t(a))) }
    #[inline] fn tensor_max(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_max(t(a))) }
    #[inline] fn tensor_min(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_min(t(a))) }
    #[inline] fn tensor_softmax(&self, a: *mut c_void, dim: i64) -> *mut c_void { v(ffi_ops::tl_metal_softmax(t(a), dim)) }
    #[inline] fn tensor_max_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_max_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_min_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_min_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_mean_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_mean_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_sum_dim(&self, a: *mut c_void, dim: usize, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_sum_dim(t(a), dim, keep_dim)) }
    #[inline] fn tensor_argmax(&self, a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_argmax(t(a), dim, keep_dim)) }
    #[inline] fn tensor_argmin(&self, a: *mut c_void, dim: i64, keep_dim: bool) -> *mut c_void { v(ffi_ops::tl_metal_argmin(t(a), dim, keep_dim)) }
    #[inline] fn tensor_tril(&self, a: *mut c_void, diagonal: i64) -> *mut c_void { v(ffi_ops::tl_metal_tril(t(a), diagonal)) }
    #[inline] fn tensor_clamp(&self, a: *mut c_void, min: f64, max: f64) -> *mut c_void { v(ffi_ops::tl_metal_clamp(t(a), min, max)) }
    #[inline] fn tensor_sample(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_sample(t(a))) }

    // ========== Autograd ==========
    #[inline] fn tensor_backward(&self, a: *mut c_void) { ffi_ops::tl_metal_backward(t(a)) }
    #[inline] fn tensor_grad(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_grad(t(a))) }
    #[inline] fn tensor_detach(&self, a: *mut c_void, _req_grad: bool) -> *mut c_void { v(ffi_ops::tl_metal_detach(t(a))) }
    #[inline] fn tensor_enable_grad(&self, a: *mut c_void) { ffi_ops::tl_metal_enable_grad(t(a)) }
    #[inline] fn clear_grads(&self) { /* Metal: handled by runtime */ }

    // ========== 形状操作 ==========
    #[inline] fn tensor_reshape_new(&self, a: *mut c_void, s: *mut c_void) -> *mut c_void {
        // Metal の reshape_new は (t, *const i64, usize) — IDevice では (t, *mut c_void) で shape テンソルを渡す
        // runtime 経由で使われるので、ここでは shape tensor から情報を取る必要がある
        // 簡易実装: shape tensor のデータポインタと長さを使う
        if s.is_null() { return a; }
        let shape_t = unsafe { &*t(s) };
        let shape_data = shape_t.to_vec_i64();
        let rank = shape_data.len();
        v(ffi_ops::tl_metal_reshape_new(t(a), shape_data.as_ptr(), rank))
    }
    #[inline] fn tensor_reshape_dims(&self, a: *mut c_void, d1: i64, d2: i64, d3: i64, d4: i64) -> *mut c_void {
        // Metal の reshape_dims は (t, *const i64, usize) — dims 配列を構築
        let mut dims = Vec::new();
        for &d in &[d1, d2, d3, d4] {
            if d != 0 { dims.push(d); } else { break; }
        }
        v(ffi_ops::tl_metal_reshape_dims(t(a), dims.as_ptr(), dims.len()))
    }
    #[inline] fn tensor_transpose(&self, a: *mut c_void, dim0: usize, dim1: usize) -> *mut c_void { v(ffi_ops::tl_metal_transpose(t(a), dim0, dim1)) }
    #[inline] fn tensor_slice(&self, a: *mut c_void, dim: i64, start: i64, len: i64) -> *mut c_void { v(ffi_ops::tl_metal_slice(t(a), dim as usize, start as usize, len as usize)) }
    #[inline] fn tensor_narrow(&self, a: *mut c_void, dim: usize, start: usize, len: usize) -> *mut c_void { v(ffi_ops::tl_metal_narrow(t(a), dim, start, len)) }
    #[inline] fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi_ops::tl_metal_cat(t(a), t(b), dim)) }
    #[inline] fn tensor_cat_i64(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi_ops::tl_metal_cat_i64(t(a), t(b), dim)) }
    #[inline] fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi_ops::tl_metal_cat(t(a), t(b), dim)) }
    #[inline] fn tensor_cat_4d(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> *mut c_void { v(ffi_ops::tl_metal_cat(t(a), t(b), dim)) }
    #[inline] fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void) { ffi_ops::tl_metal_replace_data(t(dst), t(src)) }
    #[inline] fn tensor_repeat_interleave(&self, a: *mut c_void, repeats: usize, dim: usize) -> *mut c_void { v(ffi_ops::tl_metal_repeat_interleave(t(a), repeats, dim)) }
    #[inline] fn tensor_to_device(&self, a: *mut c_void, device_id: i32) -> *mut c_void { v(ffi_ops::tl_metal_to_device(t(a), device_id)) }
    #[inline] fn tensor_to_f32(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_to_f32(t(a))) }
    #[inline] fn tensor_to_i64(&self, a: *mut c_void) -> *mut c_void { v(ffi_ops::tl_metal_to_i64(t(a))) }
    #[inline] fn tensor_embedding(&self, w: *mut c_void, idx: *mut c_void, pad: i64, sg: bool, sp: bool) -> *mut c_void { v(ffi_ops::tl_metal_embedding(t(w), t(idx), pad, sg, sp)) }

    // ========== LLM ==========
    #[inline] fn tensor_rms_norm(&self, a: *mut c_void, w: *mut c_void, eps: f32) -> *mut c_void { v(ffi_ops::tl_metal_rms_norm(t(a), t(w), eps)) }
    #[inline] fn tensor_rope_new_cos(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void { v(ffi_ops::tl_metal_rope_new_cos(seq_len, dim, base)) }
    #[inline] fn tensor_rope_new_sin(&self, seq_len: usize, dim: usize, base: f32) -> *mut c_void { v(ffi_ops::tl_metal_rope_new_sin(seq_len, dim, base)) }
    #[inline] fn tensor_apply_rope(&self, a: *mut c_void, cos: *mut c_void, sin: *mut c_void) -> *mut c_void {
        if a.is_null() || cos.is_null() || sin.is_null() { return a; }
        let tensor = unsafe { &*t(a) };
        let c = unsafe { &*t(cos) };
        let s = unsafe { &*t(sin) };
        let result = tensor.apply_rope_impl(c, s, 0);
        v(ffi_ops::make_tensor(result))
    }

    // ========== IO / Print ==========
    #[inline] fn tensor_print(&self, a: *mut c_void) {
        // Metal: runtime の print_ffi を使う
        if !a.is_null() {
            let tensor = unsafe { &*t(a) };
            println!("{:?}", tensor.to_vec_f32());
        }
    }
    #[inline] fn tensor_display(&self, a: *mut c_void) { self.tensor_print(a) }
    #[inline] fn tensor_print_1(&self, a: *mut c_void) { self.tensor_print(a) }
    #[inline] fn tensor_print_2(&self, a: *mut c_void) { self.tensor_print(a) }
    #[inline] fn tensor_print_3(&self, a: *mut c_void) { self.tensor_print(a) }
    #[inline] fn tensor_save(&self, a: *mut c_void, path: *const i8) {
        if a.is_null() || path.is_null() { return; }
        unsafe {
            let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy();
            let tensor = &*t(a);
            let data: Vec<f32> = tensor.to_vec_f32();
            let shape = MetalTensor::shape(tensor).to_vec();

            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for &dim in &shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            for &val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            if let Err(e) = std::fs::write(path_str.as_ref(), &bytes) {
                eprintln!("Failed to save tensor: {}", e);
            }
        }
    }
    #[inline] fn tensor_load(&self, path: *const i8) -> *mut c_void {
        if path.is_null() { return v(ffi_ops::make_tensor(MetalTensor::zeros(&[1], crate::DType::F32))); }
        unsafe {
            let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy();
            let bytes = match std::fs::read(path_str.as_ref()) {
                Ok(b) => b,
                Err(_) => return v(ffi_ops::make_tensor(MetalTensor::zeros(&[1], crate::DType::F32))),
            };
            if bytes.len() < 8 {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(&[1], crate::DType::F32)));
            }
            let mut offset = 0;
            let rank = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
            offset += 8;
            if bytes.len() < offset + rank * 8 {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(&[1], crate::DType::F32)));
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dim = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
                shape.push(dim);
                offset += 8;
            }
            let numel: usize = shape.iter().product();
            let expected_data_size = numel * 4;
            if bytes.len() < offset + expected_data_size {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(&[1], crate::DType::F32)));
            }
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                let val = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
                data.push(val);
                offset += 4;
            }
            v(ffi_ops::make_tensor(MetalTensor::from_slice(&data, &shape, crate::DType::F32)))
        }
    }

    // ========== NN ==========
    #[inline] fn tensor_conv2d(&self, input: *mut c_void, weight: *mut c_void, bias: *mut c_void, stride: usize, padding: usize, dilation: usize, groups: usize) -> *mut c_void {
        v(ffi_ops::tl_metal_conv2d(t(input), t(weight), t(bias), stride, padding, dilation, groups))
    }
    #[inline] fn tensor_batch_norm(&self, input: *mut c_void, running_mean: *mut c_void, running_var: *mut c_void, weight: *mut c_void, bias: *mut c_void, training: bool, momentum: f64, eps: f64) -> *mut c_void {
        v(ffi_ops::tl_metal_batch_norm(t(input), t(running_mean), t(running_var), t(weight), t(bias), training, momentum, eps))
    }
    #[inline] fn tensor_layer_norm(&self, input: *mut c_void, weight: *mut c_void, bias: *mut c_void, eps: f64) -> *mut c_void {
        v(ffi_ops::tl_metal_layer_norm(t(input), t(weight), t(bias), eps))
    }
    #[inline] fn tensor_dropout(&self, input: *mut c_void, p: f64, training: bool) -> *mut c_void {
        v(ffi_ops::tl_metal_dropout(t(input), p, training))
    }
    #[inline] fn tensor_max_pool2d(&self, input: *mut c_void, kernel_size: usize, stride: usize, padding: usize) -> *mut c_void {
        v(ffi_ops::tl_metal_max_pool2d(t(input), kernel_size, stride, padding))
    }
    #[inline] fn tensor_avg_pool2d(&self, input: *mut c_void, kernel_size: usize, stride: usize, padding: usize) -> *mut c_void {
        v(ffi_ops::tl_metal_avg_pool2d(t(input), kernel_size, stride, padding))
    }
}

use tl_backend::GpuTensor;

