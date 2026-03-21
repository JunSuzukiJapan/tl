//! IDevice トレイトの CUDA 実装
//!
//! 全メソッドを ffi_ops.rs 経由で実装。

use crate::ffi_ops;
use std::ffi::c_void;
use tl_backend::device::{BackendResult, IDevice};

pub struct CudaDeviceImpl;

/// c_void ポインタを OpaqueTensor ポインタにキャスト
fn p(t: *mut c_void) -> *mut crate::tensor::CudaTensor {
    t as *mut crate::tensor::CudaTensor
}

/// OpaqueTensor ポインタを c_void にキャスト
fn v(t: *mut crate::tensor::CudaTensor) -> *mut c_void {
    t as *mut c_void
}

impl IDevice for CudaDeviceImpl {
    // ========== テンソル作成 ==========
    fn tensor_new(
        &self,
        data: *const f32,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_new(data, rank, shape)))
    }
    fn tensor_new_i64(
        &self,
        data: *const i64,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_new_i64(data, rank, shape)))
    }
    fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_from_i64_array(data, len)))
    }
    fn tensor_zeros(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_zeros(rank, shape, req_grad)))
    }
    fn tensor_ones(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_ones(rank, shape, req_grad)))
    }
    fn tensor_randn_debug(
        &self,
        rank: usize,
        shape: *const usize,
        seed: u64,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_randn_debug(rank, shape, seed, req_grad)))
    }
    fn tensor_new_causal_mask(&self, size: usize) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_new_causal_mask(size)))
    }
    fn tensor_from_vec_u8(&self, data: *mut c_void, offset: i64, shape_ptr: *const i64, rank: i64) -> BackendResult<*mut c_void> {
        if data.is_null() || shape_ptr.is_null() || rank <= 0 {
            return Err(tl_backend::error::BackendError::InternalError(
                "tensor_from_vec_u8: null data or shape pointer".into(),
            ));
        }
        let rank_usize = rank as usize;
        let shape_slice = unsafe { std::slice::from_raw_parts(shape_ptr, rank_usize) };
        let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
        let numel: usize = shape.iter().product();
        
        let offset_usize = offset as usize;
        let data_ptr = data as *const u8;
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr.add(offset_usize), numel) };
        let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
        
        // CPU backend で一旦作成してから CUDA にコピー
        let cpu_tensor = crate::tensor::CudaTensor::from_f32_data(&f32_data, &shape);
        Ok(v(ffi_ops::make_tensor(cpu_tensor)))
    }
    fn tensor_from_u8_labels(&self, data: *const u8, len: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_from_u8_labels(data, len)))
    }
    fn tensor_full(
        &self,
        rank: usize,
        shape: *const usize,
        value: f32,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_full(rank, shape, value, req_grad)))
    }
    fn tensor_eye(&self, n: usize, req_grad: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_eye(n, req_grad)))
    }
    fn tensor_arange(&self, start: f64, end: f64, step: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_arange(start, end, step)))
    }
    fn tensor_linspace(&self, start: f64, end: f64, steps: usize) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_linspace(start, end, steps)))
    }
    fn tensor_rand(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_rand(rank as i64, shape, req_grad)))
    }
    // ========== 要素操作 ==========
    fn tensor_where_cond(
        &self,
        cond: *mut c_void,
        x: *mut c_void,
        y: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tc, tx, ty) = unsafe { (&*p(cond), &*p(x), &*p(y)) };
        let result = crate::tensor::CudaTensor::where_cond_impl(tc, tx, ty)?;
        Ok(v(ffi_ops::make_tensor(result)))
    }
    fn tensor_masked_fill(
        &self,
        t: *mut c_void,
        mask: *mut c_void,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        let (tt, tm) = unsafe { (&*p(t), &*p(mask)) };
        Ok(v(ffi_ops::make_tensor(tt.masked_fill_impl(tm, value)?)))
    }
    fn tensor_to_vec_f32(&self, t: *mut c_void) -> BackendResult<(*mut f32, usize)> {
        let tt = unsafe { &*p(t) };
        let data = tt.to_vec_f32_impl()?;
        let len = data.len();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut f32;
        Ok((ptr, len))
    }
    fn tensor_var(&self, t: *mut c_void, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.var_impl()?)))
    }
    fn tensor_std(&self, t: *mut c_void, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.std_impl()?)))
    }
    fn tensor_prod(&self, t: *mut c_void, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.prod_impl()?)))
    }
    fn tensor_cumsum(&self, t: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.cumsum_impl(dim as usize)?)))
    }
    fn tensor_norm(&self, t: *mut c_void, p_val: f32, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.norm_impl(p_val)?)))
    }
    fn tensor_topk(&self, t: *mut c_void, k: usize, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.topk_impl(k)?)))
    }
    fn tensor_logical_and(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.logical_and_impl(tb)?)))
    }
    fn tensor_logical_or(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.logical_or_impl(tb)?)))
    }
    fn tensor_logical_not(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.logical_not_impl()?)))
    }
    fn tensor_leaky_relu(&self, t: *mut c_void, slope: f32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.leaky_relu_impl(slope)?)))
    }
    fn tensor_elu(&self, t: *mut c_void, alpha: f32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.elu_impl(alpha)?)))
    }
    fn tensor_mish(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.mish_impl()?)))
    }
    fn tensor_mse_loss(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.mse_loss_impl(tb)?)))
    }
    fn tensor_l1_loss(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.l1_loss_impl(tb)?)))
    }
    fn tensor_bce_loss(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.bce_loss_impl(tb)?)))
    }
    fn tensor_nll_loss(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.nll_loss_impl(tb)?)))
    }
    fn tensor_linear(
        &self,
        i: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*p(i), &*p(w)) };
        let bias = if b.is_null() {
            None
        } else {
            Some(unsafe { &*p(b) })
        };
        Ok(v(ffi_ops::make_tensor(ti.linear_impl(tw, bias)?)))
    }
    fn tensor_hardswish(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.hardswish_impl()?)))
    }
    fn tensor_hardsigmoid(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.hardsigmoid_impl()?)))
    }
    fn tensor_group_norm(
        &self,
        i: *mut c_void,
        g: i64,
        w: *mut c_void,
        b: *mut c_void,
        e: f64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw, tb) = unsafe { (&*p(i), &*p(w), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(
            ti.group_norm_impl(tw, tb, g as usize, e as f32)?,
        )))
    }
    fn tensor_adaptive_avg_pool2d(
        &self,
        i: *mut c_void,
        h: i64,
        w: i64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*p(i) };
        Ok(v(ffi_ops::make_tensor(
            ti.adaptive_avg_pool2d_impl((h as usize, w as usize))?,
        )))
    }
    fn tensor_pad(&self, i: *mut c_void, l: i64, r: i64, val: f32) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*p(i) };
        Ok(v(ffi_ops::make_tensor(
            ti.pad_impl(&[l as usize, r as usize], val)?,
        )))
    }
    fn tensor_instance_norm(
        &self,
        i: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
        e: f64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw, tb) = unsafe { (&*p(i), &*p(w), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(
            ti.instance_norm_impl(tw, tb, e as f32)?,
        )))
    }
    fn tensor_dropout2d(
        &self,
        i: *mut c_void,
        prob: f64,
        training: bool,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*p(i) };
        Ok(v(ffi_ops::make_tensor(
            ti.dropout2d_impl(prob as f32, training)?,
        )))
    }
    fn tensor_conv1d(
        &self,
        i: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
        s: i64,
        pad: i64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*p(i), &*p(w)) };
        let result = ti.conv1d_impl(tw, s as usize, pad as usize)?;
        let result = if !b.is_null() {
            let tb = unsafe { &*p(b) };
            result.add_impl(tb)?
        } else {
            result
        };
        Ok(v(ffi_ops::make_tensor(result)))
    }
    fn tensor_kl_div_loss(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.kl_div_loss_impl(tb)?)))
    }
    fn tensor_conv_transpose2d(
        &self,
        i: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
        s: i64,
        pad: i64,
        _o: i64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*p(i), &*p(w)) };
        let result =
            ti.conv_transpose2d_impl(tw, (s as usize, s as usize), (pad as usize, pad as usize))?;
        let result = if !b.is_null() {
            let tb = unsafe { &*p(b) };
            result.add_impl(tb)?
        } else {
            result
        };
        Ok(v(ffi_ops::make_tensor(result)))
    }
    fn tensor_interpolate(
        &self,
        i: *mut c_void,
        h: i64,
        w: i64,
        m: i64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*p(i) };
        let mode = if m == 1 { "bilinear" } else { "nearest" };
        Ok(v(ffi_ops::make_tensor(
            ti.interpolate_impl((h as usize, w as usize), mode)?,
        )))
    }
    fn tensor_scaled_dot_product_attention(
        &self,
        q: *mut c_void,
        k: *mut c_void,
        val: *mut c_void,
        m: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tq, tk, tv) = unsafe { (&*p(q), &*p(k), &*p(val)) };
        let mask = if m.is_null() {
            None
        } else {
            Some(unsafe { &*p(m) })
        };
        Ok(v(ffi_ops::make_tensor(tq.sdpa_impl(tk, tv, mask)?)))
    }
    fn tensor_top_k_sample(&self, l: *mut c_void, k: i64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*p(l) };
        Ok(v(ffi_ops::make_tensor(tl.top_k_sample_impl(k as usize)?)))
    }
    fn tensor_top_p_sample(&self, l: *mut c_void, prob: f64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*p(l) };
        Ok(v(ffi_ops::make_tensor(tl.top_p_sample_impl(prob as f32)?)))
    }
    fn tensor_temperature_scale(&self, l: *mut c_void, t: f64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*p(l) };
        Ok(v(ffi_ops::make_tensor(
            tl.temperature_scale_impl(t as f32)?,
        )))
    }
    fn tensor_repetition_penalty(
        &self,
        l: *mut c_void,
        tokens: *mut c_void,
        penalty: f64,
    ) -> BackendResult<*mut c_void> {
        let (tl, tt) = unsafe { (&*p(l), &*p(tokens)) };
        Ok(v(ffi_ops::make_tensor(
            tl.repetition_penalty_impl(tt, penalty as f32)?,
        )))
    }
    fn tensor_dot(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.dot_impl(tb)?)))
    }
    fn tensor_fill_(&self, t: *mut c_void, value: f32) -> BackendResult<()> {
        let tt = unsafe { &*p(t) };
        let _ = tt.fill_impl(value)?;
        Ok(())
    }
    fn tensor_broadcast_to(&self, t: *mut c_void, shape: &[usize]) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*p(t) };
        Ok(v(ffi_ops::make_tensor(tt.broadcast_to_impl(shape)?)))
    }
    fn tensor_stack(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*p(a), &*p(b)) };
        Ok(v(ffi_ops::make_tensor(ta.stack_impl(tb, dim as usize)?)))
    }
    // ========== autograd ==========
    fn tensor_set_requires_grad(&self, t: *mut c_void, requires_grad: bool) -> BackendResult<()> {
        let tt = unsafe { &mut *p(t) };
        if requires_grad {
            tt.enable_grad();
        } else if let Some(ref mut meta) = tt.autograd {
            meta.requires_grad = false;
        }
        Ok(())
    }
    fn tensor_clip_grad_value(&self, t: *mut c_void, min: f64, max: f64) -> BackendResult<()> {
        let tt = unsafe { &mut *p(t) };
        if let Some(ref mut meta) = tt.autograd {
            if let Some(ref grad) = meta.grad {
                let clipped = grad.clamp_impl(min as f32, max as f32)?;
                meta.grad = Some(clipped);
            }
        }
        Ok(())
    }
    fn tensor_clip_grad_norm(
        &self,
        t: *mut c_void,
        max_norm: f64,
        norm_type: f64,
    ) -> BackendResult<f64> {
        let tt = unsafe { &mut *p(t) };
        if let Some(ref mut meta) = tt.autograd {
            if let Some(ref grad) = meta.grad {
                let total_norm_t = grad.norm_impl(norm_type as f32)?;
                let total_norm_vec = total_norm_t.to_vec::<f32>();
                let total_norm = total_norm_vec[0] as f64;
                if total_norm > max_norm {
                    let scale = max_norm / (total_norm + 1e-6);
                    let scaled = grad.mul_scalar_impl(scale as f32)?;
                    meta.grad = Some(scaled);
                }
                return Ok(total_norm);
            }
        }
        Ok(0.0)
    }
    // ========== メモリ管理 ==========
    fn tensor_clone(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(crate::ffi::tl_cuda_clone(p(t))))
    }
    fn tensor_shallow_clone(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(crate::ffi::tl_cuda_shallow_clone(p(t))))
    }
    fn tensor_free(&self, t: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_free(p(t));
        Ok(())
    }
    fn tensor_release(&self, t: *mut c_void) -> BackendResult<()> {
        ffi_ops::release_if_live(p(t));
        Ok(())
    }
    fn tensor_acquire(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        ffi_ops::acquire_tensor(p(t));
        Ok(t)
    }
    fn tensor_release_safe(&self, t: *mut c_void) -> BackendResult<()> {
        ffi_ops::release_if_live(p(t));
        Ok(())
    }
    fn tensor_promote(&self, _t: *mut c_void) -> BackendResult<()> {
        Ok(())
    }
    fn tensor_register(&self, _t: *mut c_void) -> BackendResult<()> {
        Ok(())
    }
    fn tensor_prepare_return(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        ffi_ops::acquire_tensor(p(t));
        Ok(t)
    }

    // ========== テンソル情報 ==========
    fn tensor_len(&self, t: *mut c_void) -> BackendResult<usize> {
        Ok(ffi_ops::tl_cuda_len(p(t)))
    }
    fn tensor_dim(&self, t: *mut c_void, dim: usize) -> BackendResult<usize> {
        Ok(ffi_ops::tl_cuda_dim(p(t), dim))
    }
    fn tensor_numel(&self, t: *mut c_void) -> BackendResult<i64> {
        Ok(crate::ffi::tl_cuda_numel(p(t)))
    }
    fn tensor_data(&self, t: *mut c_void) -> BackendResult<*const f32> {
        Ok(crate::ffi::tl_cuda_data(p(t)) as *const f32)
    }
    fn tensor_device_id(&self, _t: *mut c_void) -> BackendResult<i32> {
        Ok(0)
    }
    fn tensor_get_shape(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_get_shape(p(t))))
    }

    // ========== 要素アクセス ==========
    fn tensor_get(&self, t: *mut c_void, idx: i64) -> BackendResult<f32> {
        Ok(ffi_ops::tl_cuda_get(p(t), idx))
    }
    fn tensor_get_f32_md(
        &self,
        t: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<f32> {
        unsafe {
            let idx_slice = std::slice::from_raw_parts(indices, rank as usize);
            let tensor = &*(p(t) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let st = &*tensor.get();
            let shape = st.shape();
            // flat index を stride ベースで計算（任意次元対応）
            let mut flat = 0usize;
            let mut stride = 1usize;
            for i in (0..shape.len()).rev() {
                let dim_idx = if i < idx_slice.len() {
                    idx_slice[i] as usize
                } else {
                    0
                };
                flat += dim_idx * stride;
                stride *= shape[i];
            }
            let data = st.to_vec::<f32>();
            Ok(data[flat])
        }
    }
    fn tensor_get_i64_md(
        &self,
        t: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<i64> {
        unsafe {
            let idx_slice = std::slice::from_raw_parts(indices, rank as usize);
            let tensor = &*(p(t) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let st = &*tensor.get();
            let shape = st.shape();
            // flat index を stride ベースで計算（任意次元対応）
            let mut flat = 0usize;
            let mut stride = 1usize;
            for i in (0..shape.len()).rev() {
                let dim_idx = if i < idx_slice.len() {
                    idx_slice[i] as usize
                } else {
                    0
                };
                flat += dim_idx * stride;
                stride *= shape[i];
            }
            let data = st.to_vec::<i64>();
            Ok(data[flat])
        }
    }
    fn tensor_set_f32_md(
        &self,
        t: *mut c_void,
        indices: *const i64,
        rank: usize,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_set_f32_md(p(t), indices, rank, value)))
    }
    fn tensor_item(&self, t: *mut c_void) -> BackendResult<f32> {
        Ok(ffi_ops::tl_cuda_item(p(t)))
    }
    fn tensor_item_i64(&self, t: *mut c_void) -> BackendResult<i64> {
        Ok(ffi_ops::tl_cuda_item_i64(p(t)))
    }

    // ========== 二項演算 ==========
    fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_add(p(a), p(b))))
    }
    fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sub(p(a), p(b))))
    }
    fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_mul(p(a), p(b))))
    }
    fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_div(p(a), p(b))))
    }
    fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_rem(p(a), p(b))))
    }
    fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_matmul(p(a), p(b))))
    }
    fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_pow(p(a), p(b))))
    }
    fn tensor_cross_entropy(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cross_entropy(p(a), p(b))))
    }

    // ========== 単項演算 ==========
    fn tensor_neg(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_neg(p(t))))
    }
    fn tensor_abs(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_abs(p(t))))
    }
    fn tensor_contiguous(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_contiguous(p(t))))
    }

    // ========== 比較演算 ==========
    fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_eq(p(a), p(b))))
    }
    fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_neq(p(a), p(b))))
    }
    fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_gt(p(a), p(b))))
    }
    fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_lt(p(a), p(b))))
    }
    fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_ge(p(a), p(b))))
    }
    fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_le(p(a), p(b))))
    }

    // ========== スカラー演算 ==========
    fn tensor_add_scalar(&self, t: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_add_scalar(p(t), s)))
    }
    fn tensor_sub_scalar(&self, t: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sub_scalar(p(t), s)))
    }
    fn tensor_mul_scalar(&self, t: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_mul_scalar(p(t), s)))
    }
    fn tensor_div_scalar(&self, t: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_div_scalar(p(t), s)))
    }
    fn tensor_pow_scalar(&self, t: *mut c_void, exp: f32) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_pow_scalar(p(t), exp as f64)))
    }
    fn tensor_scale(&self, t: *mut c_void, s: f32) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_scale(p(t), s)))
    }

    // ========== インプレース演算 ==========
    fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_add_assign(p(a), p(b));
        Ok(())
    }
    fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_sub_assign(p(a), p(b));
        Ok(())
    }
    fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_mul_assign(p(a), p(b));
        Ok(())
    }
    fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_div_assign(p(a), p(b));
        Ok(())
    }
    fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_mod_assign(p(a), p(b));
        Ok(())
    }
    fn tensor_add_assign_scalar_f32(&self, t: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_cuda_add_assign_scalar_f32(p(t), s);
        Ok(())
    }
    fn tensor_sub_assign_scalar_f32(&self, t: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_cuda_sub_assign_scalar_f32(p(t), s);
        Ok(())
    }
    fn tensor_mul_assign_scalar_f32(&self, t: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_cuda_mul_assign_scalar_f32(p(t), s);
        Ok(())
    }
    fn tensor_div_assign_scalar_f32(&self, t: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_cuda_div_assign_scalar_f32(p(t), s);
        Ok(())
    }
    fn tensor_mod_assign_scalar_f32(&self, t: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_cuda_mod_assign_scalar_f32(p(t), s);
        Ok(())
    }

    // ========== 数学・活性化関数 ==========
    fn tensor_exp(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_exp(p(t))))
    }
    fn tensor_log(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_log(p(t))))
    }
    fn tensor_sqrt(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sqrt(p(t))))
    }
    fn tensor_sin(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sin(p(t))))
    }
    fn tensor_cos(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cos(p(t))))
    }
    fn tensor_tan(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_tan(p(t))))
    }
    fn tensor_tanh(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_tanh(p(t))))
    }
    fn tensor_sigmoid(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sigmoid(p(t))))
    }
    fn tensor_relu(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_relu(p(t))))
    }
    fn tensor_gelu(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_gelu(p(t))))
    }
    fn tensor_silu(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_silu(p(t))))
    }

    // ========== Reduction ==========
    fn tensor_sum(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sum(p(t))))
    }
    fn tensor_mean(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_mean(p(t))))
    }
    fn tensor_max(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_max(p(t))))
    }
    fn tensor_min(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_min(p(t))))
    }
    fn tensor_softmax(&self, t: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_softmax(p(t), dim)))
    }
    fn tensor_max_dim(&self, t: *mut c_void, dim: usize, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_max_dim(p(t), dim, kd)))
    }
    fn tensor_min_dim(&self, t: *mut c_void, dim: usize, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_min_dim(p(t), dim, kd)))
    }
    fn tensor_mean_dim(&self, t: *mut c_void, dim: usize, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_mean_dim(p(t), dim, kd)))
    }
    fn tensor_sum_dim(&self, t: *mut c_void, dim: usize, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sum_dim(p(t), dim, kd)))
    }
    fn tensor_argmax(&self, t: *mut c_void, dim: i64, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_argmax(p(t), dim, kd)))
    }
    fn tensor_argmin(&self, t: *mut c_void, dim: i64, kd: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_argmin(p(t), dim, kd)))
    }
    fn tensor_tril(&self, t: *mut c_void, d: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_tril(p(t), d)))
    }
    fn tensor_clamp(&self, t: *mut c_void, min: f64, max: f64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_clamp(p(t), min, max)))
    }
    fn tensor_sample(&self, t: *mut c_void, temp: f32, top_p: f32) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_sample(p(t), temp, top_p)))
    }

    // ========== Autograd ==========
    fn tensor_backward(&self, t: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_backward(p(t));
        Ok(())
    }
    fn tensor_grad(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_grad(p(t))))
    }
    fn tensor_detach(&self, t: *mut c_void, _req_grad: bool) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_detach(p(t))))
    }
    fn tensor_enable_grad(&self, t: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_enable_grad(p(t));
        Ok(())
    }
    fn clear_grads(&self) -> BackendResult<()> {
        Ok(())
    }

    // ========== 形状操作 ==========
    fn tensor_reshape_new(
        &self,
        t: *mut c_void,
        new_shape: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        unsafe {
            let shape_tensor =
                &*(p(new_shape) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let st = &*shape_tensor.get();

            // shape テンソルの dtype に応じて正しい型で読み出す
            // [2, 2] のような整数リテラルは I64 テンソルとして
            // 作成されるため、f32 として読むとゴミ値になる
            let dims: Vec<i64> = match st.dtype() {
                crate::DType::I64 => st.to_vec::<i64>(),
                _ => {
                    let shape_data = st.to_vec::<f32>();
                    shape_data.iter().map(|&x| x as i64).collect()
                }
            };

            Ok(v(ffi_ops::tl_cuda_reshape(p(t), dims.as_ptr(), dims.len())))
        }
    }
    fn tensor_reshape_dims(
        &self,
        t: *mut c_void,
        dims_ptr: *const i64,
        rank: i64,
    ) -> BackendResult<*mut c_void> {
        if dims_ptr.is_null() || rank <= 0 {
            return Err(tl_backend::error::BackendError::ArgumentError(
                "reshape_dims: null dims_ptr or invalid rank".into(),
            ));
        }
        let dims = unsafe { std::slice::from_raw_parts(dims_ptr, rank as usize) };
        Ok(v(ffi_ops::tl_cuda_reshape(p(t), dims.as_ptr(), dims.len())))
    }
    fn tensor_transpose(&self, t: *mut c_void, d0: usize, d1: usize) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_transpose(p(t), d0, d1)))
    }
    fn tensor_slice(
        &self,
        t: *mut c_void,
        dim: i64,
        start: i64,
        len: i64,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_slice(
            p(t),
            dim as usize,
            start as usize,
            len as usize,
        )))
    }
    fn tensor_narrow(
        &self,
        t: *mut c_void,
        dim: usize,
        start: usize,
        len: usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_narrow(p(t), dim, start, len)))
    }
    fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cat(p(a), p(b), dim)))
    }
    fn tensor_cat_i64(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cat_i64(p(a), p(b), dim)))
    }
    fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cat(p(a), p(b), dim)))
    }
    fn tensor_cat_4d(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_cat(p(a), p(b), dim)))
    }
    fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_cuda_replace_data(p(dst), p(src));
        Ok(())
    }
    fn tensor_repeat_interleave(
        &self,
        t: *mut c_void,
        repeats: usize,
        dim: usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_repeat_interleave(p(t), repeats, dim)))
    }
    fn tensor_to_device(&self, t: *mut c_void, device_id: i32) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_to_device(p(t), device_id)))
    }
    fn tensor_to_f32(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_to_f32(p(t))))
    }
    fn tensor_to_i64(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_to_i64(p(t))))
    }
    fn tensor_embedding(
        &self,
        w: *mut c_void,
        idx: *mut c_void,
        p_idx: i64,
        sg: bool,
        sp: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_embedding(p(w), p(idx), p_idx, sg, sp)))
    }

    // ========== LLM ==========
    fn tensor_rms_norm(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        eps: f32,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_rms_norm(p(input), p(weight), eps)))
    }
    fn tensor_rope_new_cos(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_rope_new_cos(dim, seq_len, base)))
    }
    fn tensor_rope_new_sin(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_rope_new_sin(dim, seq_len, base)))
    }
    fn tensor_apply_rope(
        &self,
        t: *mut c_void,
        cos: *mut c_void,
        sin: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        unsafe {
            let tensor = &*(p(t) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let cos_t = &*(p(cos) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let sin_t = &*(p(sin) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let result = (*tensor.get()).apply_rope_impl(&*cos_t.get(), &*sin_t.get(), 0)?;
            Ok(v(ffi_ops::make_tensor(result)))
        }
    }

    // ========== IO / Print ==========
    fn tensor_print(&self, t: *mut c_void) -> BackendResult<()> {
        unsafe {
            let tensor = &*(p(t) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let data = (*tensor.get()).to_vec::<f32>();
            println!(
                "CudaTensor shape={:?} data={:?}",
                (*tensor.get()).shape(),
                &data[..data.len().min(20)]
            );
        }
        Ok(())
    }
    fn tensor_display(&self, t: *mut c_void) -> BackendResult<()> {
        self.tensor_print(t)
    }
    fn tensor_print_1(&self, t: *mut c_void) -> BackendResult<()> {
        self.tensor_print(t)
    }
    fn tensor_print_2(&self, t: *mut c_void) -> BackendResult<()> {
        self.tensor_print(t)
    }
    fn tensor_print_3(&self, t: *mut c_void) -> BackendResult<()> {
        self.tensor_print(t)
    }
    fn tensor_save(&self, t: *mut c_void, path: *const i8) -> BackendResult<()> {
        if t.is_null() || path.is_null() {
            return Ok(());
        }
        unsafe {
            let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy();
            let cell = &*(p(t) as *const std::cell::UnsafeCell<crate::tensor::CudaTensor>);
            let tensor = &*cell.get();
            let data = tensor.to_vec::<f32>();
            let shape = tensor.shape().to_vec();

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
        Ok(())
    }
    fn tensor_load(&self, path: *const i8) -> BackendResult<*mut c_void> {
        if path.is_null() {
            return Ok(v(ffi_ops::make_tensor(crate::tensor::CudaTensor::zeros(
                &[1],
                crate::DType::F32,
            ))));
        }
        unsafe {
            let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy();
            let bytes = match std::fs::read(path_str.as_ref()) {
                Ok(b) => b,
                Err(_) => {
                    return Ok(v(ffi_ops::make_tensor(crate::tensor::CudaTensor::zeros(
                        &[1],
                        crate::DType::F32,
                    ))))
                }
            };
            if bytes.len() < 8 {
                return Ok(v(ffi_ops::make_tensor(crate::tensor::CudaTensor::zeros(
                    &[1],
                    crate::DType::F32,
                ))));
            }
            let mut offset = 0;
            let rank = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            if bytes.len() < offset + rank * 8 {
                return Ok(v(ffi_ops::make_tensor(crate::tensor::CudaTensor::zeros(
                    &[1],
                    crate::DType::F32,
                ))));
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dim =
                    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
                shape.push(dim);
                offset += 8;
            }
            let numel: usize = shape.iter().product();
            let expected = numel * 4;
            if bytes.len() < offset + expected {
                return Ok(v(ffi_ops::make_tensor(crate::tensor::CudaTensor::zeros(
                    &[1],
                    crate::DType::F32,
                ))));
            }
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                data.push(val);
                offset += 4;
            }
            Ok(v(ffi_ops::make_tensor(
                crate::tensor::CudaTensor::from_slice(&data, &shape, crate::DType::F32),
            )))
        }
    }

    // ========== NN ==========
    fn tensor_conv2d(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_conv2d(
            p(input),
            p(weight),
            p(bias),
            stride,
            padding,
            dilation,
            groups,
        )))
    }
    fn tensor_batch_norm(
        &self,
        input: *mut c_void,
        rm: *mut c_void,
        rv: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
        training: bool,
        momentum: f64,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_batch_norm(
            p(input),
            p(rm),
            p(rv),
            p(w),
            p(b),
            training,
            momentum,
            eps,
        )))
    }
    fn tensor_layer_norm(
        &self,
        input: *mut c_void,
        w: *mut c_void,
        b: *mut c_void,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_layer_norm(p(input), p(w), p(b), eps)))
    }
    fn tensor_dropout(
        &self,
        input: *mut c_void,
        p_val: f64,
        training: bool,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_dropout(p(input), p_val, training)))
    }
    fn tensor_max_pool2d(
        &self,
        input: *mut c_void,
        ks: usize,
        s: usize,
        pad: usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_max_pool2d(p(input), ks, s, pad)))
    }
    fn tensor_avg_pool2d(
        &self,
        input: *mut c_void,
        ks: usize,
        s: usize,
        pad: usize,
    ) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_avg_pool2d(p(input), ks, s, pad)))
    }

    // ========== 次元特化メソッド ==========
    fn tensor_transpose_2d(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_transpose(p(t), 0, 1)))
    }
    fn tensor_reshape_2d(&self, t: *mut c_void, d0: i64, d1: i64) -> BackendResult<*mut c_void> {
        let dims = [d0, d1];
        Ok(v(ffi_ops::tl_cuda_reshape(p(t), dims.as_ptr(), 2)))
    }
    fn tensor_reshape_3d_to_2d(
        &self,
        t: *mut c_void,
        d0: i64,
        d1: i64,
    ) -> BackendResult<*mut c_void> {
        let dims = [d0, d1];
        Ok(v(ffi_ops::tl_cuda_reshape(p(t), dims.as_ptr(), 2)))
    }
    fn tensor_matmul_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_matmul(p(a), p(b))))
    }
    fn tensor_add_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_add(p(a), p(b))))
    }
    fn tensor_silu_4d(&self, t: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(v(ffi_ops::tl_cuda_silu(p(t))))
    }
}
