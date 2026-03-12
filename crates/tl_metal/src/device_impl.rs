//! MetalDeviceImpl: IDevice トレイトの Metal 実装
//!
//! 既存の `ffi_ops::tl_metal_*` 関数に委譲する。
//! Metal FFI と IDevice のシグネチャ差はここで吸収する。

use crate::ffi;
use crate::ffi_ops;
use crate::tensor::MetalTensor;
use std::ffi::c_void;
use tl_backend::{BackendError, BackendResult, GpuTensor, IDevice};

/// Metal デバイス (ゼロサイズ型)
pub struct MetalDeviceImpl;

/// void* → *mut MetalTensor
#[inline(always)]
fn t(p: *mut c_void) -> *mut MetalTensor {
    p as *mut MetalTensor
}

/// *mut MetalTensor → BackendResult<void*>
/// Checks for null pointer from FFI and returns Error if null.
#[inline(always)]
fn v(p: *mut MetalTensor) -> BackendResult<*mut c_void> {
    if p.is_null() {
        Err(BackendError::InternalError(
            "Metal Backend FFI returned null pointer (operation failed)".into(),
        ))
    } else {
        Ok(p as *mut c_void)
    }
}

impl IDevice for MetalDeviceImpl {
    // ========== テンソル作成 ==========
    #[inline]
    fn tensor_new(
        &self,
        data: *const f32,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_new(data, rank, shape))
    }
    #[inline]
    fn tensor_new_i64(
        &self,
        data: *const i64,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_new_i64(data, rank, shape))
    }
    #[inline]
    fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_from_i64_array(data, len))
    }
    #[inline]
    fn tensor_zeros(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_zeros(rank, shape, req_grad))
    }
    #[inline]
    fn tensor_ones(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_ones(rank, shape, req_grad))
    }
    #[inline]
    fn tensor_randn_debug(
        &self,
        rank: usize,
        shape: *const usize,
        seed: u64,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_randn_debug(rank, shape, seed, req_grad))
    }
    #[inline]
    fn tensor_new_causal_mask(&self, size: usize) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_new_causal_mask(size))
    }
    #[inline]
    fn tensor_from_vec_u8(&self, data: *mut c_void, len: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_from_vec_u8(data, len))
    }
    #[inline]
    fn tensor_from_u8_labels(&self, data: *const u8, len: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_from_u8_labels(data, len))
    }
    #[inline]
    fn tensor_full(
        &self,
        rank: usize,
        shape: *const usize,
        value: f32,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_full(rank, shape, value, req_grad))
    }
    #[inline]
    fn tensor_eye(&self, n: usize, req_grad: bool) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_eye(n, req_grad))
    }
    #[inline]
    fn tensor_arange(&self, start: f64, end: f64, step: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_arange(start, end, step))
    }
    #[inline]
    fn tensor_linspace(&self, start: f64, end: f64, steps: usize) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_linspace(start, end, steps))
    }
    #[inline]
    fn tensor_rand(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_rand(rank as i64, shape, req_grad))
    }
    // ========== 要素操作 ==========
    fn tensor_where_cond(
        &self,
        cond: *mut c_void,
        x: *mut c_void,
        y: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tc, tx, ty) = unsafe { (&*t(cond), &*t(x), &*t(y)) };
        let result = MetalTensor::where_cond(tc, tx, ty)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_masked_fill(
        &self,
        tensor: *mut c_void,
        mask: *mut c_void,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        let (tt, tm) = unsafe { (&*t(tensor), &*t(mask)) };
        let numel: usize = tt.shape().iter().product();
        let value_tensor =
            MetalTensor::from_slice(&vec![value; numel], tt.shape(), crate::DType::F32);
        let result = MetalTensor::where_cond(tm, &value_tensor, tt)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_to_vec_f32(&self, tensor: *mut c_void) -> BackendResult<(*mut f32, usize)> {
        let tt = unsafe { &*t(tensor) };
        let vec: Vec<f32> = tt.to_vec::<f32>();
        let len = vec.len();
        let ptr = vec.as_ptr() as *mut f32;
        std::mem::forget(vec);
        Ok((ptr, len))
    }

    fn tensor_var(&self, tensor: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let mean = tt.mean_impl(dim)?;
        let diff = tt.sub_impl(&mean)?;
        let sq = diff.mul_impl(&diff)?;
        let result = sq.mean_impl(dim)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_std(&self, tensor: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let var_ptr = self.tensor_var(tensor, dim)?;
        let var_t = unsafe { &*(var_ptr as *mut MetalTensor) };
        let result = var_t.sqrt_impl()?;
        unsafe {
            let _ = std::sync::Arc::from_raw(var_ptr as *const std::cell::UnsafeCell<MetalTensor>);
        }
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_prod(&self, tensor: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let log_t = tt.log_impl()?;
        let sum_log = log_t.sum_impl(dim)?;
        let result = sum_log.exp_impl()?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_cumsum(&self, tensor: *mut c_void, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0f32;
        for &x in &data {
            acc += x;
            result.push(acc);
        }
        let out = MetalTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        Ok(ffi_ops::make_tensor(out) as *mut c_void)
    }

    fn tensor_norm(&self, tensor: *mut c_void, p: f32, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let norm_val = if p == 2.0 {
            data.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            data.iter()
                .map(|x| x.abs().powf(p))
                .sum::<f32>()
                .powf(1.0 / p)
        };
        let out = MetalTensor::from_slice(&[norm_val], &[1], crate::DType::F32);
        Ok(ffi_ops::make_tensor(out) as *mut c_void)
    }

    fn tensor_topk(&self, tensor: *mut c_void, k: usize, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let mut data = tt.to_vec::<f32>();
        data.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        data.truncate(k);
        let out = MetalTensor::from_slice(&data, &[k], crate::DType::F32);
        Ok(ffi_ops::make_tensor(out) as *mut c_void)
    }

    fn tensor_logical_and(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let r: Vec<f32> = ta
            .to_vec::<f32>()
            .iter()
            .zip(tb.to_vec::<f32>().iter())
            .map(|(&x, &y)| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 })
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, ta.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_logical_or(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let r: Vec<f32> = ta
            .to_vec::<f32>()
            .iter()
            .zip(tb.to_vec::<f32>().iter())
            .map(|(&x, &y)| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 })
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, ta.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_logical_not(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| if x == 0.0 { 1.0 } else { 0.0 })
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_leaky_relu(
        &self,
        tensor: *mut c_void,
        negative_slope: f32,
    ) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_elu(&self, tensor: *mut c_void, alpha: f32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_mish(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| x * (x.exp().ln_1p()).tanh())
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_mse_loss(
        &self,
        pred: *mut c_void,
        target: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let diff = tp.sub_impl(tt)?;
        let sq = diff.mul_impl(&diff)?;
        let result = sq.mean_impl(-1)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_l1_loss(&self, pred: *mut c_void, target: *mut c_void) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let diff = tp.sub_impl(tt)?;
        let abs_diff = diff.abs_impl()?;
        let result = abs_diff.mean_impl(-1)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_bce_loss(
        &self,
        pred: *mut c_void,
        target: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let p = tp.to_vec::<f32>();
        let y = tt.to_vec::<f32>();
        let eps = 1e-7f32;
        let sum: f32 = p
            .iter()
            .zip(y.iter())
            .map(|(&pi, &yi)| -(yi * (pi + eps).ln() + (1.0 - yi) * (1.0 - pi + eps).ln()))
            .sum();
        let loss = sum / p.len() as f32;
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&[loss], &[1], crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_nll_loss(
        &self,
        pred: *mut c_void,
        target: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let p = tp.to_vec::<f32>();
        let y = tt.to_vec::<f32>();
        let sum: f32 = p.iter().zip(y.iter()).map(|(&pi, &yi)| -pi * yi).sum();
        let loss = sum / p.len() as f32;
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&[loss], &[1], crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_linear(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*t(input), &*t(weight)) };
        let result = ti.matmul_impl(&tw.transpose_impl(0, 1)?)?;
        if !bias.is_null() {
            let tb = unsafe { &*t(bias) };
            let out = result.add_impl(tb)?;
            Ok(ffi_ops::make_tensor(out) as *mut c_void)
        } else {
            Ok(ffi_ops::make_tensor(result) as *mut c_void)
        }
    }

    fn tensor_hardswish(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| x * (x + 3.0).max(0.0).min(6.0) / 6.0)
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_hardsigmoid(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let r: Vec<f32> = tt
            .to_vec::<f32>()
            .iter()
            .map(|&x| ((x + 3.0) / 6.0).max(0.0).min(1.0))
            .collect();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&r, tt.shape(), crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_group_norm(
        &self,
        input: *mut c_void,
        num_groups: i64,
        weight: *mut c_void,
        bias: *mut c_void,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        let data = ti.to_vec::<f32>();
        let shape = ti.shape().to_vec();
        let channels = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let group_size = channels / num_groups as usize;
        let mut result = data.clone();
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);
        let batch = if shape.len() >= 2 { shape[0] } else { 1 };
        for b in 0..batch {
            for g in 0..num_groups as usize {
                let (start_c, end_c) = (g * group_size, g * group_size + group_size);
                let mut vals = Vec::new();
                for c in start_c..end_c {
                    for s in 0..spatial {
                        vals.push(data[b * channels * spatial + c * spatial + s]);
                    }
                }
                let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
                let var: f32 =
                    vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
                let std = (var + eps as f32).sqrt();
                for c in start_c..end_c {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let mut v = (result[idx] - mean) / std;
                        if !weight.is_null() {
                            v *= unsafe { &*t(weight) }.to_vec::<f32>()[c];
                        }
                        if !bias.is_null() {
                            v += unsafe { &*t(bias) }.to_vec::<f32>()[c];
                        }
                        result[idx] = v;
                    }
                }
            }
        }
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&result, &shape, crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_adaptive_avg_pool2d(
        &self,
        input: *mut c_void,
        output_h: i64,
        output_w: i64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        let data = ti.to_vec::<f32>();
        let shape = ti.shape().to_vec();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = (output_h as usize, output_w as usize);
        let mut result = vec![0.0f32; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let (hs, he) = (oi * h / oh, (oi + 1) * h / oh);
                        let (ws, we) = (oj * w / ow, (oj + 1) * w / ow);
                        let mut sum = 0.0f32;
                        let mut cnt = 0;
                        for hi in hs..he {
                            for wi in ws..we {
                                sum += data[bi * c * h * w + ci * h * w + hi * w + wi];
                                cnt += 1;
                            }
                        }
                        result[bi * c * oh * ow + ci * oh * ow + oi * ow + oj] = sum / cnt as f32;
                    }
                }
            }
        }
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            &[n, c, oh, ow],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_pad(
        &self,
        input: *mut c_void,
        pad_left: i64,
        pad_right: i64,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        let data = ti.to_vec::<f32>();
        let shape = ti.shape().to_vec();
        let last = *shape.last().unwrap();
        let new_last = last + pad_left as usize + pad_right as usize;
        let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let mut result = Vec::with_capacity(outer * new_last);
        for o in 0..outer {
            for _ in 0..pad_left {
                result.push(value);
            }
            for i in 0..last {
                result.push(data[o * last + i]);
            }
            for _ in 0..pad_right {
                result.push(value);
            }
        }
        let mut new_shape = shape;
        *new_shape.last_mut().unwrap() = new_last;
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            &new_shape,
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_instance_norm(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        let channels = if ti.shape().len() >= 2 {
            ti.shape()[1]
        } else {
            ti.shape()[0]
        };
        self.tensor_group_norm(input, channels as i64, weight, bias, eps)
    }

    fn tensor_dropout2d(
        &self,
        input: *mut c_void,
        _p: f64,
        _training: bool,
    ) -> BackendResult<*mut c_void> {
        // Metal: passthrough (no random on GPU, use CPU for training dropout)
        let ti = unsafe { &*t(input) };
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &ti.to_vec::<f32>(),
            ti.shape(),
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_conv1d(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        stride: i64,
        padding: i64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*t(input), &*t(weight)) };
        let inp = ti.to_vec::<f32>();
        let w = tw.to_vec::<f32>();
        let ishape = ti.shape().to_vec();
        let wshape = tw.shape().to_vec();
        let (batch, in_ch, in_len) = (ishape[0], ishape[1], ishape[2]);
        let (out_ch, _wch, k_len) = (wshape[0], wshape[1], wshape[2]);
        let (st, pad) = (stride as usize, padding as usize);
        let out_len = (in_len + 2 * pad - k_len) / st + 1;
        let mut result = vec![0.0f32; batch * out_ch * out_len];
        for b in 0..batch {
            for oc in 0..out_ch {
                for ol in 0..out_len {
                    let mut sum = 0.0f32;
                    for ic in 0..in_ch {
                        for ki in 0..k_len {
                            let pos = ol * st + ki;
                            if pos >= pad && pos < in_len + pad {
                                sum += inp[b * in_ch * in_len + ic * in_len + (pos - pad)]
                                    * w[oc * in_ch * k_len + ic * k_len + ki];
                            }
                        }
                    }
                    if !bias.is_null() {
                        sum += unsafe { &*t(bias) }.to_vec::<f32>()[oc];
                    }
                    result[b * out_ch * out_len + oc * out_len + ol] = sum;
                }
            }
        }
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            &[batch, out_ch, out_len],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_kl_div_loss(
        &self,
        pred: *mut c_void,
        target: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let (p, q) = (tp.to_vec::<f32>(), tt.to_vec::<f32>());
        let eps = 1e-7f32;
        let sum: f32 = q
            .iter()
            .zip(p.iter())
            .map(|(&qi, &pi)| {
                if qi > eps {
                    qi * ((qi + eps) / (pi + eps)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &[sum / q.len() as f32],
            &[1],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_conv_transpose2d(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        stride: i64,
        padding: i64,
        output_padding: i64,
    ) -> BackendResult<*mut c_void> {
        let (ti, tw) = unsafe { (&*t(input), &*t(weight)) };
        let inp = ti.to_vec::<f32>();
        let w = tw.to_vec::<f32>();
        let ishape = ti.shape().to_vec();
        let wshape = tw.shape().to_vec();
        let (batch, in_ch, ih, iw) = (ishape[0], ishape[1], ishape[2], ishape[3]);
        let (_, out_ch, kh, kw) = (wshape[0], wshape[1], wshape[2], wshape[3]);
        let (st, pad, opad) = (stride as usize, padding as usize, output_padding as usize);
        let oh = (ih - 1) * st - 2 * pad + kh + opad;
        let ow = (iw - 1) * st - 2 * pad + kw + opad;
        let mut result = vec![0.0f32; batch * out_ch * oh * ow];
        for b in 0..batch {
            for ic in 0..in_ch {
                for iy in 0..ih {
                    for ix in 0..iw {
                        let v = inp[b * in_ch * ih * iw + ic * ih * iw + iy * iw + ix];
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oy = iy * st + ky;
                                let ox = ix * st + kx;
                                if oy >= pad && ox >= pad && oy - pad < oh && ox - pad < ow {
                                    for oc in 0..out_ch {
                                        result[b * out_ch * oh * ow
                                            + oc * oh * ow
                                            + (oy - pad) * ow
                                            + (ox - pad)] += v * w
                                            [ic * out_ch * kh * kw + oc * kh * kw + ky * kw + kx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if !bias.is_null() {
            let tb = unsafe { &*t(bias) }.to_vec::<f32>();
            for b in 0..batch {
                for oc in 0..out_ch {
                    for y in 0..oh {
                        for x in 0..ow {
                            result[b * out_ch * oh * ow + oc * oh * ow + y * ow + x] += tb[oc];
                        }
                    }
                }
            }
        }
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            &[batch, out_ch, oh, ow],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_interpolate(
        &self,
        input: *mut c_void,
        output_h: i64,
        output_w: i64,
        mode: i64,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        let data = ti.to_vec::<f32>();
        let shape = ti.shape().to_vec();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = (output_h as usize, output_w as usize);
        let mut result = vec![0.0f32; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let val = if mode == 0 {
                            let si = (oi * h) / oh;
                            let sj = (oj * w) / ow;
                            data[bi * c * h * w + ci * h * w + si * w + sj]
                        } else {
                            let sy = oi as f32 * (h as f32 - 1.0) / (oh as f32 - 1.0).max(1.0);
                            let sx = oj as f32 * (w as f32 - 1.0) / (ow as f32 - 1.0).max(1.0);
                            let (y0, x0) = (sy.floor() as usize, sx.floor() as usize);
                            let (y1, x1) = ((y0 + 1).min(h - 1), (x0 + 1).min(w - 1));
                            let (fy, fx) = (sy - y0 as f32, sx - x0 as f32);
                            let base = bi * c * h * w + ci * h * w;
                            data[base + y0 * w + x0] * (1.0 - fy) * (1.0 - fx)
                                + data[base + y0 * w + x1] * (1.0 - fy) * fx
                                + data[base + y1 * w + x0] * fy * (1.0 - fx)
                                + data[base + y1 * w + x1] * fy * fx
                        };
                        result[bi * c * oh * ow + ci * oh * ow + oi * ow + oj] = val;
                    }
                }
            }
        }
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            &[n, c, oh, ow],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_scaled_dot_product_attention(
        &self,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        mask: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tq, tk, tv) = unsafe { (&*t(q), &*t(k), &*t(v)) };
        let qs = tq.shape().to_vec();
        let d_k = *qs.last().unwrap_or(&1);
        let scale = 1.0 / (d_k as f32).sqrt();

        // K^T: transpose last two dims
        let ndim = tk.shape().len();
        let kt = tk.transpose_impl(ndim - 2, ndim - 1)?;

        // scores = Q × K^T
        let scores = tq.matmul_impl(&kt)?;

        // scores = scores * scale
        let scaled = scores.mul_scalar_impl(scale)?;

        // mask 適用: mask==0 の位置を -1e9 で置換
        let masked = if !mask.is_null() {
            let tm = unsafe { &*t(mask) };
            let numel: usize = scaled.shape().iter().product();
            let neg_inf =
                MetalTensor::from_slice(&vec![-1e9f32; numel], scaled.shape(), crate::DType::F32);
            // where_cond: condition > 0 → scaled, else → -1e9
            MetalTensor::where_cond(tm, &scaled, &neg_inf)?
        } else {
            scaled
        };

        // softmax on last dim
        let attn = masked.softmax_impl(-1)?;

        // output = attn × V
        let output = attn.matmul_impl(tv)?;

        Ok(ffi_ops::make_tensor(output) as *mut c_void)
    }

    fn tensor_top_k_sample(&self, logits: *mut c_void, k: i64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        let k = k as usize;
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let thr = if k < sorted.len() {
            sorted[k - 1]
        } else {
            sorted[sorted.len() - 1]
        };
        let masked: Vec<f32> = data
            .iter()
            .map(|&v| if v >= thr { v } else { f32::NEG_INFINITY })
            .collect();
        let mx: f32 = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = masked.iter().map(|&v| (v - mx).exp()).collect();
        let sm: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|v| v / sm).collect();
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &[idx as f32],
            &[1],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_top_p_sample(&self, logits: *mut c_void, p: f64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        let mx: f32 = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|&v| (v - mx).exp()).collect();
        let sm: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|v| v / sm).collect();
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut mask = vec![false; probs.len()];
        for &i in &indices {
            cum += probs[i];
            mask[i] = true;
            if cum >= p as f32 {
                break;
            }
        }
        let masked: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &v)| if mask[i] { v } else { 0.0 })
            .collect();
        let sm2: f32 = masked.iter().sum();
        let fp: Vec<f32> = masked.iter().map(|v| v / sm2).collect();
        let idx = fp
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &[idx as f32],
            &[1],
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_temperature_scale(
        &self,
        logits: *mut c_void,
        temperature: f64,
    ) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        let temp = temperature as f32;
        let result: Vec<f32> = data.iter().map(|&v| v / temp).collect();
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &result,
            tl.shape(),
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_repetition_penalty(
        &self,
        logits: *mut c_void,
        tokens: *mut c_void,
        penalty: f64,
    ) -> BackendResult<*mut c_void> {
        let (tl, tt) = unsafe { (&*t(logits), &*t(tokens)) };
        let mut data = tl.to_vec::<f32>();
        let tok = tt.to_vec::<f32>();
        let p = penalty as f32;
        for &idx in &tok {
            let i = idx as usize;
            if i < data.len() {
                data[i] = if data[i] > 0.0 {
                    data[i] / p
                } else {
                    data[i] * p
                };
            }
        }
        Ok(ffi_ops::make_tensor(MetalTensor::from_slice(
            &data,
            tl.shape(),
            crate::DType::F32,
        )) as *mut c_void)
    }

    fn tensor_dot(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let dot: f32 = ta
            .to_vec::<f32>()
            .iter()
            .zip(tb.to_vec::<f32>().iter())
            .map(|(&x, &y)| x * y)
            .sum();
        Ok(
            ffi_ops::make_tensor(MetalTensor::from_slice(&[dot], &[1], crate::DType::F32))
                as *mut c_void,
        )
    }

    fn tensor_fill_(&self, tensor: *mut c_void, value: f32) -> BackendResult<()> {
        let tt = unsafe { &*t(tensor) };
        let numel: usize = tt.shape().iter().product();
        // Create filled tensor and copy data
        let filled = MetalTensor::from_slice(&vec![value; numel], tt.shape(), crate::DType::F32);
        unsafe {
            let dst = &mut *(tensor as *mut MetalTensor);
            *dst = filled;
        }
        Ok(())
    }

    fn tensor_broadcast_to(
        &self,
        tensor: *mut c_void,
        shape: &[usize],
    ) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let result = tt.broadcast_to(shape)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    fn tensor_stack(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let ua = ta.unsqueeze(dim as usize)?;
        let ub = tb.unsqueeze(dim as usize)?;
        let result = MetalTensor::cat(&[&ua, &ub], dim as usize)?;
        Ok(ffi_ops::make_tensor(result) as *mut c_void)
    }

    // ========== メモリ管理 ==========
    #[inline]
    fn tensor_clone(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi::tl_metal_clone(t(a)))
    }
    #[inline]
    fn tensor_shallow_clone(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi::tl_metal_shallow_clone(t(a)))
    }
    #[inline]
    fn tensor_free(&self, a: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_free(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_release(&self, a: *mut c_void) -> BackendResult<()> {
        // V6.0: Arc RC-1 (直接 Arc::from_raw → drop)
        crate::ffi_ops::release_if_live(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_acquire(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        // V6.0: Arc RC+1 (同一ポインタの参照カウントを増やす)
        crate::ffi_ops::acquire_tensor(t(a));
        Ok(a) // 同じポインタを返す
    }
    #[inline]
    fn tensor_release_safe(&self, a: *mut c_void) -> BackendResult<()> {
        // V6.0: Arc RC-1 (直接 Arc::from_raw → drop)
        crate::ffi_ops::release_if_live(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_promote(&self, _a: *mut c_void) -> BackendResult<()> {
        /* Metal: pool-based — runtime 側で処理 */
        Ok(())
    }
    #[inline]
    fn tensor_register(&self, _a: *mut c_void) -> BackendResult<()> {
        /* Metal: noop */
        Ok(())
    }
    #[inline]
    fn tensor_prepare_return(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        Ok(a)
    }

    // ========== テンソル情報 ==========
    #[inline]
    fn tensor_len(&self, a: *mut c_void) -> BackendResult<usize> {
        Ok(ffi_ops::tl_metal_len(t(a)))
    }
    #[inline]
    fn tensor_dim(&self, a: *mut c_void, dim: usize) -> BackendResult<usize> {
        Ok(ffi_ops::tl_metal_dim(t(a), dim))
    }
    #[inline]
    fn tensor_numel(&self, a: *mut c_void) -> BackendResult<i64> {
        Ok(ffi::tl_metal_numel(t(a)))
    }
    #[inline]
    fn tensor_data(&self, a: *mut c_void) -> BackendResult<*const f32> {
        Ok(ffi::tl_metal_data(t(a)) as *const f32)
    }
    #[inline]
    fn tensor_device_id(&self, a: *mut c_void) -> BackendResult<i32> {
        Ok(ffi_ops::tl_metal_device_id(t(a)))
    }
    #[inline]
    fn tensor_get_shape(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_get_shape(t(a)))
    }

    // ========== 要素アクセス ==========
    #[inline]
    fn tensor_get(&self, a: *mut c_void, idx: i64) -> BackendResult<f32> {
        Ok(ffi_ops::tl_metal_get(t(a), idx))
    }
    #[inline]
    fn tensor_get_f32_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<f32> {
        // Metal の tl_tensor_get_f32_md は (t, idx0, idx1) シグネチャ
        // IDevice では (t, *const i64, rank) — ここでアダプト
        let slice = unsafe { std::slice::from_raw_parts(indices, rank as usize) };
        if slice.len() >= 2 {
            Ok(ffi_ops::tl_metal_get_f32_md(t(a), slice[0], slice[1]))
        } else if slice.len() == 1 {
            Ok(ffi_ops::tl_metal_get_f32_md(t(a), slice[0], 0))
        } else {
            Ok(0.0)
        }
    }
    #[inline]
    fn tensor_get_i64_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<i64> {
        let slice = unsafe { std::slice::from_raw_parts(indices, rank as usize) };
        if slice.len() >= 2 {
            Ok(ffi_ops::tl_metal_get_i64_md(t(a), slice[0], slice[1]))
        } else if slice.len() == 1 {
            Ok(ffi_ops::tl_metal_get_i64_md(t(a), slice[0], 0))
        } else {
            Ok(0)
        }
    }
    #[inline]
    fn tensor_set_f32_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: usize,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_set_f32_md(t(a), indices, rank, value))
    }
    #[inline]
    fn tensor_item(&self, a: *mut c_void) -> BackendResult<f32> {
        Ok(ffi_ops::tl_metal_item(t(a)))
    }
    #[inline]
    fn tensor_item_i64(&self, a: *mut c_void) -> BackendResult<i64> {
        Ok(ffi_ops::tl_metal_item_i64(t(a)))
    }

    // ========== 二項演算 ==========
    #[inline]
    fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_add(t(a), t(b)))
    }
    #[inline]
    fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sub(t(a), t(b)))
    }
    #[inline]
    fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_mul(t(a), t(b)))
    }
    #[inline]
    fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_div(t(a), t(b)))
    }
    #[inline]
    fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_rem(t(a), t(b)))
    }
    #[inline]
    fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_matmul(t(a), t(b)))
    }
    #[inline]
    fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_pow(t(a), t(b)))
    }
    #[inline]
    fn tensor_cross_entropy(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cross_entropy(t(a), t(b)))
    }

    // ========== 単項演算 ==========
    #[inline]
    fn tensor_neg(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_neg(t(a)))
    }
    #[inline]
    fn tensor_abs(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_abs(t(a)))
    }
    #[inline]
    fn tensor_contiguous(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_contiguous(t(a)))
    }

    // ========== 比較演算 ==========
    #[inline]
    fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_eq(t(a), t(b)))
    }
    #[inline]
    fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_neq(t(a), t(b)))
    }
    #[inline]
    fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_gt(t(a), t(b)))
    }
    #[inline]
    fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_lt(t(a), t(b)))
    }
    #[inline]
    fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_ge(t(a), t(b)))
    }
    #[inline]
    fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_le(t(a), t(b)))
    }

    // ========== スカラー演算 ==========
    #[inline]
    fn tensor_add_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_add_scalar(t(a), s))
    }
    #[inline]
    fn tensor_sub_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sub_scalar(t(a), s))
    }
    #[inline]
    fn tensor_mul_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_mul_scalar(t(a), s))
    }
    #[inline]
    fn tensor_div_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_div_scalar(t(a), s))
    }
    #[inline]
    fn tensor_pow_scalar(&self, a: *mut c_void, exp: f32) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_pow_scalar(t(a), exp as f64))
    }
    #[inline]
    fn tensor_scale(&self, a: *mut c_void, s: f32) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_scale(t(a), s))
    }

    // ========== インプレース演算 ==========
    #[inline]
    fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_add_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_sub_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_mul_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_div_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_mod_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_add_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_metal_add_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_sub_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_metal_sub_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_mul_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_metal_mul_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_div_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_metal_div_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_mod_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi_ops::tl_metal_mod_assign_scalar_f32(t(a), s);
        Ok(())
    }

    // ========== 数学・活性化関数 ==========
    #[inline]
    fn tensor_exp(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_exp(t(a)))
    }
    #[inline]
    fn tensor_log(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_log(t(a)))
    }
    #[inline]
    fn tensor_sqrt(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sqrt(t(a)))
    }
    #[inline]
    fn tensor_sin(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sin(t(a)))
    }
    #[inline]
    fn tensor_cos(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cos(t(a)))
    }
    #[inline]
    fn tensor_tan(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_tan(t(a)))
    }
    #[inline]
    fn tensor_tanh(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_tanh(t(a)))
    }
    #[inline]
    fn tensor_sigmoid(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sigmoid(t(a)))
    }
    #[inline]
    fn tensor_relu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_relu(t(a)))
    }
    #[inline]
    fn tensor_gelu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_gelu(t(a)))
    }
    #[inline]
    fn tensor_silu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_silu(t(a)))
    }

    // ========== Reduction ==========
    #[inline]
    fn tensor_sum(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sum(t(a)))
    }
    #[inline]
    fn tensor_mean(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_mean(t(a)))
    }
    #[inline]
    fn tensor_max(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_max(t(a)))
    }
    #[inline]
    fn tensor_min(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_min(t(a)))
    }
    #[inline]
    fn tensor_softmax(&self, a: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_softmax(t(a), dim))
    }
    #[inline]
    fn tensor_max_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_max_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_min_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_min_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_mean_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_mean_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_sum_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sum_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_argmax(
        &self,
        a: *mut c_void,
        dim: i64,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_argmax(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_argmin(
        &self,
        a: *mut c_void,
        dim: i64,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_argmin(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_tril(&self, a: *mut c_void, diagonal: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_tril(t(a), diagonal))
    }
    #[inline]
    fn tensor_clamp(&self, a: *mut c_void, min: f64, max: f64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_clamp(t(a), min, max))
    }
    #[inline]
    fn tensor_sample(&self, a: *mut c_void, temp: f32, top_p: f32) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_sample(t(a), temp, top_p))
    }

    // ========== Autograd ==========
    #[inline]
    fn tensor_backward(&self, a: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_backward(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_grad(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_grad(t(a)))
    }
    #[inline]
    fn tensor_detach(&self, a: *mut c_void, _req_grad: bool) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_detach(t(a)))
    }
    #[inline]
    fn tensor_enable_grad(&self, a: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_enable_grad(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_set_requires_grad(&self, a: *mut c_void, req_grad: bool) -> BackendResult<()> {
        ffi_ops::tl_metal_set_requires_grad(t(a), req_grad);
        Ok(())
    }
    #[inline]
    fn tensor_clip_grad_value(&self, a: *mut c_void, min: f64, max: f64) -> BackendResult<()> {
        ffi_ops::tl_metal_clip_grad_value(t(a), min, max);
        Ok(())
    }
    #[inline]
    fn tensor_clip_grad_norm(
        &self,
        a: *mut c_void,
        max_norm: f64,
        norm_type: f64,
    ) -> BackendResult<f64> {
        Ok(ffi_ops::tl_metal_clip_grad_norm(t(a), max_norm, norm_type))
    }
    #[inline]
    fn clear_grads(&self) -> BackendResult<()> {
        // 各 for iteration 終了時にコマンドストリームを同期し、
        // Metal ドライバにメモリ回収の機会を与える。
        crate::command_stream::sync_stream();
        Ok(())
    }

    // ========== 形状操作 ==========
    #[inline]
    fn tensor_reshape_new(&self, a: *mut c_void, s: *mut c_void) -> BackendResult<*mut c_void> {
        // Metal の reshape_new は (t, *const i64, usize) — IDevice では (t, *mut c_void) で shape テンソルを渡す
        // runtime 経由で使われるので、ここでは shape tensor から情報を取る必要がある
        // 簡易実装: shape tensor のデータポインタと長さを使う
        if s.is_null() {
            return Ok(a);
        }
        let shape_t = unsafe { &*t(s) };
        let shape_data = shape_t.to_vec_i64();
        let rank = shape_data.len();
        v(ffi_ops::tl_metal_reshape_new(
            t(a),
            shape_data.as_ptr(),
            rank,
        ))
    }
    #[inline]
    fn tensor_reshape_dims(
        &self,
        a: *mut c_void,
        dims_ptr: *const i64,
        rank: i64,
    ) -> BackendResult<*mut c_void> {
        let dims = unsafe { std::slice::from_raw_parts(dims_ptr, rank as usize) };
        v(ffi_ops::tl_metal_reshape_dims(
            t(a),
            dims.as_ptr(),
            dims.len(),
        ))
    }
    #[inline]
    fn tensor_transpose(
        &self,
        a: *mut c_void,
        dim0: usize,
        dim1: usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_transpose(t(a), dim0, dim1))
    }
    #[inline]
    fn tensor_slice(
        &self,
        a: *mut c_void,
        dim: i64,
        start: i64,
        len: i64,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_slice(
            t(a),
            dim as usize,
            start as usize,
            len as usize,
        ))
    }
    #[inline]
    fn tensor_narrow(
        &self,
        a: *mut c_void,
        dim: usize,
        start: usize,
        len: usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_narrow(t(a), dim, start, len))
    }
    #[inline]
    fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cat(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat_i64(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cat_i64(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cat(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat_4d(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_cat(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void) -> BackendResult<()> {
        ffi_ops::tl_metal_replace_data(t(dst), t(src));
        Ok(())
    }
    #[inline]
    fn tensor_repeat_interleave(
        &self,
        a: *mut c_void,
        repeats: usize,
        dim: usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_repeat_interleave(t(a), repeats, dim))
    }
    #[inline]
    fn tensor_to_device(&self, a: *mut c_void, device_id: i32) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_to_device(t(a), device_id))
    }
    #[inline]
    fn tensor_to_f32(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_to_f32(t(a)))
    }
    #[inline]
    fn tensor_to_i64(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_to_i64(t(a)))
    }
    #[inline]
    fn tensor_embedding(
        &self,
        w: *mut c_void,
        idx: *mut c_void,
        pad: i64,
        sg: bool,
        sp: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_embedding(t(w), t(idx), pad, sg, sp))
    }

    // ========== LLM ==========
    #[inline]
    fn tensor_rms_norm(
        &self,
        a: *mut c_void,
        w: *mut c_void,
        eps: f32,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_rms_norm(t(a), t(w), eps))
    }
    #[inline]
    fn tensor_rope_new_cos(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_rope_new_cos(dim, seq_len, base))
    }
    #[inline]
    fn tensor_rope_new_sin(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_rope_new_sin(dim, seq_len, base))
    }
    #[inline]
    fn tensor_apply_rope(
        &self,
        a: *mut c_void,
        cos: *mut c_void,
        sin: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        if a.is_null() || cos.is_null() || sin.is_null() {
            return Ok(a);
        }
        let tensor = unsafe { &*t(a) };
        let c = unsafe { &*t(cos) };
        let s = unsafe { &*t(sin) };
        let result = tensor.apply_rope_impl(c, s, 0);
        v(ffi_ops::make_tensor(result?))
    }

    // ========== IO / Print ==========
    #[inline]
    fn tensor_print(&self, a: *mut c_void) -> BackendResult<()> {
        // Metal: runtime の print_ffi を使う
        if !a.is_null() {
            let tensor = unsafe { &*t(a) };
            println!("{:?}", tensor.to_vec_f32());
        }
        Ok(())
    }
    #[inline]
    fn tensor_display(&self, a: *mut c_void) -> BackendResult<()> {
        self.tensor_print(a)
    }
    #[inline]
    fn tensor_print_1(&self, a: *mut c_void) -> BackendResult<()> {
        self.tensor_print(a)
    }
    #[inline]
    fn tensor_print_2(&self, a: *mut c_void) -> BackendResult<()> {
        self.tensor_print(a)
    }
    #[inline]
    fn tensor_print_3(&self, a: *mut c_void) -> BackendResult<()> {
        self.tensor_print(a)
    }
    #[inline]
    fn tensor_save(&self, a: *mut c_void, path: *const i8) -> BackendResult<()> {
        if a.is_null() || path.is_null() {
            return Ok(());
        }
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
                return Err(BackendError::InternalError(format!(
                    "Failed to save tensor: {}",
                    e
                )));
            }
        }
        Ok(())
    }
    #[inline]
    fn tensor_load(&self, path: *const i8) -> BackendResult<*mut c_void> {
        if path.is_null() {
            return v(ffi_ops::make_tensor(MetalTensor::zeros(
                &[1],
                crate::DType::F32,
            )));
        }
        unsafe {
            let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy();
            let bytes = match std::fs::read(path_str.as_ref()) {
                Ok(b) => b,
                Err(_) => {
                    return v(ffi_ops::make_tensor(MetalTensor::zeros(
                        &[1],
                        crate::DType::F32,
                    )))
                }
            };
            if bytes.len() < 8 {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(
                    &[1],
                    crate::DType::F32,
                )));
            }
            let mut offset = 0;
            let rank = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            if bytes.len() < offset + rank * 8 {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(
                    &[1],
                    crate::DType::F32,
                )));
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let dim =
                    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
                shape.push(dim);
                offset += 8;
            }
            let numel: usize = shape.iter().product();
            let expected_data_size = numel * 4;
            if bytes.len() < offset + expected_data_size {
                return v(ffi_ops::make_tensor(MetalTensor::zeros(
                    &[1],
                    crate::DType::F32,
                )));
            }
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                data.push(val);
                offset += 4;
            }
            v(ffi_ops::make_tensor(MetalTensor::from_slice(
                &data,
                &shape,
                crate::DType::F32,
            )))
        }
    }

    // ========== NN ==========
    #[inline]
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
        v(ffi_ops::tl_metal_conv2d(
            t(input),
            t(weight),
            t(bias),
            stride,
            padding,
            dilation,
            groups,
        ))
    }
    #[inline]
    fn tensor_batch_norm(
        &self,
        input: *mut c_void,
        running_mean: *mut c_void,
        running_var: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        training: bool,
        momentum: f64,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_batch_norm(
            t(input),
            t(running_mean),
            t(running_var),
            t(weight),
            t(bias),
            training,
            momentum,
            eps,
        ))
    }
    #[inline]
    fn tensor_layer_norm(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_layer_norm(
            t(input),
            t(weight),
            t(bias),
            eps,
        ))
    }
    #[inline]
    fn tensor_dropout(
        &self,
        input: *mut c_void,
        p: f64,
        training: bool,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_dropout(t(input), p, training))
    }
    #[inline]
    fn tensor_max_pool2d(
        &self,
        input: *mut c_void,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_max_pool2d(
            t(input),
            kernel_size,
            stride,
            padding,
        ))
    }
    #[inline]
    fn tensor_avg_pool2d(
        &self,
        input: *mut c_void,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> BackendResult<*mut c_void> {
        v(ffi_ops::tl_metal_avg_pool2d(
            t(input),
            kernel_size,
            stride,
            padding,
        ))
    }

    // ========== 次元特化メソッド (汎用 GPU メソッドに委譲) ==========
    #[inline]
    fn tensor_transpose_2d(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        self.tensor_transpose(a, 0, 1)
    }
    #[inline]
    fn tensor_reshape_2d(&self, a: *mut c_void, d0: i64, d1: i64) -> BackendResult<*mut c_void> {
        // shape テンソルを作成して reshape に委譲
        let shape_data = [d0 as f32, d1 as f32];
        let shape_tensor = v(ffi_ops::make_tensor(MetalTensor::from_slice(
            &shape_data,
            &[2],
            crate::DType::F32,
        )))?;
        self.tensor_reshape_new(a, shape_tensor)
    }
    #[inline]
    fn tensor_reshape_3d_to_2d(
        &self,
        a: *mut c_void,
        d0: i64,
        d1: i64,
    ) -> BackendResult<*mut c_void> {
        self.tensor_reshape_2d(a, d0, d1)
    }
    #[inline]
    fn tensor_matmul_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        self.tensor_matmul(a, b)
    }
    #[inline]
    fn tensor_add_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        self.tensor_add(a, b)
    }
    #[inline]
    fn tensor_silu_4d(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        self.tensor_silu(a)
    }
}
