//! CpuDevice: IDevice トレイトの CPU 実装
//!
//! 既存の `ffi::tl_cpu_tensor_*` 関数に委譲する。
//! 全メソッドは void* と CpuTensor のキャストで橋渡しする。

use crate::ffi;
use crate::tensor::CpuTensor;
use std::ffi::c_void;
use tl_backend::{BackendError, BackendResult, IDevice};

/// CPU デバイス (ゼロサイズ型)
pub struct CpuDevice;

/// void* → *mut CpuTensor キャスト
#[inline(always)]
fn t(p: *mut c_void) -> *mut CpuTensor {
    p as *mut CpuTensor
}

/// *mut CpuTensor → void* キャスト
#[inline(always)]
fn v(p: *mut CpuTensor) -> *mut c_void {
    p as *mut c_void
}

/// 結果ポインタの null チェックを行うヘルパー
#[inline(always)]
fn check(ptr: *mut crate::ffi::OpaqueTensor) -> BackendResult<*mut c_void> {
    if ptr.is_null() {
        Err(BackendError::InternalError(
            "CPU operation failed (check stderr)".to_string(),
        ))
    } else {
        Ok(v(ptr))
    }
}

impl IDevice for CpuDevice {
    // ========== テンソル作成 ==========
    #[inline]
    fn tensor_new(
        &self,
        data: *const f32,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_new(data, rank, shape))
    }
    #[inline]
    fn tensor_new_i64(
        &self,
        data: *const i64,
        rank: usize,
        shape: *const usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_new_i64(data, rank, shape))
    }
    #[inline]
    fn tensor_from_i64_array(&self, data: *const i64, len: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_from_i64(data, len))
    }
    #[inline]
    fn tensor_zeros(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_zeros(rank, shape, req_grad))
    }
    #[inline]
    fn tensor_ones(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_ones(rank, shape, req_grad))
    }
    #[inline]
    fn tensor_randn_debug(
        &self,
        rank: usize,
        shape: *const usize,
        seed: u64,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_randn_debug(rank, shape, seed, req_grad))
    }
    #[inline]
    fn tensor_new_causal_mask(&self, size: usize) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_new_causal_mask(size))
    }
    #[inline]
    fn tensor_from_vec_u8(&self, data: *mut c_void, len: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_from_vec_u8(data, len))
    }
    #[inline]
    fn tensor_from_u8_labels(&self, data: *const u8, len: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_from_u8_labels(data, len))
    }
    #[inline]
    fn tensor_full(
        &self,
        rank: usize,
        shape: *const usize,
        value: f32,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_full(rank, shape, value, req_grad))
    }
    #[inline]
    fn tensor_eye(&self, n: usize, req_grad: bool) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_eye(n, req_grad))
    }
    #[inline]
    fn tensor_arange(&self, start: f64, end: f64, step: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_arange(start, end, step))
    }
    #[inline]
    fn tensor_linspace(&self, start: f64, end: f64, steps: usize) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_linspace(start, end, steps))
    }
    #[inline]
    fn tensor_rand(
        &self,
        rank: usize,
        shape: *const usize,
        req_grad: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_rand(rank as i64, shape, req_grad))
    }

    // ========== 要素操作 ==========
    fn tensor_where_cond(
        &self,
        cond: *mut c_void,
        x: *mut c_void,
        y: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tc, tx, ty) = unsafe { (&*t(cond), &*t(x), &*t(y)) };
        let result = CpuTensor::where_cond(tc, tx, ty)?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_masked_fill(
        &self,
        tensor: *mut c_void,
        mask: *mut c_void,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        use crate::tensor::CpuTensor as CT;
        let (tt, tm) = unsafe { (&*t(tensor), &*t(mask)) };
        let numel: usize = tt.shape().iter().product();
        let value_tensor = CT::from_slice(&vec![value; numel], tt.shape(), crate::DType::F32);
        let result = CT::where_cond(tm, &value_tensor, tt)?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_std(&self, tensor: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let var_ptr = self.tensor_var(tensor, dim)?;
        let var_t = unsafe { &*(var_ptr as *mut crate::tensor::CpuTensor) };
        let result = var_t.sqrt_impl()?;
        unsafe {
            let _ = std::sync::Arc::from_raw(
                var_ptr as *const std::cell::UnsafeCell<crate::tensor::CpuTensor>,
            );
        }
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_prod(&self, tensor: *mut c_void, dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        // prod = exp(sum(log(abs(x)))) with sign tracking
        // Simpler: use log-sum-exp approach: prod = exp(sum(log(x)))
        let log_t = tt.log_impl()?;
        let sum_log = log_t.sum_impl(dim)?;
        let result = sum_log.exp_impl()?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(&[norm_val], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_topk(&self, tensor: *mut c_void, k: usize, _dim: i32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let mut data = tt.to_vec::<f32>();
        data.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        data.truncate(k);
        let out = crate::tensor::CpuTensor::from_slice(&data, &[k], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_logical_and(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let da = ta.to_vec::<f32>();
        let db = tb.to_vec::<f32>();
        let result: Vec<f32> = da
            .iter()
            .zip(db.iter())
            .map(|(&x, &y)| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 })
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, ta.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_logical_or(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let da = ta.to_vec::<f32>();
        let db = tb.to_vec::<f32>();
        let result: Vec<f32> = da
            .iter()
            .zip(db.iter())
            .map(|(&x, &y)| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 })
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, ta.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_logical_not(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x == 0.0 { 1.0 } else { 0.0 })
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_leaky_relu(
        &self,
        tensor: *mut c_void,
        negative_slope: f32,
    ) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_elu(&self, tensor: *mut c_void, alpha: f32) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_mish(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x * (x.exp().ln_1p()).tanh()).collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_l1_loss(&self, pred: *mut c_void, target: *mut c_void) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let diff = tp.sub_impl(tt)?;
        let abs_diff = diff.abs_impl()?;
        let result = abs_diff.mean_impl(-1)?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(&[loss], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(&[loss], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
            let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
            Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
        } else {
            let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
            Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
        }
    }

    fn tensor_hardswish(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let r: Vec<f32> = data
            .iter()
            .map(|&x| x * (x + 3.0).max(0.0).min(6.0) / 6.0)
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&r, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_hardsigmoid(&self, tensor: *mut c_void) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let data = tt.to_vec::<f32>();
        let r: Vec<f32> = data
            .iter()
            .map(|&x| ((x + 3.0) / 6.0).max(0.0).min(1.0))
            .collect();
        let out = crate::tensor::CpuTensor::from_slice(&r, tt.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
                let start_c = g * group_size;
                let end_c = start_c + group_size;
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
                            let tw = unsafe { &*t(weight) };
                            v *= tw.to_vec::<f32>()[c];
                        }
                        if !bias.is_null() {
                            let tb = unsafe { &*t(bias) };
                            v += tb.to_vec::<f32>()[c];
                        }
                        result[idx] = v;
                    }
                }
            }
        }
        let out = crate::tensor::CpuTensor::from_slice(&result, &shape, crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let oh = output_h as usize;
        let ow = output_w as usize;
        let mut result = vec![0.0f32; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let h_start = oi * h / oh;
                        let h_end = (oi + 1) * h / oh;
                        let w_start = oj * w / ow;
                        let w_end = (oj + 1) * w / ow;
                        let mut sum = 0.0f32;
                        let mut cnt = 0;
                        for hi in h_start..h_end {
                            for wi in w_start..w_end {
                                sum += data[bi * c * h * w + ci * h * w + hi * w + wi];
                                cnt += 1;
                            }
                        }
                        result[bi * c * oh * ow + ci * oh * ow + oi * ow + oj] = sum / cnt as f32;
                    }
                }
            }
        }
        let out = crate::tensor::CpuTensor::from_slice(&result, &[n, c, oh, ow], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let last_dim = *shape.last().unwrap();
        let new_last = last_dim + pad_left as usize + pad_right as usize;
        let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let mut result = Vec::with_capacity(outer * new_last);
        for o in 0..outer {
            for _ in 0..pad_left {
                result.push(value);
            }
            for i in 0..last_dim {
                result.push(data[o * last_dim + i]);
            }
            for _ in 0..pad_right {
                result.push(value);
            }
        }
        let mut new_shape = shape.clone();
        *new_shape.last_mut().unwrap() = new_last;
        let out = crate::tensor::CpuTensor::from_slice(&result, &new_shape, crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_instance_norm(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
        eps: f64,
    ) -> BackendResult<*mut c_void> {
        // Instance norm = group norm with num_groups == num_channels
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
        p: f64,
        training: bool,
    ) -> BackendResult<*mut c_void> {
        let ti = unsafe { &*t(input) };
        if !training {
            let out = crate::tensor::CpuTensor::from_slice(
                &ti.to_vec::<f32>(),
                ti.shape(),
                crate::DType::F32,
            );
            let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
            return Ok(std::sync::Arc::into_raw(arc) as *mut c_void);
        }
        let data = ti.to_vec::<f32>();
        let shape = ti.shape().to_vec();
        let channels = if shape.len() >= 2 { shape[1] } else { 1 };
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);
        let batch = if shape.len() >= 2 { shape[0] } else { 1 };
        let mut result = data.clone();
        let scale = 1.0 / (1.0 - p) as f32;
        for b in 0..batch {
            for c in 0..channels {
                let drop = (c as f64 * 0.618 + b as f64) % 1.0 < p;
                for s in 0..spatial {
                    let idx = b * channels * spatial + c * spatial + s;
                    result[idx] = if drop { 0.0 } else { result[idx] * scale };
                }
            }
        }
        let out = crate::tensor::CpuTensor::from_slice(&result, &shape, crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let st = stride as usize;
        let pad = padding as usize;
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
        let out = crate::tensor::CpuTensor::from_slice(
            &result,
            &[batch, out_ch, out_len],
            crate::DType::F32,
        );
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_kl_div_loss(
        &self,
        pred: *mut c_void,
        target: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tp, tt) = unsafe { (&*t(pred), &*t(target)) };
        let p = tp.to_vec::<f32>();
        let q = tt.to_vec::<f32>();
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
        let loss = sum / q.len() as f32;
        let out = crate::tensor::CpuTensor::from_slice(&[loss], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(
            &result,
            &[batch, out_ch, oh, ow],
            crate::DType::F32,
        );
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
                            // nearest
                            let si = (oi * h) / oh;
                            let sj = (oj * w) / ow;
                            data[bi * c * h * w + ci * h * w + si * w + sj]
                        } else {
                            // bilinear
                            let sy = oi as f32 * (h as f32 - 1.0) / (oh as f32 - 1.0).max(1.0);
                            let sx = oj as f32 * (w as f32 - 1.0) / (ow as f32 - 1.0).max(1.0);
                            let y0 = sy.floor() as usize;
                            let x0 = sx.floor() as usize;
                            let y1 = (y0 + 1).min(h - 1);
                            let x1 = (x0 + 1).min(w - 1);
                            let fy = sy - y0 as f32;
                            let fx = sx - x0 as f32;
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
        let out = crate::tensor::CpuTensor::from_slice(&result, &[n, c, oh, ow], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_scaled_dot_product_attention(
        &self,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        mask: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        let (tq, tk, tv) = unsafe { (&*t(q), &*t(k), &*t(v)) };
        // Q: [batch, heads, seq_q, d_k], K: [batch, heads, seq_k, d_k], V: [batch, heads, seq_k, d_v]
        let qd = tq.to_vec::<f32>();
        let kd = tk.to_vec::<f32>();
        let vd = tv.to_vec::<f32>();
        let qs = tq.shape().to_vec();
        let ks = tk.shape().to_vec();
        let vs = tv.shape().to_vec();
        let (batch, heads) = if qs.len() == 4 {
            (qs[0], qs[1])
        } else if qs.len() == 3 {
            (1, qs[0])
        } else {
            (1, 1)
        };
        let (seq_q, d_k) = (qs[qs.len() - 2], qs[qs.len() - 1]);
        let seq_k = ks[ks.len() - 2];
        let d_v = vs[vs.len() - 1];
        let scale = 1.0 / (d_k as f32).sqrt();
        let has_mask = !mask.is_null();
        let mask_data = if has_mask {
            unsafe { &*t(mask) }.to_vec::<f32>()
        } else {
            vec![]
        };

        let mut result = vec![0.0f32; batch * heads * seq_q * d_v];
        for b in 0..batch {
            for h in 0..heads {
                // scores = Q @ K^T * scale
                let q_off = b * heads * seq_q * d_k + h * seq_q * d_k;
                let k_off = b * heads * seq_k * d_k + h * seq_k * d_k;
                let v_off = b * heads * seq_k * d_v + h * seq_k * d_v;
                let mut scores = vec![0.0f32; seq_q * seq_k];
                for i in 0..seq_q {
                    for j in 0..seq_k {
                        let mut s = 0.0f32;
                        for d in 0..d_k {
                            s += qd[q_off + i * d_k + d] * kd[k_off + j * d_k + d];
                        }
                        scores[i * seq_k + j] = s * scale;
                    }
                }
                // apply mask
                if has_mask {
                    for i in 0..seq_q {
                        for j in 0..seq_k {
                            if mask_data.len() > i * seq_k + j && mask_data[i * seq_k + j] == 0.0 {
                                scores[i * seq_k + j] = -1e9;
                            }
                        }
                    }
                }
                // softmax per row
                for i in 0..seq_q {
                    let max_v: f32 = scores[i * seq_k..(i + 1) * seq_k]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..seq_k {
                        scores[i * seq_k + j] = (scores[i * seq_k + j] - max_v).exp();
                        sum += scores[i * seq_k + j];
                    }
                    for j in 0..seq_k {
                        scores[i * seq_k + j] /= sum;
                    }
                }
                // attn @ V
                let r_off = b * heads * seq_q * d_v + h * seq_q * d_v;
                for i in 0..seq_q {
                    for j in 0..d_v {
                        let mut s = 0.0f32;
                        for l in 0..seq_k {
                            s += scores[i * seq_k + l] * vd[v_off + l * d_v + j];
                        }
                        result[r_off + i * d_v + j] = s;
                    }
                }
            }
        }
        let out_shape = if qs.len() == 4 {
            vec![batch, heads, seq_q, d_v]
        } else if qs.len() == 3 {
            vec![heads, seq_q, d_v]
        } else {
            vec![seq_q, d_v]
        };
        let out = crate::tensor::CpuTensor::from_slice(&result, &out_shape, crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_top_k_sample(&self, logits: *mut c_void, k: i64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        let k = k as usize;
        // find k-th largest value
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = if k < sorted.len() {
            sorted[k - 1]
        } else {
            sorted[sorted.len() - 1]
        };
        // mask and softmax
        let masked: Vec<f32> = data
            .iter()
            .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
            .collect();
        let max_v: f32 = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = masked.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
        // argmax
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let out = crate::tensor::CpuTensor::from_slice(&[idx as f32], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_top_p_sample(&self, logits: *mut c_void, p: f64) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        // softmax
        let max_v: f32 = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
        // sort indices by probability descending
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // cumulative sum, mask below p
        let mut cum = 0.0f32;
        let mut mask = vec![false; probs.len()];
        for &i in &indices {
            cum += probs[i];
            mask[i] = true;
            if cum >= p as f32 {
                break;
            }
        }
        // mask and re-normalize
        let masked: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &v)| if mask[i] { v } else { 0.0 })
            .collect();
        let sum2: f32 = masked.iter().sum();
        let final_probs: Vec<f32> = masked.iter().map(|v| v / sum2).collect();
        // argmax
        let idx = final_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let out = crate::tensor::CpuTensor::from_slice(&[idx as f32], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_temperature_scale(
        &self,
        logits: *mut c_void,
        temperature: f64,
    ) -> BackendResult<*mut c_void> {
        let tl = unsafe { &*t(logits) };
        let data = tl.to_vec::<f32>();
        let t = temperature as f32;
        let result: Vec<f32> = data.iter().map(|&v| v / t).collect();
        let out = crate::tensor::CpuTensor::from_slice(&result, tl.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
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
        let out = crate::tensor::CpuTensor::from_slice(&data, tl.shape(), crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_dot(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let da = ta.to_vec::<f32>();
        let db = tb.to_vec::<f32>();
        let dot: f32 = da.iter().zip(db.iter()).map(|(&x, &y)| x * y).sum();
        let out = crate::tensor::CpuTensor::from_slice(&[dot], &[1], crate::DType::F32);
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(out));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_fill_(&self, tensor: *mut c_void, value: f32) -> BackendResult<()> {
        let tt = unsafe { &*t(tensor) };
        let numel: usize = tt.shape().iter().product();
        let filled = crate::tensor::CpuTensor::from_slice(
            &vec![value; numel],
            tt.shape(),
            crate::DType::F32,
        );
        unsafe {
            let dst = &mut *(tensor as *mut crate::tensor::CpuTensor);
            *dst = filled;
        }
        Ok(())
    }

    fn tensor_set_requires_grad(&self, tensor: *mut c_void, req_grad: bool) -> BackendResult<()> {
        let tt = unsafe { &mut *t(tensor) };
        tt.set_requires_grad(req_grad);
        Ok(())
    }

    fn tensor_clip_grad_value(&self, tensor: *mut c_void, min: f64, max: f64) -> BackendResult<()> {
        let tt = unsafe { &mut *t(tensor) };
        tt.clip_grad_value(min as f32, max as f32)?;
        Ok(())
    }

    fn tensor_clip_grad_norm(
        &self,
        tensor: *mut c_void,
        max_norm: f64,
        norm_type: f64,
    ) -> BackendResult<f64> {
        let tt = unsafe { &mut *t(tensor) };
        let norm = tt.clip_grad_norm(max_norm as f32, norm_type as f32)?;
        Ok(norm as f64)
    }

    fn tensor_broadcast_to(
        &self,
        tensor: *mut c_void,
        shape: &[usize],
    ) -> BackendResult<*mut c_void> {
        let tt = unsafe { &*t(tensor) };
        let result = tt.broadcast_to(shape)?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    fn tensor_stack(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        use crate::tensor::CpuTensor as CT;
        let (ta, tb) = unsafe { (&*t(a), &*t(b)) };
        let ua = ta.unsqueeze(dim as usize)?;
        let ub = tb.unsqueeze(dim as usize)?;
        let result = CT::cat(&[&ua, &ub], dim as usize)?;
        let arc = std::sync::Arc::new(std::cell::UnsafeCell::new(result));
        Ok(std::sync::Arc::into_raw(arc) as *mut c_void)
    }

    // ========== メモリ管理 ==========
    #[inline]
    fn tensor_clone(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_clone(t(a)))
    }
    #[inline]
    fn tensor_shallow_clone(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_shallow_clone(t(a)))
    }
    #[inline]
    fn tensor_free(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_free(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_release(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_release(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_acquire(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_acquire(t(a)))
    }
    #[inline]
    fn tensor_release_safe(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_release(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_promote(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_promote(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_register(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_register(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_prepare_return(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_prepare_return(t(a)))
    }

    // ========== テンソル情報 ==========
    #[inline]
    fn tensor_len(&self, a: *mut c_void) -> BackendResult<usize> {
        Ok(ffi::tl_cpu_tensor_len(t(a)))
    }
    #[inline]
    fn tensor_dim(&self, a: *mut c_void, dim: usize) -> BackendResult<usize> {
        Ok(ffi::tl_cpu_tensor_dim(t(a), dim))
    }
    #[inline]
    fn tensor_numel(&self, a: *mut c_void) -> BackendResult<i64> {
        Ok(ffi::tl_cpu_tensor_numel(t(a)))
    }
    #[inline]
    fn tensor_data(&self, a: *mut c_void) -> BackendResult<*const f32> {
        Ok(ffi::tl_cpu_tensor_data(t(a)))
    }
    #[inline]
    fn tensor_device_id(&self, a: *mut c_void) -> BackendResult<i32> {
        Ok(ffi::tl_cpu_tensor_device_id(t(a)))
    }
    #[inline]
    fn tensor_get_shape(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_get_shape(t(a)))
    }

    // ========== 要素アクセス ==========
    #[inline]
    fn tensor_get(&self, a: *mut c_void, idx: i64) -> BackendResult<f32> {
        Ok(ffi::tl_cpu_tensor_get(t(a), idx))
    }
    #[inline]
    fn tensor_get_f32_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<f32> {
        Ok(ffi::tl_cpu_tensor_get_f32_md(t(a), indices, rank))
    }
    #[inline]
    fn tensor_get_i64_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: i64,
    ) -> BackendResult<i64> {
        Ok(ffi::tl_cpu_tensor_get_i64_md(t(a), indices, rank))
    }
    #[inline]
    fn tensor_set_f32_md(
        &self,
        a: *mut c_void,
        indices: *const i64,
        rank: usize,
        value: f32,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_set_f32_md(t(a), indices, rank, value))
    }
    #[inline]
    fn tensor_item(&self, a: *mut c_void) -> BackendResult<f32> {
        Ok(ffi::tl_cpu_tensor_item(t(a)))
    }
    #[inline]
    fn tensor_item_i64(&self, a: *mut c_void) -> BackendResult<i64> {
        Ok(ffi::tl_cpu_tensor_item_i64(t(a)))
    }

    // ========== 二項演算 ==========
    #[inline]
    fn tensor_add(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_add(t(a), t(b)))
    }
    #[inline]
    fn tensor_sub(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sub(t(a), t(b)))
    }
    #[inline]
    fn tensor_mul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_mul(t(a), t(b)))
    }
    #[inline]
    fn tensor_div(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_div(t(a), t(b)))
    }
    #[inline]
    fn tensor_rem(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_rem(t(a), t(b)))
    }
    #[inline]
    fn tensor_matmul(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_matmul(t(a), t(b)))
    }
    #[inline]
    fn tensor_pow(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_pow(t(a), t(b)))
    }
    #[inline]
    fn tensor_cross_entropy(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cross_entropy(t(a), t(b)))
    }

    // ========== 単項演算 ==========
    #[inline]
    fn tensor_neg(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_neg(t(a)))
    }
    #[inline]
    fn tensor_abs(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_abs(t(a)))
    }
    #[inline]
    fn tensor_contiguous(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_contiguous(t(a)))
    }

    // ========== 比較演算 ==========
    #[inline]
    fn tensor_eq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_eq(t(a), t(b)))
    }
    #[inline]
    fn tensor_neq(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_neq(t(a), t(b)))
    }
    #[inline]
    fn tensor_gt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_gt(t(a), t(b)))
    }
    #[inline]
    fn tensor_lt(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_lt(t(a), t(b)))
    }
    #[inline]
    fn tensor_ge(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_ge(t(a), t(b)))
    }
    #[inline]
    fn tensor_le(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_le(t(a), t(b)))
    }

    // ========== スカラー演算 ==========
    #[inline]
    fn tensor_add_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_add_scalar(t(a), s))
    }
    #[inline]
    fn tensor_sub_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sub_scalar(t(a), s))
    }
    #[inline]
    fn tensor_mul_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_mul_scalar(t(a), s))
    }
    #[inline]
    fn tensor_div_scalar(&self, a: *mut c_void, s: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_div_scalar(t(a), s))
    }
    #[inline]
    fn tensor_pow_scalar(&self, a: *mut c_void, exp: f32) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_pow_scalar(t(a), exp))
    }
    #[inline]
    fn tensor_scale(&self, a: *mut c_void, s: f32) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_scale(t(a), s as f64))
    }

    // ========== インプレース演算 ==========
    #[inline]
    fn tensor_add_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_add_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_sub_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_sub_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_mul_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_mul_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_div_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_div_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_mod_assign(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_mod_assign(t(a), t(b));
        Ok(())
    }
    #[inline]
    fn tensor_add_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi::tl_cpu_tensor_add_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_sub_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi::tl_cpu_tensor_sub_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_mul_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi::tl_cpu_tensor_mul_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_div_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi::tl_cpu_tensor_div_assign_scalar_f32(t(a), s);
        Ok(())
    }
    #[inline]
    fn tensor_mod_assign_scalar_f32(&self, a: *mut c_void, s: f32) -> BackendResult<()> {
        ffi::tl_cpu_tensor_mod_assign_scalar_f32(t(a), s);
        Ok(())
    }

    // ========== 数学・活性化関数 ==========
    #[inline]
    fn tensor_exp(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_exp(t(a)))
    }
    #[inline]
    fn tensor_log(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_log(t(a)))
    }
    #[inline]
    fn tensor_sqrt(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sqrt(t(a)))
    }
    #[inline]
    fn tensor_sin(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sin(t(a)))
    }
    #[inline]
    fn tensor_cos(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cos(t(a)))
    }
    #[inline]
    fn tensor_tan(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_tan(t(a)))
    }
    #[inline]
    fn tensor_tanh(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_tanh(t(a)))
    }
    #[inline]
    fn tensor_sigmoid(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sigmoid(t(a)))
    }
    #[inline]
    fn tensor_relu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_relu(t(a)))
    }
    #[inline]
    fn tensor_gelu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_gelu(t(a)))
    }
    #[inline]
    fn tensor_silu(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_silu(t(a)))
    }

    // ========== Reduction ==========
    #[inline]
    fn tensor_sum(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sum(t(a)))
    }
    #[inline]
    fn tensor_mean(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_mean(t(a)))
    }
    #[inline]
    fn tensor_max(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_max(t(a)))
    }
    #[inline]
    fn tensor_min(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_min(t(a)))
    }
    #[inline]
    fn tensor_softmax(&self, a: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_softmax(t(a), dim))
    }
    #[inline]
    fn tensor_max_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_max_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_min_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_min_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_mean_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_mean_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_sum_dim(
        &self,
        a: *mut c_void,
        dim: usize,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sum_dim(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_argmax(
        &self,
        a: *mut c_void,
        dim: i64,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_argmax(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_argmin(
        &self,
        a: *mut c_void,
        dim: i64,
        keep_dim: bool,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_argmin(t(a), dim, keep_dim))
    }
    #[inline]
    fn tensor_tril(&self, a: *mut c_void, diagonal: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_tril(t(a), diagonal))
    }
    #[inline]
    fn tensor_clamp(&self, a: *mut c_void, min: f64, max: f64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_clamp(t(a), min, max))
    }
    #[inline]
    fn tensor_sample(&self, a: *mut c_void, temp: f32, top_p: f32) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_sample(t(a), temp, top_p))
    }

    // ========== Autograd ==========
    #[inline]
    fn tensor_backward(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_backward(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_grad(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_grad(t(a)))
    }
    #[inline]
    fn tensor_detach(&self, a: *mut c_void, req_grad: bool) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_detach(t(a), req_grad))
    }
    #[inline]
    fn tensor_enable_grad(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_enable_grad(t(a));
        Ok(())
    }
    #[inline]
    fn clear_grads(&self) -> BackendResult<()> {
        ffi::tl_cpu_clear_grads();
        Ok(())
    }

    // ========== 形状操作 ==========
    #[inline]
    fn tensor_reshape_new(&self, a: *mut c_void, s: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_reshape_new(t(a), t(s)))
    }
    #[inline]
    fn tensor_reshape_dims(
        &self,
        a: *mut c_void,
        dims_ptr: *const i64,
        rank: i64,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_reshape_dims(t(a), dims_ptr, rank))
    }
    #[inline]
    fn tensor_transpose(
        &self,
        a: *mut c_void,
        dim0: usize,
        dim1: usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_transpose(t(a), dim0, dim1))
    }
    #[inline]
    fn tensor_slice(
        &self,
        a: *mut c_void,
        dim: i64,
        start: i64,
        len: i64,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_slice(t(a), dim, start, len))
    }
    #[inline]
    fn tensor_narrow(
        &self,
        a: *mut c_void,
        dim: usize,
        start: usize,
        len: usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_narrow(t(a), dim, start, len))
    }
    #[inline]
    fn tensor_cat(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cat(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat_i64(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cat_i64(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat2(&self, a: *mut c_void, b: *mut c_void, dim: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cat2(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_cat_4d(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        dim: i64,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_cat_4d(t(a), t(b), dim))
    }
    #[inline]
    fn tensor_replace_data(&self, dst: *mut c_void, src: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_replace_data(t(dst), t(src));
        Ok(())
    }
    #[inline]
    fn tensor_repeat_interleave(
        &self,
        a: *mut c_void,
        repeats: usize,
        dim: usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_repeat_interleave(t(a), repeats, dim))
    }
    #[inline]
    fn tensor_to_device(&self, a: *mut c_void, device_id: i32) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_to_device(t(a), device_id))
    }
    #[inline]
    fn tensor_to_f32(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_to_f32(t(a)))
    }
    #[inline]
    fn tensor_to_i64(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_to_i64(t(a)))
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
        check(ffi::tl_cpu_tensor_embedding(t(w), t(idx), pad, sg, sp))
    }

    // ========== LLM ==========
    #[inline]
    fn tensor_rms_norm(
        &self,
        a: *mut c_void,
        w: *mut c_void,
        eps: f32,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_rms_norm(t(a), t(w), eps))
    }
    #[inline]
    fn tensor_rope_new_cos(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_rope_new_cos(seq_len, dim, base))
    }
    #[inline]
    fn tensor_rope_new_sin(
        &self,
        dim: usize,
        seq_len: usize,
        base: f32,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_rope_new_sin(seq_len, dim, base))
    }
    #[inline]
    fn tensor_apply_rope(
        &self,
        a: *mut c_void,
        cos: *mut c_void,
        sin: *mut c_void,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_apply_rope(t(a), t(cos), t(sin)))
    }

    // ========== IO / Print ==========
    #[inline]
    fn tensor_print(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_print(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_display(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_print(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_print_1(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_print_1(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_print_2(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_print_2(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_print_3(&self, a: *mut c_void) -> BackendResult<()> {
        ffi::tl_cpu_tensor_print_3(t(a));
        Ok(())
    }
    #[inline]
    fn tensor_save(&self, a: *mut c_void, path: *const i8) -> BackendResult<()> {
        ffi::tl_cpu_tensor_save(t(a), path);
        Ok(())
    }
    #[inline]
    fn tensor_load(&self, path: *const i8) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_load(path))
    }

    // ========== NN ==========
    #[inline]
    fn tensor_conv2d(
        &self,
        input: *mut c_void,
        weight: *mut c_void,
        _bias: *mut c_void,
        stride: usize,
        padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_conv2d(
            t(input),
            t(weight),
            padding as i64,
            stride as i64,
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
        check(ffi::tl_cpu_tensor_batch_norm(
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
        check(ffi::tl_cpu_tensor_layer_norm(
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
        check(ffi::tl_cpu_tensor_dropout(t(input), p, training))
    }
    #[inline]
    fn tensor_max_pool2d(
        &self,
        input: *mut c_void,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_max_pool2d(
            t(input),
            kernel_size as i64,
            stride as i64,
            padding as i64,
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
        check(ffi::tl_cpu_tensor_avg_pool2d(
            t(input),
            kernel_size as i64,
            stride as i64,
            padding as i64,
        ))
    }

    // ========== CPU 専用 ==========
    #[inline]
    fn tensor_transpose_2d(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_transpose_2d(t(a)))
    }
    #[inline]
    fn tensor_reshape_2d(&self, a: *mut c_void, d0: i64, d1: i64) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_reshape_2d(t(a), d0, d1))
    }
    #[inline]
    fn tensor_reshape_3d_to_2d(
        &self,
        a: *mut c_void,
        d0: i64,
        d1: i64,
    ) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_reshape_3d_to_2d(t(a), d0, d1))
    }
    #[inline]
    fn tensor_matmul_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_matmul_4d(t(a), t(b)))
    }
    #[inline]
    fn tensor_add_4d(&self, a: *mut c_void, b: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_add_4d(t(a), t(b)))
    }
    #[inline]
    fn tensor_silu_4d(&self, a: *mut c_void) -> BackendResult<*mut c_void> {
        check(ffi::tl_cpu_tensor_silu_4d(t(a)))
    }
}
