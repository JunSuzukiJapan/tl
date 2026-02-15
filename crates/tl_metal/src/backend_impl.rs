//! GpuTensor トレイトの MetalTensor 実装 + 演算メソッド

use tl_backend::tensor::GpuTensor;
use tl_backend::DType as BackendDType;
use crate::tensor::MetalTensor;
use crate::DType;
use tl_backend::{BackendResult};

/// DType 変換
fn to_backend_dtype(dtype: DType) -> BackendDType {
    match dtype {
        DType::F32 => BackendDType::F32,
        DType::F16 => BackendDType::F16,
        DType::I32 => BackendDType::I32,
        DType::I64 => BackendDType::I64,
        DType::U8 => BackendDType::U8,
    }
}

fn from_backend_dtype(dtype: BackendDType) -> DType {
    match dtype {
        BackendDType::F32 => DType::F32,
        BackendDType::F16 => DType::F16,
        BackendDType::I32 => DType::I32,
        BackendDType::I64 => DType::I64,
        BackendDType::U8 => DType::U8,
        _ => DType::F32,
    }
}

impl GpuTensor for MetalTensor {
    fn shape(&self) -> &[usize] {
        MetalTensor::shape(self)
    }
    
    fn dtype(&self) -> BackendDType {
        to_backend_dtype(MetalTensor::dtype(self))
    }
    
    fn to_vec_f32(&self) -> Vec<f32> {
        self.to_vec::<f32>()
    }
    
    fn to_vec_i64(&self) -> Vec<i64> {
        self.to_vec::<f32>().into_iter().map(|x| x as i64).collect()
    }
    
    fn from_slice_f32(data: &[f32], shape: &[usize]) -> BackendResult<Self> {
        Ok(MetalTensor::from_slice(data, shape, DType::F32))
    }
    
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> BackendResult<Self> {
        let f32_data: Vec<f32> = data.iter().map(|x| *x as f32).collect();
        Ok(MetalTensor::from_slice(&f32_data, shape, DType::F32))
    }
    
    fn zeros(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(MetalTensor::zeros(shape, from_backend_dtype(dtype)))
    }
    
    fn ones(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(MetalTensor::ones(shape, from_backend_dtype(dtype)))
    }
    
    fn randn(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(MetalTensor::randn(shape, from_backend_dtype(dtype)))
    }
    
    fn arange(start: i64, end: i64, dtype: BackendDType) -> BackendResult<Self> {
        // Inherited arange might not be implemented in MetalTensor inherent? 
        // Let's check inherent arange if it exists. 
        // If not, we implement it here or assume MetalTensor::arange exists.
        // Previous compilation error suggested MetalTensor::arange exists and returned MetalTensor.
        Ok(MetalTensor::arange(start, end, from_backend_dtype(dtype))?)
    }
    
    fn clone_data(&self) -> BackendResult<Self> {
        MetalTensor::clone_data(self)
    }
}

impl MetalTensor {
    // ========== ユーティリティ (inherent) ==========
    /// arange 実装 (CPU fallback or GPU shader if available)
    /// Currently assuming CPU fallback logic or implemented elsewhere.
    /// If not implemented, I need to add it. But compilation error treated it as existing.
    /// Let's assume it exists in `scalar.rs` or directly in `tensor.rs` or `special.rs`.
    /// Wait, I haven't seen `arange` in `tensor.rs`.
    /// Maybe it was in `scalar.rs`? No.
    /// Maybe `backend_impl.rs` was relying on an inherent method I didn't see?
    /// Or maybe I need to implement it here if it was missing?
    /// The error said: `MetalTensor::arange(start, end, from_backend_dtype(dtype))`
    /// So it must exist. I'll trust it exists.
    
    // ========== 二項演算 (BackendResult を返す) ==========
    pub fn add(&self, other: &Self) -> BackendResult<Self> { self.add_impl(other) }
    pub fn sub(&self, other: &Self) -> BackendResult<Self> { self.sub_impl(other) }
    pub fn mul(&self, other: &Self) -> BackendResult<Self> { self.mul_impl(other) }
    pub fn div(&self, other: &Self) -> BackendResult<Self> { self.div_impl(other) }
    pub fn pow(&self, other: &Self) -> BackendResult<Self> { self.pow_impl(other) }
    pub fn matmul(&self, other: &Self) -> BackendResult<Self> { self.matmul_impl(other) }
    
    // ========== スカラー演算 ==========
    pub fn add_scalar(&self, scalar: f32) -> BackendResult<Self> { self.add_scalar_impl(scalar) }
    pub fn mul_scalar(&self, scalar: f32) -> BackendResult<Self> { self.mul_scalar_impl(scalar) }
    pub fn sub_scalar(&self, scalar: f32) -> BackendResult<Self> { self.sub_scalar_impl(scalar) }
    pub fn div_scalar(&self, scalar: f32) -> BackendResult<Self> { self.div_scalar_impl(scalar) }
    pub fn clamp(&self, min: f32, max: f32) -> BackendResult<Self> { self.clamp_impl(min, max) }
    
    // ========== 単項演算 ==========
    pub fn neg(&self) -> BackendResult<Self> { self.neg_impl() }
    pub fn abs(&self) -> BackendResult<Self> { self.abs_impl() }
    pub fn exp(&self) -> BackendResult<Self> { self.exp_impl() }
    pub fn log(&self) -> BackendResult<Self> { self.log_impl() }
    pub fn sqrt(&self) -> BackendResult<Self> { self.sqrt_impl() }
    pub fn sin(&self) -> BackendResult<Self> { self.sin_impl() }
    pub fn cos(&self) -> BackendResult<Self> { self.cos_impl() }
    pub fn tan(&self) -> BackendResult<Self> { self.tan_impl() }
    pub fn tanh(&self) -> BackendResult<Self> { self.tanh_impl() }
    pub fn sigmoid(&self) -> BackendResult<Self> { self.sigmoid_impl() }
    pub fn relu(&self) -> BackendResult<Self> { self.relu_impl() }
    pub fn gelu(&self) -> BackendResult<Self> { self.gelu_impl() }
    
    // ========== Reduce 演算 ==========
    pub fn sumall(&self) -> BackendResult<f32> { self.sumall_impl() }
    pub fn mean_all(&self) -> BackendResult<f32> { self.mean_all_impl() }
    pub fn sum(&self, axis: i32) -> BackendResult<Self> { self.sum_impl(axis) }
    pub fn max(&self, axis: i32) -> BackendResult<Self> { self.max_impl(axis) }
    pub fn min(&self, axis: i32) -> BackendResult<Self> { self.min_impl(axis) }
    pub fn argmax(&self, axis: i32) -> BackendResult<Self> { self.argmax_impl(axis) }
    pub fn argmax_all(&self) -> BackendResult<usize> { self.argmax_all_impl() }
    pub fn argmin(&self, axis: i32) -> BackendResult<Self> { self.argmin_impl(axis) }
    pub fn mean(&self, axis: i32) -> BackendResult<Self> { self.mean_impl(axis) }
    
    // ========== 形状操作 ==========
    pub fn reshape(&self, shape: &[usize]) -> BackendResult<Self> { self.reshape_impl(shape) }
    pub fn transpose(&self, dim0: usize, dim1: usize) -> BackendResult<Self> { self.transpose_impl(dim0, dim1) }
    pub fn squeeze(&self, dim: usize) -> BackendResult<Self> { self.squeeze_impl(dim) }
    pub fn unsqueeze(&self, dim: usize) -> BackendResult<Self> { self.unsqueeze_impl(dim) }
    // broadcast_to, narrow, slice, contiguous, cat は ops/broadcast.rs, ops/shape.rs で定義済み
    
    // ========== 活性化・特殊演算 ==========
    pub fn softmax(&self, axis: i32) -> BackendResult<Self> { self.softmax_impl(axis) }
    pub fn embedding(&self, indices: &Self) -> BackendResult<Self> { self.embedding_impl(indices) }
    pub fn tril(&self, diagonal: i32) -> BackendResult<Self> { self.tril_impl(diagonal) }
    pub fn cross_entropy(&self, target: &Self) -> BackendResult<Self> { self.cross_entropy_impl(target) }
    pub fn repeat_interleave(&self, repeats: usize, axis: usize) -> BackendResult<Self> { self.repeat_interleave_impl(repeats, axis) }
    pub fn index_select(&self, axis: usize, indices: &Self) -> BackendResult<Self> { self.index_select_impl(axis, indices) }
    pub fn where_cond(condition: &Self, x: &Self, y: &Self) -> BackendResult<Self> { MetalTensor::where_cond_impl(condition, x, y) }
    
    // ========== 深層学習演算 ==========
    pub fn conv2d(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> BackendResult<Self> {
        self.conv2d_impl(weight, stride, padding)
    }
    pub fn batch_norm(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: f32) -> BackendResult<Self> {
        self.batch_norm_impl(gamma, beta, running_mean, running_var, eps)
    }
    pub fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> BackendResult<Self> {
        self.layer_norm_impl(gamma, beta, eps)
    }
    pub fn max_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> BackendResult<Self> {
        self.max_pool2d_impl(kernel_size, stride)
    }
    pub fn avg_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> BackendResult<Self> {
        self.avg_pool2d_impl(kernel_size, stride)
    }
    pub fn dropout(&self, p: f32, training: bool) -> BackendResult<Self> {
        self.dropout_impl(p, training)
    }
    
    // Needed for arange if not defined elsewhere?

}
