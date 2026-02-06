//! GpuTensor / GpuOps トレイトの MetalTensor 実装

use tl_backend::tensor::GpuTensor;
use tl_backend::ops::GpuOps;
use tl_backend::DType as BackendDType;
use crate::tensor::MetalTensor;
use crate::DType;

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
        self.shape()
    }
    
    fn dtype(&self) -> BackendDType {
        to_backend_dtype(self.dtype())
    }
    
    fn to_vec_f32(&self) -> Vec<f32> {
        self.to_vec::<f32>()
    }
    
    fn to_vec_i64(&self) -> Vec<i64> {
        self.to_vec::<f32>().into_iter().map(|x| x as i64).collect()
    }
    
    fn from_slice_f32(data: &[f32], shape: &[usize]) -> Self {
        MetalTensor::from_slice(data, shape, DType::F32)
    }
    
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> Self {
        let f32_data: Vec<f32> = data.iter().map(|x| *x as f32).collect();
        MetalTensor::from_slice(&f32_data, shape, DType::F32)
    }
    
    fn zeros(shape: &[usize], dtype: BackendDType) -> Self {
        MetalTensor::zeros(shape, from_backend_dtype(dtype))
    }
    
    fn ones(shape: &[usize], dtype: BackendDType) -> Self {
        MetalTensor::ones(shape, from_backend_dtype(dtype))
    }
    
    fn randn(shape: &[usize], dtype: BackendDType) -> Self {
        MetalTensor::randn(shape, from_backend_dtype(dtype))
    }
    
    fn arange(start: i64, end: i64, dtype: BackendDType) -> Self {
        MetalTensor::arange(start, end, from_backend_dtype(dtype))
    }
    
    fn clone_data(&self) -> Self {
        self.clone_data()
    }
}

impl GpuOps for MetalTensor {
    // ========== 二項演算 ==========
    fn add(&self, other: &Self) -> Self { self.add(other) }
    fn sub(&self, other: &Self) -> Self { self.sub(other) }
    fn mul(&self, other: &Self) -> Self { self.mul(other) }
    fn div(&self, other: &Self) -> Self { self.div(other) }
    fn pow(&self, other: &Self) -> Self { self.pow(other) }
    fn matmul(&self, other: &Self) -> Self { self.matmul(other) }
    
    // ========== スカラー演算 ==========
    fn add_scalar(&self, scalar: f32) -> Self { self.add_scalar(scalar) }
    fn mul_scalar(&self, scalar: f32) -> Self { self.mul_scalar(scalar) }
    fn sub_scalar(&self, scalar: f32) -> Self { self.sub_scalar(scalar) }
    fn div_scalar(&self, scalar: f32) -> Self { self.div_scalar(scalar) }
    fn clamp(&self, min: f32, max: f32) -> Self { self.clamp(min, max) }
    
    // ========== 単項演算 ==========
    fn neg(&self) -> Self { self.neg() }
    fn abs(&self) -> Self { self.abs() }
    fn exp(&self) -> Self { self.exp() }
    fn log(&self) -> Self { self.log() }
    fn sqrt(&self) -> Self { self.sqrt() }
    fn sin(&self) -> Self { self.sin() }
    fn cos(&self) -> Self { self.cos() }
    fn tan(&self) -> Self { self.tan() }
    fn tanh(&self) -> Self { self.tanh() }
    fn sigmoid(&self) -> Self { self.sigmoid() }
    fn relu(&self) -> Self { self.relu() }
    fn gelu(&self) -> Self { self.gelu() }
    
    // ========== Reduce 演算 ==========
    fn sumall(&self) -> f32 { self.sumall() }
    fn mean_all(&self) -> f32 { self.mean_all() }
    fn sum(&self, axis: i32) -> Self { self.sum(axis) }
    fn max(&self, axis: i32) -> Self { self.max(axis) }
    fn min(&self, axis: i32) -> Self { self.min(axis) }
    fn argmax(&self, axis: i32) -> Self { self.argmax(axis) }
    fn argmax_all(&self) -> usize { self.argmax_all() }
    fn argmin(&self, axis: i32) -> Self { self.argmin(axis) }
    fn mean(&self, axis: i32) -> Self { self.mean(axis) }
    
    // ========== 形状操作 ==========
    fn reshape(&self, shape: &[usize]) -> Self { self.reshape(shape) }
    fn transpose(&self, dim0: usize, dim1: usize) -> Self { self.transpose(dim0, dim1) }
    fn squeeze(&self, dim: usize) -> Self { self.squeeze(dim) }
    fn unsqueeze(&self, dim: usize) -> Self { self.unsqueeze(dim) }
    fn broadcast_to(&self, shape: &[usize]) -> Self { self.broadcast_to(shape) }
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self { self.narrow(axis, start, len) }
    fn slice(&self, axis: usize, start: usize, len: usize) -> Self { self.slice(axis, start, len) }
    fn contiguous(&self) -> Self { self.contiguous() }
    fn cat(tensors: &[&Self], axis: usize) -> Self { MetalTensor::cat(tensors, axis) }
    
    // ========== 活性化・特殊演算 ==========
    fn softmax(&self, axis: i32) -> Self { self.softmax(axis) }
    fn embedding(&self, indices: &Self) -> Self { self.embedding(indices) }
    fn tril(&self, diagonal: i32) -> Self { self.tril(diagonal) }
    fn cross_entropy(&self, target: &Self) -> Self { self.cross_entropy(target) }
    fn repeat_interleave(&self, repeats: usize, axis: usize) -> Self { self.repeat_interleave(repeats, axis) }
    fn index_select(&self, axis: usize, indices: &Self) -> Self { self.index_select(axis, indices) }
    fn where_cond(condition: &Self, x: &Self, y: &Self) -> Self { MetalTensor::where_cond(condition, x, y) }
    
    // ========== 深層学習演算 ==========
    fn conv2d(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> Self {
        self.conv2d(weight, stride, padding)
    }
    fn batch_norm(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: f32) -> Self {
        self.batch_norm(gamma, beta, running_mean, running_var, eps)
    }
    fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Self {
        self.layer_norm(gamma, beta, eps)
    }
    fn max_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        self.max_pool2d(kernel_size, stride)
    }
    fn avg_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        self.avg_pool2d(kernel_size, stride)
    }
    fn dropout(&self, p: f32, training: bool) -> Self {
        self.dropout(p, training)
    }
}

