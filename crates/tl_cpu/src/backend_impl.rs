//! GpuTensor / GpuOps トレイトの CpuTensor 実装

use tl_backend::tensor::GpuTensor;
use tl_backend::ops::GpuOps;
use tl_backend::DType as BackendDType;
use crate::tensor::CpuTensor;
use crate::DType;

fn to_backend_dtype(dtype: DType) -> BackendDType {
    match dtype {
        DType::F32 => BackendDType::F32,
        DType::F16 => BackendDType::F16,
        DType::I32 => BackendDType::I32,
        DType::I64 => BackendDType::I64,
        DType::U8 => BackendDType::U8,
        _ => BackendDType::F32,
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

impl GpuTensor for CpuTensor {
    fn shape(&self) -> &[usize] { CpuTensor::shape(self) }
    fn dtype(&self) -> BackendDType { to_backend_dtype(CpuTensor::dtype(self)) }
    fn to_vec_f32(&self) -> Vec<f32> { self.to_vec::<f32>() }
    fn to_vec_i64(&self) -> Vec<i64> { self.to_vec::<f32>().into_iter().map(|x| x as i64).collect() }
    fn from_slice_f32(data: &[f32], shape: &[usize]) -> Self { CpuTensor::from_slice(data, shape, DType::F32) }
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> Self {
        let f32_data: Vec<f32> = data.iter().map(|x| *x as f32).collect();
        CpuTensor::from_slice(&f32_data, shape, DType::F32)
    }
    fn zeros(shape: &[usize], dtype: BackendDType) -> Self { CpuTensor::zeros(shape, from_backend_dtype(dtype)) }
    fn ones(shape: &[usize], dtype: BackendDType) -> Self { CpuTensor::ones(shape, from_backend_dtype(dtype)) }
    fn randn(shape: &[usize], dtype: BackendDType) -> Self { CpuTensor::randn(shape, from_backend_dtype(dtype)) }
    fn arange(start: i64, end: i64, dtype: BackendDType) -> Self { CpuTensor::arange(start, end, from_backend_dtype(dtype)) }
    fn clone_data(&self) -> Self { CpuTensor::clone_data(self) }
}

impl GpuOps for CpuTensor {
    fn add(&self, other: &Self) -> Self { self.add_impl(other) }
    fn sub(&self, other: &Self) -> Self { self.sub_impl(other) }
    fn mul(&self, other: &Self) -> Self { self.mul_impl(other) }
    fn div(&self, other: &Self) -> Self { self.div_impl(other) }
    fn pow(&self, other: &Self) -> Self { self.pow_impl(other) }
    fn matmul(&self, other: &Self) -> Self { self.matmul_impl(other) }

    fn add_scalar(&self, scalar: f32) -> Self { self.add_scalar_impl(scalar) }
    fn mul_scalar(&self, scalar: f32) -> Self { self.mul_scalar_impl(scalar) }
    fn sub_scalar(&self, scalar: f32) -> Self { self.sub_scalar_impl(scalar) }
    fn div_scalar(&self, scalar: f32) -> Self { self.div_scalar_impl(scalar) }
    fn clamp(&self, min: f32, max: f32) -> Self { self.clamp_impl(min, max) }

    fn neg(&self) -> Self { self.neg_impl() }
    fn abs(&self) -> Self { self.abs_impl() }
    fn exp(&self) -> Self { self.exp_impl() }
    fn log(&self) -> Self { self.log_impl() }
    fn sqrt(&self) -> Self { self.sqrt_impl() }
    fn sin(&self) -> Self { self.sin_impl() }
    fn cos(&self) -> Self { self.cos_impl() }
    fn tan(&self) -> Self { self.tan_impl() }
    fn tanh(&self) -> Self { self.tanh_impl() }
    fn sigmoid(&self) -> Self { self.sigmoid_impl() }
    fn relu(&self) -> Self { self.relu_impl() }
    fn gelu(&self) -> Self { self.gelu_impl() }

    fn sumall(&self) -> f32 { self.sumall_impl() }
    fn mean_all(&self) -> f32 { self.mean_all_impl() }
    fn sum(&self, axis: i32) -> Self { self.sum_impl(axis) }
    fn max(&self, axis: i32) -> Self { self.max_impl(axis) }
    fn min(&self, axis: i32) -> Self { self.min_impl(axis) }
    fn argmax(&self, axis: i32) -> Self { self.argmax_impl(axis) }
    fn argmax_all(&self) -> usize { self.argmax_all_impl() }
    fn argmin(&self, axis: i32) -> Self { self.argmin_impl(axis) }
    fn mean(&self, axis: i32) -> Self { self.mean_impl(axis) }

    fn reshape(&self, shape: &[usize]) -> Self { self.reshape_impl(shape) }
    fn transpose(&self, dim0: usize, dim1: usize) -> Self { self.transpose_impl(dim0, dim1) }
    fn squeeze(&self, dim: usize) -> Self { self.squeeze_impl(dim) }
    fn unsqueeze(&self, dim: usize) -> Self { self.unsqueeze_impl(dim) }
    fn broadcast_to(&self, shape: &[usize]) -> Self { self.broadcast_to_impl(shape) }
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self { self.narrow_impl(axis, start, len) }
    fn slice(&self, axis: usize, start: usize, len: usize) -> Self { self.slice_impl(axis, start, len) }
    fn contiguous(&self) -> Self { self.contiguous_impl() }
    fn cat(tensors: &[&Self], axis: usize) -> Self { CpuTensor::cat_impl(tensors, axis) }

    fn softmax(&self, axis: i32) -> Self { self.softmax_impl(axis) }
    fn embedding(&self, indices: &Self) -> Self { self.embedding_impl(indices) }
    fn tril(&self, diagonal: i32) -> Self { self.tril_impl(diagonal) }
    fn cross_entropy(&self, target: &Self) -> Self { self.cross_entropy_impl(target) }
    fn repeat_interleave(&self, repeats: usize, axis: usize) -> Self { self.repeat_interleave_impl(repeats, axis) }
    fn index_select(&self, axis: usize, indices: &Self) -> Self { self.index_select_impl(axis, indices) }
    fn where_cond(condition: &Self, x: &Self, y: &Self) -> Self { CpuTensor::where_cond_impl(condition, x, y) }

    fn conv2d(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> Self { self.conv2d_impl(weight, stride, padding) }
    fn batch_norm(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: f32) -> Self {
        self.batch_norm_impl(gamma, beta, running_mean, running_var, eps)
    }
    fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Self { self.layer_norm_impl(gamma, beta, eps) }
    fn max_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self { self.max_pool2d_impl(kernel_size, stride) }
    fn avg_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self { self.avg_pool2d_impl(kernel_size, stride) }
    fn dropout(&self, p: f32, training: bool) -> Self { self.dropout_impl(p, training) }
}
