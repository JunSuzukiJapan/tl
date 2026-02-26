//! tl_backend::GpuTensor トレイト実装 + 未実装 _impl スタブ

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{self, BackendResult, DType as BackendDType, GpuTensor};

fn to_backend_dtype(dtype: DType) -> BackendDType {
    match dtype {
        DType::F32 => BackendDType::F32,
        DType::I64 => BackendDType::I64,
        DType::I32 => BackendDType::I32,
        DType::F16 => BackendDType::F16,
        DType::U8 => BackendDType::U8,
    }
}

fn from_backend_dtype(dtype: BackendDType) -> DType {
    match dtype {
        BackendDType::F32 => DType::F32,
        BackendDType::I64 => DType::I64,
        BackendDType::I32 => DType::I32,
        BackendDType::F16 => DType::F16,
        BackendDType::U8 => DType::U8,
        _ => DType::F32,
    }
}

impl GpuTensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn dtype(&self) -> BackendDType {
        to_backend_dtype(self.dtype)
    }
    fn to_vec_f32(&self) -> Vec<f32> {
        self.to_vec::<f32>()
    }
    fn to_vec_i64(&self) -> Vec<i64> {
        self.to_vec::<i64>()
    }

    fn from_slice_f32(data: &[f32], shape: &[usize]) -> BackendResult<Self> {
        Ok(CudaTensor::from_slice(data, shape, DType::F32))
    }
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> BackendResult<Self> {
        Ok(CudaTensor::from_slice(data, shape, DType::I64))
    }
    fn zeros(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::zeros(shape, from_backend_dtype(dtype)))
    }
    fn ones(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::ones(shape, from_backend_dtype(dtype)))
    }
    fn randn(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::randn(shape, from_backend_dtype(dtype)))
    }
    fn arange(start: i64, end: i64, dtype: BackendDType) -> BackendResult<Self> {
        let local_dtype = from_backend_dtype(dtype);
        let count = (end - start) as usize;
        match local_dtype {
            DType::F32 => {
                let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
                Ok(CudaTensor::from_slice(&data, &[count], local_dtype))
            }
            DType::I64 => {
                let data: Vec<i64> = (start..end).collect();
                Ok(CudaTensor::from_slice(&data, &[count], local_dtype))
            }
            _ => {
                let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
                Ok(CudaTensor::from_slice(&data, &[count], DType::F32))
            }
        }
    }
    fn clone_data(&self) -> BackendResult<Self> {
        CudaTensor::clone_data(self)
    }
}

// ========== 未実装スタブ（ops/ に移動する前の一時置き場）==========
// add_impl, sub_impl, mul_impl, div_impl, pow_impl, rem_impl → ops/binary.rs
// neg_impl, abs_impl → ops/unary.rs
// add_scalar_impl, mul_scalar_impl, div_scalar_impl, pow_scalar_impl, scale_impl, clamp_impl → ops/scalar.rs
// eq_impl, ne_impl, lt_impl, le_impl, gt_impl, ge_impl → ops/binary.rs

impl CudaTensor {
    // === 型変換 ===
    pub fn to_dtype(&self, _dtype: DType) -> BackendResult<Self> {
        unimplemented!("to_dtype")
    }

    // === Activation → ops/activation.rs に移動済み ===

    // === Matmul (Step 7 で実装) ===
    pub fn matmul_impl(&self, _other: &Self) -> BackendResult<Self> {
        unimplemented!("matmul_impl")
    }

    // === Reduce (Step 5 で実装) ===
    pub fn sumall_impl(&self) -> BackendResult<f32> {
        unimplemented!("sumall_impl")
    }
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        unimplemented!("mean_all_impl")
    }
    pub fn sum_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("sum_impl")
    }
    pub fn mean_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("mean_impl")
    }
    pub fn max_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("max_impl")
    }
    pub fn min_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("min_impl")
    }
    pub fn argmax_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("argmax_impl")
    }
    pub fn argmin_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("argmin_impl")
    }

    // === Shape (Step 6 で実装) ===
    pub fn reshape_impl(&self, _shape: &[usize]) -> BackendResult<Self> {
        unimplemented!("reshape_impl")
    }
    pub fn transpose_impl(&self, _dim0: usize, _dim1: usize) -> BackendResult<Self> {
        unimplemented!("transpose_impl")
    }
    pub fn squeeze_impl(&self, _dim: usize) -> BackendResult<Self> {
        unimplemented!("squeeze_impl")
    }
    pub fn unsqueeze_impl(&self, _dim: usize) -> BackendResult<Self> {
        unimplemented!("unsqueeze_impl")
    }
    pub fn narrow_impl(&self, _dim: usize, _start: usize, _len: usize) -> BackendResult<Self> {
        unimplemented!("narrow_impl")
    }
    pub fn cat_impl(&self, _other: &Self, _dim: usize) -> BackendResult<Self> {
        unimplemented!("cat_impl")
    }
    pub fn contiguous_impl(&self) -> BackendResult<Self> {
        unimplemented!("contiguous_impl")
    }
    pub fn broadcast_to_impl(&self, _shape: &[usize]) -> BackendResult<Self> {
        unimplemented!("broadcast_to_impl")
    }
    pub fn slice_impl(&self, _dim: usize, _start: usize, _len: usize) -> BackendResult<Self> {
        unimplemented!("slice_impl")
    }

    // === Special (Step 10 で実装) ===
    pub fn softmax_impl(&self, _axis: i32) -> BackendResult<Self> {
        unimplemented!("softmax_impl")
    }
    pub fn embedding_impl(&self, _indices: &Self) -> BackendResult<Self> {
        unimplemented!("embedding_impl")
    }
    pub fn cross_entropy_impl(&self, _target: &Self) -> BackendResult<Self> {
        unimplemented!("cross_entropy_impl")
    }
    pub fn tril_impl(&self, _diagonal: i32) -> BackendResult<Self> {
        unimplemented!("tril_impl")
    }
    pub fn index_select_impl(&self, _axis: usize, _indices: &Self) -> BackendResult<Self> {
        unimplemented!("index_select_impl")
    }
    pub fn where_cond_impl(_cond: &Self, _x: &Self, _y: &Self) -> BackendResult<Self> {
        unimplemented!("where_cond_impl")
    }
    pub fn repeat_interleave_impl(&self, _repeats: usize, _axis: usize) -> BackendResult<Self> {
        unimplemented!("repeat_interleave_impl")
    }

    // === NN (Step 10 で実装) ===
    pub fn conv2d_impl(
        &self,
        _w: &Self,
        _stride: (usize, usize),
        _pad: (usize, usize),
    ) -> BackendResult<Self> {
        unimplemented!("conv2d_impl")
    }
    pub fn batch_norm_impl(
        &self,
        _g: &Self,
        _b: &Self,
        _m: &Self,
        _v: &Self,
        _eps: f32,
    ) -> BackendResult<Self> {
        unimplemented!("batch_norm_impl")
    }
    pub fn layer_norm_impl(&self, _g: &Self, _b: &Self, _eps: f32) -> BackendResult<Self> {
        unimplemented!("layer_norm_impl")
    }
    pub fn max_pool2d_impl(&self, _k: (usize, usize), _s: (usize, usize)) -> BackendResult<Self> {
        unimplemented!("max_pool2d_impl")
    }
    pub fn avg_pool2d_impl(&self, _k: (usize, usize), _s: (usize, usize)) -> BackendResult<Self> {
        unimplemented!("avg_pool2d_impl")
    }
    pub fn dropout_impl(&self, _p: f32, _training: bool) -> BackendResult<Self> {
        unimplemented!("dropout_impl")
    }

    // === LLM ===
    pub fn rms_norm_impl(&self, _eps: f32) -> BackendResult<Self> {
        unimplemented!("rms_norm_impl")
    }

    // === 量子化 ===
    pub fn dequantize_q4_k(&self, _target_shape: &[usize]) -> BackendResult<Self> {
        unimplemented!("dequantize_q4_k")
    }
}
