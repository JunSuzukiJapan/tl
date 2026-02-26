//! tl_cuda - CUDA GPU Tensor Library
//!
//! Metal バックエンド (tl_metal) と同じ API を CUDA で実装する。
#![allow(unused)]

#[macro_use]
pub mod cuda_sys;
pub mod buffer_pool;
pub mod device;
pub mod stream;
pub mod tensor;
pub mod ops {
    pub mod activation;
    pub mod binary;
    pub mod broadcast;
    pub mod fused;
    pub mod index;
    pub mod llm;
    pub mod matmul;
    pub mod nn;
    pub mod quantized;
    pub mod reduce;
    pub mod reduce_axis;
    pub mod scalar;
    pub mod shape;
    pub mod special;
    pub mod unary;
}
pub mod autograd;
pub mod backend_impl;
pub mod device_impl;
pub mod ffi;
pub mod ffi_ops;
pub mod graph;
pub mod optim;

pub use autograd::GradFn;
pub use buffer_pool::CudaBufferPool;
pub use device::{get_device, CudaDevice};
pub use optim::{clip_grad_norm, Adam, AdamW, SGD};
pub use tensor::CudaTensor;

// Re-export tl_backend traits
pub use ffi::OpaqueTensor;
pub use tl_backend::GpuTensor;

/// データ型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I32,
    I64,
    U8,
}

impl DType {
    /// バイト数
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}

/// 形状からバイト数を計算
pub fn shape_to_bytes(shape: &[usize], dtype: DType) -> usize {
    let elem_count: usize = shape.iter().product();
    elem_count * dtype.size_in_bytes()
}
