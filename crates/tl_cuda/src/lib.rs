//! tl_cuda - CUDA GPU Tensor Library
//!
//! Metal バックエンド (tl_metal) と同じ API を CUDA で実装する。
//! 現時点ではすべてスタブ (unimplemented!) 実装。

pub mod buffer_pool;
pub mod device;
pub mod tensor;
pub mod ops {
    pub mod binary;
    pub mod unary;
    pub mod activation;
    pub mod reduce;
    pub mod reduce_axis;
    pub mod matmul;
    pub mod broadcast;
    pub mod index;
    pub mod shape;
    pub mod special;
    pub mod scalar;
    pub mod llm;
    pub mod nn;
    pub mod quantized;
}
pub mod autograd;
pub mod backend_impl;
pub mod optim;
pub mod ffi;
pub mod ffi_ops;
pub mod device_impl;

pub use buffer_pool::CudaBufferPool;
pub use device::{get_device, CudaDevice};
pub use tensor::CudaTensor;
pub use autograd::GradFn;
pub use optim::{SGD, Adam, AdamW, clip_grad_norm};

// Re-export tl_backend traits
pub use tl_backend::GpuTensor;
pub use ffi::OpaqueTensor;

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
