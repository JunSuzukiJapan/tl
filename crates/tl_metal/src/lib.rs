//! tl_metal - Metal GPU Tensor Library
//!
//! Candle を使わず Metal を直接使用するテンソルライブラリ。
//! GPU バッファの真の再利用を実現し、メモリ増加問題を根本解決する。

pub mod buffer_pool;
pub mod command_stream;
pub mod graph;
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
    pub mod fused;
}
pub mod shaders;
pub mod autograd;
pub mod backend_impl;
pub mod optim;
pub mod ffi;
pub mod ffi_ops;
pub mod device_impl;

pub use buffer_pool::MetalBufferPool;
pub use device::{get_device, MetalDevice};
pub use tensor::MetalTensor;
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
