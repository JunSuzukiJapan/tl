//! GPU Backend Trait Definitions
//!
//! Metal, CUDA などの GPU バックエンドの共通インターフェースを定義。

pub mod dtype;
pub mod tensor;
pub mod ops;
pub mod autograd;
pub mod error;
pub mod device;
pub mod graph;
pub mod fused_ops;

pub use dtype::DType;
pub use tensor::GpuTensor;
pub use autograd::{GpuAutograd, GpuVar};
pub use error::BackendError;
pub use device::{IDevice, BackendResult};
pub use fused_ops::GpuFusedOps;

/// GPU バックエンドの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Metal,
    Cuda,
    Cpu,
}
