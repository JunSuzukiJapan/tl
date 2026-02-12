//! GPU Backend Trait Definitions
//!
//! Metal, CUDA などの GPU バックエンドの共通インターフェースを定義。

pub mod dtype;
pub mod tensor;
pub mod ops;
pub mod autograd;
pub mod error;
pub mod device;

pub use dtype::DType;
pub use tensor::GpuTensor;
pub use autograd::{GpuAutograd, GpuVar};
pub use error::BackendError;
pub use device::IDevice;

/// GPU バックエンドの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Metal,
    Cuda,
    Cpu,
}
