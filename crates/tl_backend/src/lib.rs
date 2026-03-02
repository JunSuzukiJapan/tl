//! GPU Backend Trait Definitions
//!
//! Metal, CUDA などの GPU バックエンドの共通インターフェースを定義。

pub mod autograd;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ffi_helpers;
pub mod fused_ops;
pub mod fusion;
pub mod graph;
pub mod ops;
pub mod optim;
pub mod stream;
pub mod tensor;

pub use autograd::{GpuAutograd, GpuVar};
pub use device::{BackendResult, IDevice};
pub use dtype::DType;
pub use error::BackendError;
pub use fused_ops::GpuFusedOps;
pub use stream::GpuStream;
pub use tensor::GpuTensor;

/// GPU バックエンドの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Metal,
    Cuda,
    Cpu,
}
