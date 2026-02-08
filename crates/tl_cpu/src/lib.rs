//! CPU テンソルバックエンド

pub mod tensor;
pub mod autograd;
pub mod backend_impl;
pub mod ffi;

pub use tensor::CpuTensor;
pub use tl_backend::DType;

