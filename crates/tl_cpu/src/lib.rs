#![allow(clippy::not_unsafe_ptr_arg_deref)]
//! CPU テンソルバックエンド

pub mod tensor;
pub mod autograd;
pub mod backend_impl;
pub mod ffi;
pub mod memory;
pub mod device_impl;

pub use tensor::CpuTensor;
pub use tl_backend::DType;
