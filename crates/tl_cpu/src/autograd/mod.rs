//! Autograd - 自動微分（CPU版）

pub mod ops;

use crate::scalar::TensorScalar;
use crate::tensor::{CpuTensor, TensorRef};
use tl_backend::BackendResult;

/// 勾配関数のトレイト
pub trait GradFn<T: TensorScalar>: Send + Sync {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>>;
    fn inputs(&self) -> Vec<TensorRef<T>>;
}
