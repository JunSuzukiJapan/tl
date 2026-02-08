//! Autograd - 自動微分（CPU版）

pub mod ops;

use crate::tensor::CpuTensor;

/// 勾配関数のトレイト
pub trait GradFn {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor>;
    fn inputs(&self) -> Vec<*mut CpuTensor>;
}
