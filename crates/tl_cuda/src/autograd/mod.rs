//! CUDA Autograd

pub mod ops;

use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

/// Autograd 勾配関数トレイト
pub trait GradFn: Send + Sync {
    /// backward 計算
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>>;
    /// 入力テンソルへの参照
    fn inputs(&self) -> Vec<()>;
}
