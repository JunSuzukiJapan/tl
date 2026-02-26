//! CUDA Autograd

pub mod ops;

use crate::tensor::{CudaTensor, TensorRef};
use tl_backend::BackendResult;

/// Autograd 勾配関数トレイト（V5.0 Arc ベース）
pub trait GradFn: Send + Sync {
    /// backward 計算
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>>;
    /// 入力テンソルへの参照（TensorRef = Arc<UnsafeCell<CudaTensor>>）
    fn inputs(&self) -> Vec<TensorRef>;
}
