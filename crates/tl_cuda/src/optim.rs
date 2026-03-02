//! CUDA オプティマイザ — tl_backend ジェネリック実装の re-export
//!
//! SGD, Adam, AdamW, clip_grad_norm は tl_backend の共通実装を使用。

use crate::tensor::CudaTensor;

/// SGD optimizer (CudaTensor 特殊化)
pub type SGD = tl_backend::optim::SGD<CudaTensor>;

/// Adam optimizer (CudaTensor 特殊化)
pub type Adam = tl_backend::optim::Adam<CudaTensor>;

/// AdamW optimizer (CudaTensor 特殊化)
pub type AdamW = tl_backend::optim::AdamW<CudaTensor>;

/// 勾配ノルムのクリッピング
pub fn clip_grad_norm(grads: &mut [CudaTensor], max_norm: f32) -> f32 {
    tl_backend::optim::clip_grad_norm(grads, max_norm)
}
