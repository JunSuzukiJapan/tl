//! 融合カーネル — CUDA スタブ実装

use crate::CudaTensor;
use tl_backend::error::BackendError;
use tl_backend::fused_ops::GpuFusedOps;

type Result<T> = std::result::Result<T, BackendError>;

impl GpuFusedOps for CudaTensor {
    fn fused_silu_mul(&self, _up: &Self) -> Result<Self> {
        unimplemented!("CUDA fused_silu_mul not yet implemented")
    }

    fn fused_rms_norm(&self, _weight: &Self, _eps: f32) -> Result<Self> {
        unimplemented!("CUDA fused_rms_norm not yet implemented")
    }

    fn fused_add_rms_norm(&self, _residual: &Self, _weight: &Self, _eps: f32) -> Result<Self> {
        unimplemented!("CUDA fused_add_rms_norm not yet implemented")
    }

    fn fused_rotary_emb(&self, _freqs: &Self, _head_dim: usize) -> Result<Self> {
        unimplemented!("CUDA fused_rotary_emb not yet implemented")
    }

    fn fused_add_relu(&self, _other: &Self) -> Result<Self> {
        unimplemented!("CUDA fused_add_relu not yet implemented")
    }

    fn fused_bias_gelu(&self, _bias: &Self) -> Result<Self> {
        unimplemented!("CUDA fused_bias_gelu not yet implemented")
    }
}
