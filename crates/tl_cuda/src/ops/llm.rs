//! LLM 特化演算

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// RMS Normalization: x * rsqrt(mean(x²) + eps)
    pub fn rms_norm_impl(&self, eps: f32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let norm_size = *shape.last().unwrap();
        let outer = data.len() / norm_size;
        let mut result = vec![0.0f32; data.len()];

        for o in 0..outer {
            let offset = o * norm_size;
            let slice = &data[offset..offset + norm_size];

            let rms: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / norm_size as f32;
            let inv_rms = 1.0 / (rms + eps).sqrt();

            for i in 0..norm_size {
                result[offset + i] = slice[i] * inv_rms;
            }
        }

        Ok(CudaTensor::from_slice(&result, &shape, DType::F32))
    }

    /// RoPE cos/sin テーブル生成
    pub fn rope_cos_sin_impl(
        seq_len: usize,
        dim: usize,
        freq_base: f32,
    ) -> BackendResult<(CudaTensor, CudaTensor)> {
        let half_dim = dim / 2;
        let mut cos_data = vec![0.0f32; seq_len * dim];
        let mut sin_data = vec![0.0f32; seq_len * dim];

        for pos in 0..seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / freq_base.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let s = angle.sin();

                cos_data[pos * dim + i] = c;
                cos_data[pos * dim + half_dim + i] = c;
                sin_data[pos * dim + i] = s;
                sin_data[pos * dim + half_dim + i] = s;
            }
        }

        let cos = CudaTensor::from_slice(&cos_data, &[seq_len, dim], DType::F32);
        let sin = CudaTensor::from_slice(&sin_data, &[seq_len, dim], DType::F32);
        Ok((cos, sin))
    }

    /// RoPE 適用
    pub fn apply_rope_impl(
        &self,
        cos: &CudaTensor,
        sin: &CudaTensor,
        pos: usize,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let cos_data = cos.to_vec::<f32>();
        let sin_data = sin.to_vec::<f32>();

        let dim = *shape.last().unwrap();
        let half_dim = dim / 2;
        let outer = data.len() / dim;

        let cos_shape = cos.shape();
        let cos_dim = cos_shape[cos_shape.len() - 1];

        let mut result = data.clone();

        for o in 0..outer {
            let offset = o * dim;
            let cos_offset = pos * cos_dim;

            for i in 0..half_dim {
                let x0 = data[offset + i];
                let x1 = data[offset + half_dim + i];
                let c = cos_data[cos_offset + i];
                let s = sin_data[cos_offset + i];

                result[offset + i] = x0 * c - x1 * s;
                result[offset + half_dim + i] = x0 * s + x1 * c;
            }
        }

        Ok(CudaTensor::from_slice(&result, &shape, DType::F32))
    }

    /// Causal mask: 上三角を -inf にしたマスク
    pub fn causal_mask_impl(size: usize) -> BackendResult<CudaTensor> {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                if j > i {
                    data[i * size + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(CudaTensor::from_slice(&data, &[size, size], DType::F32))
    }
}
