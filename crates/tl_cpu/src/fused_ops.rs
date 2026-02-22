//! 融合カーネル — CPU 実装
//!
//! GPU のような融合の性能メリットはないが、
//! バックエンド互換性のために実装する。
//! 内部的には個別操作を逐次実行する。

use crate::CpuTensor;
use crate::DType;
use tl_backend::error::BackendError;
use tl_backend::fused_ops::GpuFusedOps;

type Result<T> = std::result::Result<T, BackendError>;

/// SiLU 関数: x / (1 + exp(-x))
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU 関数: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
}

impl GpuFusedOps for CpuTensor {
    fn fused_silu_mul(&self, up: &Self) -> Result<Self> {
        let count = self.elem_count();
        assert_eq!(count, up.elem_count(), "fused_silu_mul: shape mismatch");

        let data: Vec<f32> = self.data_f32.iter()
            .zip(up.data_f32.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        Ok(CpuTensor::from_slice(&data, self.shape(), DType::F32))
    }

    fn fused_rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape();
        let d = *shape.last().unwrap();
        let n = self.elem_count() / d;
        let mut data = vec![0.0f32; self.elem_count()];

        for row in 0..n {
            let offset = row * d;
            let row_data = &self.data_f32[offset..offset + d];

            // 二乗平均
            let sum_sq: f32 = row_data.iter().map(|x| x * x).sum();
            let rms = (sum_sq / d as f32 + eps).sqrt().recip();

            for col in 0..d {
                data[offset + col] = row_data[col] * rms * weight.data_f32[col];
            }
        }

        Ok(CpuTensor::from_slice(&data, shape, DType::F32))
    }

    fn fused_add_rms_norm(&self, residual: &Self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape();
        assert_eq!(self.elem_count(), residual.elem_count(), "fused_add_rms_norm: shape mismatch");
        let d = *shape.last().unwrap();
        let n = self.elem_count() / d;
        let mut data = vec![0.0f32; self.elem_count()];

        for row in 0..n {
            let offset = row * d;

            // residual 加算 + 二乗平均
            let sum_sq: f32 = (0..d)
                .map(|i| {
                    let v = self.data_f32[offset + i] + residual.data_f32[offset + i];
                    v * v
                })
                .sum();
            let rms = (sum_sq / d as f32 + eps).sqrt().recip();

            for col in 0..d {
                let added = self.data_f32[offset + col] + residual.data_f32[offset + col];
                data[offset + col] = added * rms * weight.data_f32[col];
            }
        }

        Ok(CpuTensor::from_slice(&data, shape, DType::F32))
    }

    fn fused_rotary_emb(&self, freqs: &Self, head_dim: usize) -> Result<Self> {
        let count = self.elem_count();
        let half_dim = head_dim / 2;
        let mut data = vec![0.0f32; count];

        for id in 0..(count / 2) {
            let pair_idx = id % half_dim;
            let base = id - pair_idx;

            let freq = freqs.data_f32[pair_idx];
            let cos_val = freq.cos();
            let sin_val = freq.sin();

            let x0 = self.data_f32[base + pair_idx];
            let x1 = self.data_f32[base + pair_idx + half_dim];

            data[base + pair_idx] = x0 * cos_val - x1 * sin_val;
            data[base + pair_idx + half_dim] = x0 * sin_val + x1 * cos_val;
        }

        Ok(CpuTensor::from_slice(&data, self.shape(), DType::F32))
    }

    fn fused_add_relu(&self, other: &Self) -> Result<Self> {
        let count = self.elem_count();
        assert_eq!(count, other.elem_count(), "fused_add_relu: shape mismatch");

        let data: Vec<f32> = self.data_f32.iter()
            .zip(other.data_f32.iter())
            .map(|(&a, &b)| (a + b).max(0.0))
            .collect();

        Ok(CpuTensor::from_slice(&data, self.shape(), DType::F32))
    }

    fn fused_bias_gelu(&self, bias: &Self) -> Result<Self> {
        let bias_len = bias.elem_count();

        let data: Vec<f32> = self.data_f32.iter()
            .enumerate()
            .map(|(i, &x)| gelu(x + bias.data_f32[i % bias_len]))
            .collect();

        Ok(CpuTensor::from_slice(&data, self.shape(), DType::F32))
    }
}
