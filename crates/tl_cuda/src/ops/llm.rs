//! LLM 特化演算 — rms_norm/causal_mask は GPU カーネル

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_rms_norm_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        norm_size: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_causal_mask_kernel(output: *mut f32, size: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// RMS Normalization — GPU カーネル
    pub fn rms_norm_impl(&self, eps: f32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let norm_size = *shape.last().unwrap();
        let outer = self.elem_count() / norm_size;

        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_rms_norm_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                norm_size as i32,
                eps,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// RoPE cos/sin テーブル生成（CPU — テーブル生成は一度きり）
    pub fn rope_cos_sin_impl(
        seq_len: usize,
        dim: usize,
        freq_base: f32,
    ) -> BackendResult<(CudaTensor, CudaTensor)> {
        let half_dim = dim / 2;
        let mut cos_data = vec![0.0f32; seq_len * half_dim];
        let mut sin_data = vec![0.0f32; seq_len * half_dim];

        for pos in 0..seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / freq_base.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos = CudaTensor::from_slice(&cos_data, &[seq_len, half_dim], DType::F32);
        let sin = CudaTensor::from_slice(&sin_data, &[seq_len, half_dim], DType::F32);
        Ok((cos, sin))
    }

    /// RoPE 適用（CPU — 複雑な座標変換のため）
    pub fn apply_rope_impl(
        &self,
        cos: &CudaTensor,
        sin: &CudaTensor,
        _pos: usize,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let cos_data = cos.to_vec::<f32>();
        let sin_data = sin.to_vec::<f32>();

        let dim = *shape.last().unwrap();
        let half_dim = dim / 2;

        let cos_shape = cos.shape();
        let cos_dim = cos_shape[cos_shape.len() - 1];
        let cos_seq_len = if cos_shape.len() >= 2 {
            cos_shape[0]
        } else {
            1
        };

        let (num_tokens, heads_per_token) = if shape.len() == 4 {
            (shape[1], shape[0] * shape[2])
        } else if shape.len() == 3 {
            (shape[0], shape[1])
        } else if shape.len() == 2 {
            (shape[0], 1usize)
        } else {
            (1, data.len() / dim)
        };

        let mut result = data.clone();

        for token_pos in 0..num_tokens {
            let cos_row = token_pos.min(cos_seq_len.saturating_sub(1));
            let cos_offset = cos_row * cos_dim;

            for head in 0..heads_per_token {
                let offset = (token_pos * heads_per_token + head) * dim;

                for i in 0..half_dim {
                    let ci = if cos_offset + i < cos_data.len() {
                        cos_data[cos_offset + i]
                    } else {
                        1.0
                    };
                    let si = if cos_offset + i < sin_data.len() {
                        sin_data[cos_offset + i]
                    } else {
                        0.0
                    };

                    let x0 = data[offset + i];
                    let x1 = data[offset + half_dim + i];

                    result[offset + i] = x0 * ci - x1 * si;
                    result[offset + half_dim + i] = x0 * si + x1 * ci;
                }
            }
        }

        Ok(CudaTensor::from_slice(&result, &shape, DType::F32))
    }

    /// Causal mask — GPU カーネル
    pub fn causal_mask_impl(size: usize) -> BackendResult<CudaTensor> {
        let output = CudaTensor::uninit(&[size, size], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_causal_mask_kernel(output.buffer.ptr() as *mut f32, size as i32, stream);
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
