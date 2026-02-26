//! 特殊演算

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// Softmax: softmax(x, axis)
    pub fn softmax_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        let data = self.to_vec::<f32>();
        let count = data.len();
        let axis_size = shape[axis];

        // 出力も同じ shape
        let mut result = vec![0.0f32; count];

        // 軸に沿って softmax を計算
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // max を先に求める（数値安定性）
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..axis_size {
                    let idx = o * axis_size * inner + k * inner + i;
                    max_val = max_val.max(data[idx]);
                }
                // exp(x - max)
                let mut sum_exp = 0.0f32;
                for k in 0..axis_size {
                    let idx = o * axis_size * inner + k * inner + i;
                    let e = (data[idx] - max_val).exp();
                    result[idx] = e;
                    sum_exp += e;
                }
                // normalize
                for k in 0..axis_size {
                    let idx = o * axis_size * inner + k * inner + i;
                    result[idx] /= sum_exp;
                }
            }
        }

        Ok(CudaTensor::from_slice(&result, &shape, DType::F32))
    }

    /// Embedding lookup
    pub fn embedding_impl(&self, indices: &CudaTensor) -> BackendResult<CudaTensor> {
        let weight_shape = self.shape();
        if weight_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(
                "embedding weight must be 2D".into(),
            ));
        }
        let vocab_size = weight_shape[0];
        let embed_dim = weight_shape[1];

        let weight_data = self.to_vec::<f32>();
        let index_data: Vec<i64> = indices.to_vec();

        let batch_size = index_data.len();
        let mut result = vec![0.0f32; batch_size * embed_dim];

        for (i, &idx) in index_data.iter().enumerate() {
            let idx = idx as usize;
            if idx >= vocab_size {
                return Err(BackendError::IndexOutOfBounds(format!(
                    "index {} >= vocab_size {}",
                    idx, vocab_size
                )));
            }
            let src_start = idx * embed_dim;
            result[i * embed_dim..(i + 1) * embed_dim]
                .copy_from_slice(&weight_data[src_start..src_start + embed_dim]);
        }

        let mut out_shape = indices.shape().to_vec();
        out_shape.push(embed_dim);

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::F32))
    }

    /// Cross entropy loss
    pub fn cross_entropy_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        let logits_shape = self.shape();
        if logits_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(
                "cross_entropy logits must be 2D [N, C]".into(),
            ));
        }
        let n = logits_shape[0];
        let c = logits_shape[1];

        let logits_data = self.to_vec::<f32>();
        let target_data: Vec<i64> = target.to_vec();

        let mut total_loss = 0.0f32;
        for i in 0..n {
            // max for numerical stability
            let row = &logits_data[i * c..(i + 1) * c];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_sum_exp = max_val + sum_exp.ln();
            let target_idx = target_data[i] as usize;
            total_loss += log_sum_exp - row[target_idx];
        }
        total_loss /= n as f32;

        Ok(CudaTensor::from_slice(&[total_loss], &[1], DType::F32))
    }

    /// 下三角行列
    pub fn tril_impl(&self, diagonal: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(BackendError::ShapeMismatch("tril requires >= 2D".into()));
        }
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

        let data = self.to_vec::<f32>();
        let mut result = data.clone();

        for b in 0..batch {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = b * rows * cols + r * cols + c;
                    if (c as i32) > (r as i32) + diagonal {
                        result[idx] = 0.0;
                    }
                }
            }
        }

        Ok(CudaTensor::from_slice(&result, shape, self.dtype()))
    }

    /// index_select
    pub fn index_select_impl(
        &self,
        axis: usize,
        indices: &CudaTensor,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let idx_data: Vec<i64> = indices.to_vec();

        let mut out_shape = shape.clone();
        out_shape[axis] = idx_data.len();
        let out_count: usize = out_shape.iter().product();
        let ndim = shape.len();

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            let orig_coord = idx_data[coords[axis]] as usize;
            coords[axis] = orig_coord;
            let src_idx: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            result[out_idx] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, self.dtype()))
    }

    /// where_cond
    pub fn where_cond_impl(
        cond: &CudaTensor,
        x: &CudaTensor,
        y: &CudaTensor,
    ) -> BackendResult<CudaTensor> {
        let cond_data = cond.to_vec::<f32>();
        let x_data = x.to_vec::<f32>();
        let y_data = y.to_vec::<f32>();
        let result: Vec<f32> = cond_data
            .iter()
            .zip(x_data.iter().zip(y_data.iter()))
            .map(|(&c, (&xv, &yv))| if c > 0.0 { xv } else { yv })
            .collect();
        Ok(CudaTensor::from_slice(&result, x.shape(), x.dtype()))
    }

    /// repeat_interleave
    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let ndim = shape.len();

        let mut out_shape = shape.clone();
        out_shape[axis] *= repeats;
        let out_count: usize = out_shape.iter().product();

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            coords[axis] /= repeats;
            let src_idx: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            result[out_idx] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, self.dtype()))
    }
}
