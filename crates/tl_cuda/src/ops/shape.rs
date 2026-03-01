//! 形状操作 — reshape/unsqueeze/squeeze/contiguous は GPU buffer 共有（ゼロコピー）
//! transpose/narrow/cat/broadcast_to は CPU フォールバック (Phase C で GPU 化)

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// リシェイプ — GPU buffer 共有（ゼロコピー、to_vec なし）
    pub fn reshape_impl(&self, new_shape: &[usize]) -> BackendResult<CudaTensor> {
        let old_count = self.elem_count();
        let new_count: usize = new_shape.iter().product();
        if old_count != new_count {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot reshape from {:?} ({}) to {:?} ({})",
                self.shape(),
                old_count,
                new_shape,
                new_count
            )));
        }
        Ok(self.view_with_shape(new_shape))
    }

    /// squeeze: 指定次元を除去 — GPU buffer 共有（ゼロコピー）
    pub fn squeeze_impl(&self, dim: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if dim >= shape.len() || shape[dim] != 1 {
            return self.clone_data();
        }
        let mut new_shape = shape;
        new_shape.remove(dim);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        Ok(self.view_with_shape(&new_shape))
    }

    /// unsqueeze: 指定位置にサイズ 1 の次元を挿入 — GPU buffer 共有（ゼロコピー）
    pub fn unsqueeze_impl(&self, dim: usize) -> BackendResult<CudaTensor> {
        let mut new_shape = self.shape().to_vec();
        if dim > new_shape.len() {
            return Err(BackendError::ArgumentError(format!(
                "unsqueeze dim {} out of range for ndim {}",
                dim,
                new_shape.len()
            )));
        }
        new_shape.insert(dim, 1);
        Ok(self.view_with_shape(&new_shape))
    }

    /// contiguous: メモリ配置を連続にする — GPU buffer 共有
    pub fn contiguous_impl(&self) -> BackendResult<CudaTensor> {
        self.clone_data()
    }

    /// 転置（CPU — GPU カーネル化は Phase C）
    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "transpose dims ({}, {}) out of range for ndim {}",
                dim0, dim1, ndim
            )));
        }
        if dim0 == dim1 {
            return self.clone_data();
        }

        let data = self.to_vec::<f32>();
        let mut new_shape = shape.clone();
        new_shape.swap(dim0, dim1);
        let count = data.len();
        let mut result = vec![0.0f32; count];

        let mut old_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            old_strides[d] = old_strides[d + 1] * shape[d + 1];
        }
        let mut new_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            new_strides[d] = new_strides[d + 1] * new_shape[d + 1];
        }

        for i in 0..count {
            let mut rem = i;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / old_strides[d];
                rem %= old_strides[d];
            }
            coords.swap(dim0, dim1);
            let new_idx: usize = coords
                .iter()
                .zip(new_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            result[new_idx] = data[i];
        }

        Ok(CudaTensor::from_slice(&result, &new_shape, self.dtype()))
    }

    /// narrow: 指定次元で部分抽出（CPU — Phase C）
    pub fn narrow_impl(&self, dim: usize, start: usize, len: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if dim >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "narrow dim {} >= ndim {}",
                dim, ndim
            )));
        }
        if start + len > shape[dim] {
            return Err(BackendError::ArgumentError(format!(
                "narrow: start {} + len {} > dim size {}",
                start, len, shape[dim]
            )));
        }

        let data = self.to_vec::<f32>();
        let mut new_shape = shape.clone();
        new_shape[dim] = len;
        let new_count: usize = new_shape.iter().product();
        let mut result = vec![0.0f32; new_count];

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        for out_idx in 0..new_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            let mut new_strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                new_strides[d] = new_strides[d + 1] * new_shape[d + 1];
            }
            for d in 0..ndim {
                coords[d] = rem / new_strides[d];
                rem %= new_strides[d];
            }
            coords[dim] += start;
            let src_idx: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            result[out_idx] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, &new_shape, self.dtype()))
    }

    /// slice（narrow の別名）
    pub fn slice_impl(&self, dim: usize, start: usize, len: usize) -> BackendResult<CudaTensor> {
        self.narrow_impl(dim, start, len)
    }

    /// cat: 指定次元で結合（CPU — Phase C）
    pub fn cat_impl(&self, other: &CudaTensor, dim: usize) -> BackendResult<CudaTensor> {
        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        if a_shape.len() != b_shape.len() {
            return Err(BackendError::ShapeMismatch(format!(
                "cat: rank mismatch {:?} vs {:?}",
                a_shape, b_shape
            )));
        }
        for (i, (&a, &b)) in a_shape.iter().zip(b_shape.iter()).enumerate() {
            if i != dim && a != b {
                return Err(BackendError::ShapeMismatch(format!(
                    "cat: dim {} mismatch: {} vs {}",
                    i, a, b
                )));
            }
        }

        let a_data = self.to_vec::<f32>();
        let b_data = other.to_vec::<f32>();
        let mut out_shape = a_shape.clone();
        out_shape[dim] = a_shape[dim] + b_shape[dim];
        let out_count: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; out_count];

        let ndim = out_shape.len();
        let mut out_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }
        let mut a_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            a_strides[d] = a_strides[d + 1] * a_shape[d + 1];
        }
        let mut b_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            b_strides[d] = b_strides[d + 1] * b_shape[d + 1];
        }

        for out_idx in 0..out_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }

            if coords[dim] < a_shape[dim] {
                let src_idx: usize = coords
                    .iter()
                    .zip(a_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                result[out_idx] = a_data[src_idx];
            } else {
                let mut src_coords = coords.clone();
                src_coords[dim] -= a_shape[dim];
                let src_idx: usize = src_coords
                    .iter()
                    .zip(b_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                result[out_idx] = b_data[src_idx];
            }
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, self.dtype()))
    }

    /// broadcast_to: 指定 shape へブロードキャスト（CPU — Phase C）
    pub fn broadcast_to_impl(&self, target_shape: &[usize]) -> BackendResult<CudaTensor> {
        let src_shape = self.shape();
        let data = self.to_vec::<f32>();
        let out_count: usize = target_shape.iter().product();
        let mut result = vec![0.0f32; out_count];

        let ndim = target_shape.len();
        for i in 0..out_count {
            let mut rem = i;
            let mut src_idx = 0;
            let mut src_stride = 1;

            for d in (0..ndim).rev() {
                let coord = rem % target_shape[d];
                rem /= target_shape[d];

                let src_dim = if d >= ndim - src_shape.len() {
                    src_shape[d - (ndim - src_shape.len())]
                } else {
                    1
                };
                if src_dim > 1 {
                    src_idx += coord * src_stride;
                }
                src_stride *= src_dim;
            }
            result[i] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, target_shape, self.dtype()))
    }
}
