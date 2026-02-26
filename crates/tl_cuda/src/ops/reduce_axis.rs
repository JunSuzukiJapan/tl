//! 軸付きリダクション演算

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// 軸に沿った合計
    pub fn sum_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_impl(axis, false, |slice| slice.iter().sum())
    }

    /// 軸に沿った平均
    pub fn mean_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_impl(axis, false, |slice| {
            let len = slice.len() as f32;
            slice.iter().sum::<f32>() / len
        })
    }

    /// 軸に沿った最大値
    pub fn max_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_impl(axis, false, |slice| {
            slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        })
    }

    /// 軸に沿った最小値
    pub fn min_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_impl(axis, false, |slice| {
            slice.iter().cloned().fold(f32::INFINITY, f32::min)
        })
    }

    /// 軸に沿った argmax
    pub fn argmax_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range for ndim {}",
                axis, ndim
            )));
        }

        let data = self.to_vec::<f32>();
        let axis_size = shape[axis];

        // 出力 shape
        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_count: usize = out_shape.iter().product();

        let mut result = vec![0i64; out_count];
        for out_idx in 0..out_count {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for k in 0..axis_size {
                let src_idx = self.reduce_src_index(out_idx, k, axis, &shape);
                if data[src_idx] > best_v {
                    best_v = data[src_idx];
                    best_i = k;
                }
            }
            result[out_idx] = best_i as i64;
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::I64))
    }

    /// 軸に沿った argmin
    pub fn argmin_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range for ndim {}",
                axis, ndim
            )));
        }

        let data = self.to_vec::<f32>();
        let axis_size = shape[axis];

        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_count: usize = out_shape.iter().product();

        let mut result = vec![0i64; out_count];
        for out_idx in 0..out_count {
            let mut best_i = 0usize;
            let mut best_v = f32::INFINITY;
            for k in 0..axis_size {
                let src_idx = self.reduce_src_index(out_idx, k, axis, &shape);
                if data[src_idx] < best_v {
                    best_v = data[src_idx];
                    best_i = k;
                }
            }
            result[out_idx] = best_i as i64;
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::I64))
    }

    /// 軸リダクションの共通実装
    fn reduce_axis_impl<F>(
        &self,
        axis: i32,
        _keep_dim: bool,
        reduce_fn: F,
    ) -> BackendResult<CudaTensor>
    where
        F: Fn(&[f32]) -> f32,
    {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range",
                axis
            )));
        }

        let data = self.to_vec::<f32>();
        let axis_size = shape[axis];

        // 出力 shape（軸を除去）
        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_count: usize = out_shape.iter().product();

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut slice = vec![0.0f32; axis_size];
            for k in 0..axis_size {
                let src_idx = self.reduce_src_index(out_idx, k, axis, &shape);
                slice[k] = data[src_idx];
            }
            result[out_idx] = reduce_fn(&slice);
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::F32))
    }

    /// reduce 用: 出力 flat index + 軸位置 k → ソース flat index
    fn reduce_src_index(&self, out_flat: usize, k: usize, axis: usize, shape: &[usize]) -> usize {
        let ndim = shape.len();
        // 出力の多次元 index を計算
        let mut out_shape = shape.to_vec();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            return k;
        }

        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = out_flat;
        for d in (0..out_shape.len()).rev() {
            out_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        // ソースの多次元 index を再構築
        let mut src_coords = Vec::with_capacity(ndim);
        let mut out_d = 0;
        for d in 0..ndim {
            if d == axis {
                src_coords.push(k);
            } else {
                src_coords.push(out_coords[out_d]);
                out_d += 1;
            }
        }

        // flat index
        let mut idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            idx += src_coords[d] * stride;
            stride *= shape[d];
        }
        idx
    }
}
