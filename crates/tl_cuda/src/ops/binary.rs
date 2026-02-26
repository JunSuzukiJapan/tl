//! 二項演算（要素ごと）

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

/// NumPy スタイルの broadcast shape 計算
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> BackendResult<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        let out_dim = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        };
        result[max_rank - 1 - i] = out_dim;
    }
    Ok(result)
}

/// broadcast 用: flat index をソーステンソルの index に変換
fn broadcast_index(flat_idx: usize, out_shape: &[usize], src_shape: &[usize]) -> usize {
    let rank = out_shape.len();
    let mut idx = flat_idx;
    let mut src_idx = 0;
    let mut src_stride = 1;

    for d in (0..rank).rev() {
        let out_dim = out_shape[d];
        let coord = idx % out_dim;
        idx /= out_dim;

        let src_dim = if d >= rank - src_shape.len() {
            src_shape[d - (rank - src_shape.len())]
        } else {
            1
        };

        if src_dim > 1 {
            src_idx += coord * src_stride;
        }
        src_stride *= src_dim;
    }
    src_idx
}

impl CudaTensor {
    /// 二項演算の共通実装
    fn binary_op<F: Fn(f32, f32) -> f32>(
        &self,
        other: &CudaTensor,
        op: F,
    ) -> BackendResult<CudaTensor> {
        if self.dtype() != other.dtype() {
            return Err(BackendError::TypeMismatch(format!(
                "DType mismatch: {:?} vs {:?}",
                self.dtype(),
                other.dtype()
            )));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();

        // broadcast shape を計算
        let out_shape = if self_shape == other_shape {
            self_shape.to_vec()
        } else {
            broadcast_shape(self_shape, other_shape)?
        };

        let out_count = out_shape.iter().product::<usize>();
        let a_data = self.to_vec::<f32>();
        let b_data = other.to_vec::<f32>();

        let result_data: Vec<f32> = if self_shape == other_shape {
            // 同一 shape → 直接 elementwise
            a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&a, &b)| op(a, b))
                .collect()
        } else {
            // broadcast
            (0..out_count)
                .map(|i| {
                    let a_idx = broadcast_index(i, &out_shape, self_shape);
                    let b_idx = broadcast_index(i, &out_shape, other_shape);
                    op(a_data[a_idx], b_data[b_idx])
                })
                .collect()
        };

        Ok(CudaTensor::from_slice(
            &result_data,
            &out_shape,
            self.dtype(),
        ))
    }

    /// 要素ごとの加算
    pub fn add_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a + b)
    }

    /// 要素ごとの減算
    pub fn sub_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a - b)
    }

    /// 要素ごとの乗算
    pub fn mul_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a * b)
    }

    /// 要素ごとの除算
    pub fn div_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a / b)
    }

    /// 要素ごとのべき乗
    pub fn pow_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a.powf(b))
    }

    /// 要素ごとの剰余
    pub fn rem_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| a % b)
    }

    // ========== 比較演算 ==========

    pub fn eq_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 })
    }
    pub fn ne_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 })
    }
    pub fn lt_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }
    pub fn le_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }
    pub fn gt_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }
    pub fn ge_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }
}
