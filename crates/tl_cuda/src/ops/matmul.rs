//! 行列積

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// 行列積（2D × 2D → 2D）
    pub fn matmul_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(BackendError::ShapeMismatch(format!(
                "matmul requires at least 2D tensors, got {:?} and {:?}",
                a_shape, b_shape
            )));
        }

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        if k != k2 {
            return Err(BackendError::ShapeMismatch(format!(
                "matmul inner dims mismatch: {} vs {}",
                k, k2
            )));
        }

        let a_data = self.to_vec::<f32>();
        let b_data = other.to_vec::<f32>();

        // バッチ matmul 対応
        let batch_size = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };

        let b_batched = b_shape.len() > 2;

        let a_mat_size = m * k;
        let b_mat_size = k * n;
        let c_mat_size = m * n;

        let mut result = vec![0.0f32; batch_size * c_mat_size];

        for batch in 0..batch_size {
            let a_off = batch * a_mat_size;
            let b_off = if b_batched { batch * b_mat_size } else { 0 };
            let c_off = batch * c_mat_size;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                    }
                    result[c_off + i * n + j] = sum;
                }
            }
        }

        let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::F32))
    }
}
