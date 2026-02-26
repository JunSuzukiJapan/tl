//! 単項演算（要素ごと）

use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

impl CudaTensor {
    /// 単項演算の共通実装
    fn unary_op<F: Fn(f32) -> f32>(&self, op: F) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| op(x)).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// 負数
    pub fn neg_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_op(|x| -x)
    }

    /// 絶対値
    pub fn abs_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_op(|x| x.abs())
    }
}
