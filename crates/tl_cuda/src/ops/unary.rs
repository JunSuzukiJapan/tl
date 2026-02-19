//! CUDA 単項演算スタブ
use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

impl CudaTensor {
    pub fn unary_op(&self, _op_name: &str) -> BackendResult<CudaTensor> {
        unimplemented!("CudaTensor::unary_op")
    }
}
