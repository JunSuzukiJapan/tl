//! CUDA 二項演算スタブ
use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

impl CudaTensor {
    pub fn binary_op(&self, _other: &CudaTensor, _op_name: &str) -> BackendResult<CudaTensor> {
        unimplemented!("CudaTensor::binary_op")
    }
}
