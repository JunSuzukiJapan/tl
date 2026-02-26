//! CUDA Autograd 演算のスタブ
//!
//! tl_metal/src/autograd/ops.rs の全 GradFn 実装に対応するスタブ。

use crate::autograd::GradFn;
use crate::tensor::{CudaTensor, TensorRef};
use tl_backend::BackendResult;

// ========== 基本二項演算 ==========

pub struct AddBackward;
impl GradFn for AddBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("AddBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SubBackward;
impl GradFn for SubBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SubBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct MulBackward;
impl GradFn for MulBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("MulBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct DivBackward;
impl GradFn for DivBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("DivBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct PowBackward;
impl GradFn for PowBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("PowBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct MatmulBackward;
impl GradFn for MatmulBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("MatmulBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// ========== 単項演算 ==========

pub struct NegBackward;
impl GradFn for NegBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("NegBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct AbsBackward;
impl GradFn for AbsBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("AbsBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct ExpBackward;
impl GradFn for ExpBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("ExpBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct LogBackward;
impl GradFn for LogBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("LogBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SqrtBackward;
impl GradFn for SqrtBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SqrtBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct TanhBackward;
impl GradFn for TanhBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("TanhBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SigmoidBackward;
impl GradFn for SigmoidBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SigmoidBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct ReluBackward;
impl GradFn for ReluBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("ReluBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct GeluBackward;
impl GradFn for GeluBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("GeluBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SiluBackward;
impl GradFn for SiluBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SiluBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// ========== リダクション ==========

pub struct SumallBackward;
impl GradFn for SumallBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SumallBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct MeanAllBackward;
impl GradFn for MeanAllBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("MeanAllBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SumDimBackward;
impl GradFn for SumDimBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SumDimBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct MeanDimBackward;
impl GradFn for MeanDimBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("MeanDimBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// ========== 形状操作 ==========

pub struct ReshapeBackward;
impl GradFn for ReshapeBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("ReshapeBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct TransposeBackward;
impl GradFn for TransposeBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("TransposeBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// ========== 特殊 ==========

pub struct SoftmaxBackward;
impl GradFn for SoftmaxBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SoftmaxBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct CrossEntropyBackward;
impl GradFn for CrossEntropyBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("CrossEntropyBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct EmbeddingBackward;
impl GradFn for EmbeddingBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("EmbeddingBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct LayerNormBackward;
impl GradFn for LayerNormBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("LayerNormBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct ScaleBackward;
impl GradFn for ScaleBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("ScaleBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct Conv2dBackward;
impl GradFn for Conv2dBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("Conv2dBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct BatchNormBackward;
impl GradFn for BatchNormBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("BatchNormBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct DropoutBackward;
impl GradFn for DropoutBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("DropoutBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

// ========== スカラー演算 ==========

pub struct AddScalarBackward;
impl GradFn for AddScalarBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("AddScalarBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct SubScalarBackward;
impl GradFn for SubScalarBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("SubScalarBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct MulScalarBackward;
impl GradFn for MulScalarBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("MulScalarBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}

pub struct DivScalarBackward;
impl GradFn for DivScalarBackward {
    fn backward(&self, _grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        unimplemented!("DivScalarBackward")
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![]
    }
}
