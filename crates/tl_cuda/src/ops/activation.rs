//! 活性化関数（要素ごと）

use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

impl CudaTensor {
    /// ReLU: max(0, x)
    pub fn relu_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Sigmoid: 1 / (1 + exp(-x))
    pub fn sigmoid_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Tanh
    pub fn tanh_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Exp
    pub fn exp_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Log (natural)
    pub fn log_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.ln()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Sqrt
    pub fn sqrt_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pub fn gelu_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let sqrt_2_over_pi: f32 = (2.0f32 / std::f32::consts::PI).sqrt();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// SiLU (Swish): x * sigmoid(x)
    pub fn silu_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Sin
    pub fn sin_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.sin()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Cos
    pub fn cos_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.cos()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Tan
    pub fn tan_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.tan()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Floor
    pub fn floor_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.floor()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Ceil
    pub fn ceil_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.ceil()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Round
    pub fn round_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.round()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }
}
