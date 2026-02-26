//! スカラー演算

use crate::tensor::CudaTensor;
use tl_backend::BackendResult;

impl CudaTensor {
    /// スカラー加算
    pub fn add_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x + scalar).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// スカラー乗算
    pub fn mul_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x * scalar).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// スカラー除算
    pub fn div_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x / scalar).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// スカラーべき乗
    pub fn pow_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.powf(scalar)).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// スケール（スカラー乗算の別名）
    pub fn scale_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.mul_scalar_impl(scalar)
    }

    /// クランプ
    pub fn clamp_impl(&self, min: f32, max: f32) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.max(min).min(max)).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }
}
