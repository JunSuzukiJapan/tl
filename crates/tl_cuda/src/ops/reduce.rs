//! リダクション演算

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

impl CudaTensor {
    /// 全要素の合計（スカラー返却）
    pub fn sumall_impl(&self) -> BackendResult<f32> {
        let data = self.to_vec::<f32>();
        Ok(data.iter().sum())
    }

    /// 全要素の平均（スカラー返却）
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        let data = self.to_vec::<f32>();
        let len = data.len() as f32;
        Ok(data.iter().sum::<f32>() / len)
    }

    /// 全要素の合計（テンソル返却: shape=[1]）
    pub fn sum_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
        let s = self.sumall_impl()?;
        Ok(CudaTensor::from_slice(&[s], &[1], DType::F32))
    }

    /// 全要素の平均（テンソル返却: shape=[1]）
    pub fn mean_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
        let m = self.mean_all_impl()?;
        Ok(CudaTensor::from_slice(&[m], &[1], DType::F32))
    }

    /// 全要素の最大値（テンソル返却）
    pub fn max_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Ok(CudaTensor::from_slice(&[max], &[1], DType::F32))
    }

    /// 全要素の最小値（テンソル返却）
    pub fn min_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        Ok(CudaTensor::from_slice(&[min], &[1], DType::F32))
    }

    /// 全要素の argmax（テンソル返却）
    pub fn argmax_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let idx = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as i64;
        Ok(CudaTensor::from_slice(&[idx], &[1], DType::I64))
    }

    /// 全要素の argmin（テンソル返却）
    pub fn argmin_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let idx = data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as i64;
        Ok(CudaTensor::from_slice(&[idx], &[1], DType::I64))
    }
}
