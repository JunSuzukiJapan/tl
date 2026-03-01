//! リダクション演算 — 全て CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_sum_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_max_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_min_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// 全要素の合計（スカラー返却）— GPU カーネル
    pub fn sumall_impl(&self) -> BackendResult<f32> {
        let n = self.elem_count();
        let output = CudaTensor::zeros(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_sum_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        let result = output.to_vec::<f32>();
        Ok(result[0])
    }

    /// 全要素の平均（スカラー返却）— GPU カーネル
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        let sum = self.sumall_impl()?;
        let len = self.elem_count() as f32;
        Ok(sum / len)
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

    /// 全要素の最大値（GPU カーネル）
    pub fn max_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        // -FLT_MAX で初期化
        let output = CudaTensor::from_slice(&[f32::NEG_INFINITY], &[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_max_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の最小値（GPU カーネル）
    pub fn min_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::from_slice(&[f32::INFINITY], &[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_min_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の argmax（CPU — GPU argmax は複雑なため）
    pub fn argmax_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let idx = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as f32;
        Ok(CudaTensor::from_slice(&[idx], &[1], DType::F32))
    }

    /// 全要素の argmin（CPU — GPU argmin は複雑なため）
    pub fn argmin_all_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let idx = data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as f32;
        Ok(CudaTensor::from_slice(&[idx], &[1], DType::F32))
    }
}
