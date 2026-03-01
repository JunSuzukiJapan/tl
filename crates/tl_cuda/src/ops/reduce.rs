//! リダクション演算 — 全て GPU 上で完結 (to_vec ゼロ)

use crate::cuda_sys::{cudaMemcpy, cudaMemcpyKind, cudaStream_t};
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_sum_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_max_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_min_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_argmax_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_argmin_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// 全要素の合計（スカラー返却）— GPU カーネル + cudaMemcpy (1要素)
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
        // 1要素だけ cudaMemcpy D→H
        let mut result = 0.0f32;
        unsafe {
            cudaMemcpy(
                &mut result as *mut f32 as *mut std::ffi::c_void,
                output.buffer.ptr(),
                std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
        }
        Ok(result)
    }

    /// 全要素の平均（スカラー返却）
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        let sum = self.sumall_impl()?;
        Ok(sum / self.elem_count() as f32)
    }

    /// 全要素の合計（テンソル返却: shape=[1]）
    pub fn sum_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
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
        Ok(output)
    }

    /// 全要素の平均（テンソル返却: shape=[1]）
    pub fn mean_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
        let sum_tensor = self.sum_all_tensor_impl()?;
        let len = self.elem_count() as f32;
        sum_tensor.div_scalar_impl(len)
    }

    /// 全要素の最大値 — GPU カーネル
    pub fn max_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
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

    /// 全要素の最小値 — GPU カーネル
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

    /// 全要素の argmax — GPU カーネル
    pub fn argmax_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_argmax_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の argmin — GPU カーネル
    pub fn argmin_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_argmin_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
