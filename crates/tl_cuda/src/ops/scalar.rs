//! スカラー演算 — CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_add_scalar_kernel(x: *const f32, y: *mut f32, n: i32, s: f32, stream: cudaStream_t);
    fn launch_mul_scalar_kernel(x: *const f32, y: *mut f32, n: i32, s: f32, stream: cudaStream_t);
    fn launch_div_scalar_kernel(x: *const f32, y: *mut f32, n: i32, s: f32, stream: cudaStream_t);
    fn launch_pow_scalar_kernel(x: *const f32, y: *mut f32, n: i32, s: f32, stream: cudaStream_t);
    fn launch_clamp_kernel(
        x: *const f32,
        y: *mut f32,
        n: i32,
        lo: f32,
        hi: f32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// GPU カーネルによるスカラー演算の共通パターン
    fn scalar_kernel_op(
        &self,
        launch: unsafe extern "C" fn(*const f32, *mut f32, i32, f32, cudaStream_t),
        scalar: f32,
    ) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                scalar,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// スカラー加算
    pub fn add_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.scalar_kernel_op(launch_add_scalar_kernel, scalar)
    }

    /// スカラー乗算
    pub fn mul_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.scalar_kernel_op(launch_mul_scalar_kernel, scalar)
    }

    /// スカラー除算
    pub fn div_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.scalar_kernel_op(launch_div_scalar_kernel, scalar)
    }

    /// スカラーべき乗
    pub fn pow_scalar_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.scalar_kernel_op(launch_pow_scalar_kernel, scalar)
    }

    /// スケール（スカラー乗算の別名）
    pub fn scale_impl(&self, scalar: f32) -> BackendResult<CudaTensor> {
        self.mul_scalar_impl(scalar)
    }

    /// クランプ
    pub fn clamp_impl(&self, min: f32, max: f32) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_clamp_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                min,
                max,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
