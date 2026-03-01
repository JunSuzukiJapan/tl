//! 単項演算（要素ごと）— CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_neg_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_abs_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_exp_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_log_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_sqrt_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// GPU カーネルによる単項演算の共通パターン
    fn unary_kernel_op(
        &self,
        launch: unsafe extern "C" fn(*const f32, *mut f32, i32, cudaStream_t),
    ) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 負数
    pub fn neg_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_kernel_op(launch_neg_kernel)
    }

    /// 絶対値
    pub fn abs_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_kernel_op(launch_abs_kernel)
    }

    /// 指数関数
    pub fn exp_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_kernel_op(launch_exp_kernel)
    }

    /// 対数関数
    pub fn log_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_kernel_op(launch_log_kernel)
    }

    /// 平方根
    pub fn sqrt_impl(&self) -> BackendResult<CudaTensor> {
        self.unary_kernel_op(launch_sqrt_kernel)
    }
}
