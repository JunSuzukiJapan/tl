//! 活性化関数（要素ごと）— 全て CUDA カーネルで GPU 上で完結
//!
//! exp_impl, log_impl, sqrt_impl は unary.rs に定義済み（カーネル化済み）

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_relu_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_sigmoid_kernel2(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_tanh_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_gelu_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_silu_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_sin_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_cos_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_tan_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_floor_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_ceil_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_round_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// GPU カーネルによる活性化関数の共通パターン
    fn activation_kernel_op(
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

    pub fn relu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_relu_kernel)
    }
    pub fn sigmoid_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_sigmoid_kernel2)
    }
    pub fn tanh_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_tanh_kernel)
    }
    pub fn gelu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_gelu_kernel)
    }
    pub fn silu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_silu_kernel)
    }
    pub fn sin_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_sin_kernel)
    }
    pub fn cos_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_cos_kernel)
    }
    pub fn tan_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_tan_kernel)
    }
    pub fn floor_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_floor_kernel)
    }
    pub fn ceil_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_ceil_kernel)
    }
    pub fn round_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_round_kernel)
    }
}
