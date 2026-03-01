//! 活性化関数（要素ごと）— CUDA カーネルで GPU 上で完結
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

    /// ReLU: max(0, x)
    pub fn relu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_relu_kernel)
    }

    /// Sigmoid: 1 / (1 + exp(-x))
    pub fn sigmoid_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_sigmoid_kernel2)
    }

    /// Tanh
    pub fn tanh_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_tanh_kernel)
    }

    /// GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pub fn gelu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_gelu_kernel)
    }

    /// SiLU (Swish): x * sigmoid(x)
    pub fn silu_impl(&self) -> BackendResult<CudaTensor> {
        self.activation_kernel_op(launch_silu_kernel)
    }

    /// Sin (CPU フォールバック — 使用頻度低)
    pub fn sin_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.sin()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Cos (CPU フォールバック)
    pub fn cos_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.cos()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Tan (CPU フォールバック)
    pub fn tan_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.tan()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Floor (CPU フォールバック)
    pub fn floor_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.floor()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Ceil (CPU フォールバック)
    pub fn ceil_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.ceil()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }

    /// Round (CPU フォールバック)
    pub fn round_impl(&self) -> BackendResult<CudaTensor> {
        let data = self.to_vec::<f32>();
        let result: Vec<f32> = data.iter().map(|&x| x.round()).collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }
}
