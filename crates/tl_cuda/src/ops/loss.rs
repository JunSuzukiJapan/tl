//! 損失関数 — 全て CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_mse_loss_elem_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_l1_loss_elem_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_bce_loss_elem_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_nll_loss_elem_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_kl_div_loss_elem_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// 損失関数の共通パターン: element-wise カーネル → mean (sum_all / n)
    fn loss_impl(
        &self,
        other: &CudaTensor,
        launch: unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, cudaStream_t),
    ) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let elem_output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch(
                self.buffer.ptr() as *const f32,
                other.buffer.ptr() as *const f32,
                elem_output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        // mean = sum / n
        let sum = elem_output.sum_all_tensor_impl()?;
        sum.div_scalar_impl(n as f32)
    }

    pub fn mse_loss_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        self.loss_impl(target, launch_mse_loss_elem_kernel)
    }

    pub fn l1_loss_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        self.loss_impl(target, launch_l1_loss_elem_kernel)
    }

    pub fn bce_loss_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        self.loss_impl(target, launch_bce_loss_elem_kernel)
    }

    pub fn nll_loss_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        self.loss_impl(target, launch_nll_loss_elem_kernel)
    }

    pub fn kl_div_loss_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        self.loss_impl(target, launch_kl_div_loss_elem_kernel)
    }
}
