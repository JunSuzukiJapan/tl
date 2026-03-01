//! 行列積 — CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_matmul_kernel(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        k: i32,
        n: i32,
        batch: i32,
        b_batched: i32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// 行列積（2D × 2D → 2D）— GPU カーネル
    pub fn matmul_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(BackendError::ShapeMismatch(format!(
                "matmul requires at least 2D tensors, got {:?} and {:?}",
                a_shape, b_shape
            )));
        }

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        if k != k2 {
            return Err(BackendError::ShapeMismatch(format!(
                "matmul inner dims mismatch: {} vs {}",
                k, k2
            )));
        }

        let batch_size: usize = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };
        let b_batched = if b_shape.len() > 2 { 1 } else { 0 };

        let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();

        unsafe {
            launch_matmul_kernel(
                self.buffer.ptr() as *const f32,
                other.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                batch_size as i32,
                b_batched,
                stream,
            );
        }
        crate::stream::sync_stream();

        Ok(output)
    }
}
