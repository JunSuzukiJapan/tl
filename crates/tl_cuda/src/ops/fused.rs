//! 融合カーネル — 専用 CUDA カーネルで実装

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::error::BackendError;
use tl_backend::fused_ops::GpuFusedOps;

type Result<T> = std::result::Result<T, BackendError>;

extern "C" {
    fn launch_fused_silu_mul_kernel(
        x: *const f32,
        up: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_fused_rms_norm_kernel(
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        outer: i32,
        norm_size: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_fused_add_rms_norm_kernel(
        x: *const f32,
        residual: *const f32,
        weight: *const f32,
        y: *mut f32,
        outer: i32,
        norm_size: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_apply_rope_kernel(
        input: *const f32,
        cos_table: *const f32,
        sin_table: *const f32,
        output: *mut f32,
        num_tokens: i32,
        heads_per_token: i32,
        dim: i32,
        half_dim: i32,
        cos_dim: i32,
        cos_seq_len: i32,
        stream: cudaStream_t,
    );
    fn launch_fused_add_relu_kernel(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_fused_bias_gelu_kernel(
        x: *const f32,
        bias: *const f32,
        y: *mut f32,
        outer: i32,
        dim: i32,
        stream: cudaStream_t,
    );
}

impl GpuFusedOps for CudaTensor {
    fn fused_silu_mul(&self, up: &Self) -> Result<Self> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fused_silu_mul_kernel(
                self.buffer.ptr() as *const f32,
                up.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    fn fused_rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape().to_vec();
        let norm_size = *shape.last().unwrap();
        let outer = self.elem_count() / norm_size;
        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fused_rms_norm_kernel(
                self.buffer.ptr() as *const f32,
                weight.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                norm_size as i32,
                eps,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    fn fused_add_rms_norm(&self, residual: &Self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape().to_vec();
        let norm_size = *shape.last().unwrap();
        let outer = self.elem_count() / norm_size;
        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fused_add_rms_norm_kernel(
                self.buffer.ptr() as *const f32,
                residual.buffer.ptr() as *const f32,
                weight.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                norm_size as i32,
                eps,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    fn fused_rotary_emb(&self, freqs: &Self, head_dim: usize) -> Result<Self> {
        // freqs は cos/sin をインターリーブしている想定
        // 既存の apply_rope_kernel を活用: freqs を cos/sin に分割
        let shape = self.shape().to_vec();
        let dim = head_dim;
        let half_dim = dim / 2;

        let freqs_shape = freqs.shape();
        let cos_dim = freqs_shape[freqs_shape.len() - 1];
        let cos_seq_len = if freqs_shape.len() >= 2 {
            freqs_shape[0]
        } else {
            1
        };

        let (num_tokens, heads_per_token) = if shape.len() == 4 {
            (shape[1], shape[0] * shape[2])
        } else if shape.len() == 3 {
            (shape[0], shape[1])
        } else if shape.len() == 2 {
            (shape[0], 1usize)
        } else {
            (1, self.elem_count() / dim)
        };

        let output = CudaTensor::uninit(&shape, DType::F32);
        unsafe {
            crate::cuda_sys::cudaMemcpy(
                output.buffer.ptr(),
                self.buffer.ptr(),
                self.elem_count() * std::mem::size_of::<f32>(),
                crate::cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
        }

        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_apply_rope_kernel(
                self.buffer.ptr() as *const f32,
                freqs.buffer.ptr() as *const f32,
                freqs.buffer.ptr() as *const f32, // same for cos/sin
                output.buffer.ptr() as *mut f32,
                num_tokens as i32,
                heads_per_token as i32,
                dim as i32,
                half_dim as i32,
                cos_dim as i32,
                cos_seq_len as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    fn fused_add_relu(&self, other: &Self) -> Result<Self> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fused_add_relu_kernel(
                self.buffer.ptr() as *const f32,
                other.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    fn fused_bias_gelu(&self, bias: &Self) -> Result<Self> {
        let shape = self.shape().to_vec();
        let dim = *shape.last().unwrap();
        let outer = self.elem_count() / dim;
        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fused_bias_gelu_kernel(
                self.buffer.ptr() as *const f32,
                bias.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                dim as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
