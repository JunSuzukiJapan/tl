//! ニューラルネットワーク演算 — 全て CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_conv2d_kernel(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        n: i32,
        c_in: i32,
        h_in: i32,
        w_in: i32,
        c_out: i32,
        kh: i32,
        kw: i32,
        h_out: i32,
        w_out: i32,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        stream: cudaStream_t,
    );
    fn launch_batch_norm_kernel(
        input: *const f32,
        gamma: *const f32,
        beta: *const f32,
        mean: *const f32,
        var: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        spatial: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_layer_norm_kernel(
        input: *const f32,
        gamma: *const f32,
        beta: *const f32,
        output: *mut f32,
        outer: i32,
        norm_size: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_max_pool2d_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        h_out: i32,
        w_out: i32,
        kh: i32,
        kw: i32,
        stride_h: i32,
        stride_w: i32,
        stream: cudaStream_t,
    );
    fn launch_avg_pool2d_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        h_out: i32,
        w_out: i32,
        kh: i32,
        kw: i32,
        stride_h: i32,
        stride_w: i32,
        stream: cudaStream_t,
    );
    fn launch_dropout_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        p: f32,
        scale: f32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// Conv2D — GPU カーネル
    pub fn conv2d_impl(
        &self,
        weight: &CudaTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(BackendError::ShapeMismatch(
                "conv2d requires 4D input and weight".into(),
            ));
        }
        let (n, c_in, h_in, w_in) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (c_out, wc_in, kh, kw) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );
        if c_in != wc_in {
            return Err(BackendError::ShapeMismatch(format!(
                "conv2d channel mismatch: {} vs {}",
                c_in, wc_in
            )));
        }
        let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;

        let output = CudaTensor::uninit(&[n, c_out, h_out, w_out], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_conv2d_kernel(
                self.buffer.ptr() as *const f32,
                weight.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c_in as i32,
                h_in as i32,
                w_in as i32,
                c_out as i32,
                kh as i32,
                kw as i32,
                h_out as i32,
                w_out as i32,
                stride.0 as i32,
                stride.1 as i32,
                padding.0 as i32,
                padding.1 as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Batch Normalization — GPU カーネル
    pub fn batch_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        running_mean: &CudaTensor,
        running_var: &CudaTensor,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(BackendError::ShapeMismatch(
                "batch_norm requires >= 2D".into(),
            ));
        }
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);

        let output = CudaTensor::uninit(shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_batch_norm_kernel(
                self.buffer.ptr() as *const f32,
                gamma.buffer.ptr() as *const f32,
                beta.buffer.ptr() as *const f32,
                running_mean.buffer.ptr() as *const f32,
                running_var.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                spatial as i32,
                eps,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Layer Normalization — GPU カーネル
    pub fn layer_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        let norm_size = *shape.last().unwrap();
        let outer = self.elem_count() / norm_size;

        let output = CudaTensor::uninit(shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_layer_norm_kernel(
                self.buffer.ptr() as *const f32,
                gamma.buffer.ptr() as *const f32,
                beta.buffer.ptr() as *const f32,
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

    /// Max Pooling 2D — GPU カーネル
    pub fn max_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch("max_pool2d requires 4D".into()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h - kernel.0) / stride.0 + 1;
        let w_out = (w - kernel.1) / stride.1 + 1;

        let output = CudaTensor::uninit(&[n, c, h_out, w_out], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_max_pool2d_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                h as i32,
                w as i32,
                h_out as i32,
                w_out as i32,
                kernel.0 as i32,
                kernel.1 as i32,
                stride.0 as i32,
                stride.1 as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Average Pooling 2D — GPU カーネル
    pub fn avg_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch("avg_pool2d requires 4D".into()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h - kernel.0) / stride.0 + 1;
        let w_out = (w - kernel.1) / stride.1 + 1;

        let output = CudaTensor::uninit(&[n, c, h_out, w_out], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_avg_pool2d_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                h as i32,
                w as i32,
                h_out as i32,
                w_out as i32,
                kernel.0 as i32,
                kernel.1 as i32,
                stride.0 as i32,
                stride.1 as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Dropout — GPU カーネル
    pub fn dropout_impl(&self, p: f32, training: bool) -> BackendResult<CudaTensor> {
        if !training || p == 0.0 {
            return self.clone_data();
        }
        let n = self.elem_count();
        let scale = 1.0 / (1.0 - p);
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_dropout_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                p,
                scale,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
