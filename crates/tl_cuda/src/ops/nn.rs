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
    // Phase C: 新規 NN カーネル
    fn launch_group_norm_kernel(
        input: *const f32,
        gamma: *const f32,
        beta: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        spatial: i32,
        num_groups: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn launch_adaptive_avg_pool2d_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        h_in: i32,
        w_in: i32,
        h_out: i32,
        w_out: i32,
        stream: cudaStream_t,
    );
    fn launch_pad_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        old_dim: i32,
        new_dim: i32,
        pad_left: i32,
        pad_value: f32,
        stream: cudaStream_t,
    );
    fn launch_dropout2d_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        spatial: i32,
        p: f32,
        scale: f32,
        stream: cudaStream_t,
    );
    fn launch_conv1d_kernel(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        n: i32,
        c_in: i32,
        l_in: i32,
        c_out: i32,
        kl: i32,
        l_out: i32,
        stride: i32,
        pad: i32,
        stream: cudaStream_t,
    );
    fn launch_conv_transpose2d_kernel(
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
    fn launch_interpolate_kernel(
        input: *const f32,
        output: *mut f32,
        n: i32,
        c: i32,
        h_in: i32,
        w_in: i32,
        h_out: i32,
        w_out: i32,
        mode: i32,
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

    // ========== Phase C: 新規 NN 層 ==========

    /// Linear (matmul + bias)
    pub fn linear_impl(
        &self,
        weight: &CudaTensor,
        bias: Option<&CudaTensor>,
    ) -> BackendResult<CudaTensor> {
        let result = self.matmul_impl(weight)?;
        if let Some(b) = bias {
            result.add_impl(b)
        } else {
            Ok(result)
        }
    }

    /// Group Normalization — GPU カーネル
    pub fn group_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        num_groups: usize,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if shape.len() < 2 {
            return Err(BackendError::ShapeMismatch(
                "group_norm requires >= 2D".into(),
            ));
        }
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);
        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_group_norm_kernel(
                self.buffer.ptr() as *const f32,
                gamma.buffer.ptr() as *const f32,
                beta.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                spatial as i32,
                num_groups as i32,
                eps,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Instance Normalization — group_norm(groups=C)
    pub fn instance_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let c = self.shape()[1];
        self.group_norm_impl(gamma, beta, c, eps)
    }

    /// Adaptive Average Pooling 2D — GPU カーネル
    pub fn adaptive_avg_pool2d_impl(
        &self,
        output_size: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch(
                "adaptive_avg_pool2d requires 4D [N,C,H,W]".into(),
            ));
        }
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = output_size;
        let out_shape = vec![n, c, h_out, w_out];
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_adaptive_avg_pool2d_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                h_in as i32,
                w_in as i32,
                h_out as i32,
                w_out as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Padding — GPU カーネル
    pub fn pad_impl(&self, pad_sizes: &[usize], value: f32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let pad_left = if !pad_sizes.is_empty() {
            pad_sizes[0]
        } else {
            0
        };
        let pad_right = if pad_sizes.len() > 1 { pad_sizes[1] } else { 0 };
        let old_dim = shape[rank - 1];
        let new_dim = old_dim + pad_left + pad_right;
        let n: usize = shape[..rank - 1].iter().product::<usize>().max(1);
        let mut out_shape = shape[..rank - 1].to_vec();
        out_shape.push(new_dim);
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_pad_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                old_dim as i32,
                new_dim as i32,
                pad_left as i32,
                value,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Dropout2D — GPU カーネル
    pub fn dropout2d_impl(&self, p: f32, training: bool) -> BackendResult<CudaTensor> {
        if !training || p == 0.0 {
            return self.clone_data();
        }
        let shape = self.shape().to_vec();
        if shape.len() < 3 {
            return Err(BackendError::ShapeMismatch(
                "dropout2d requires >= 3D".into(),
            ));
        }
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let scale = 1.0 / (1.0 - p);
        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_dropout2d_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                spatial as i32,
                p,
                scale,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Conv1D — GPU カーネル
    pub fn conv1d_impl(
        &self,
        weight: &CudaTensor,
        stride: usize,
        padding: usize,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if shape.len() != 3 {
            return Err(BackendError::ShapeMismatch(
                "conv1d requires 3D [N,C_in,L]".into(),
            ));
        }
        let w_shape = weight.shape().to_vec();
        let (n, c_in, l_in) = (shape[0], shape[1], shape[2]);
        let (c_out, kl) = (w_shape[0], w_shape[2]);
        let l_out = (l_in + 2 * padding - kl) / stride + 1;
        let out_shape = vec![n, c_out, l_out];
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_conv1d_kernel(
                self.buffer.ptr() as *const f32,
                weight.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c_in as i32,
                l_in as i32,
                c_out as i32,
                kl as i32,
                l_out as i32,
                stride as i32,
                padding as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Conv Transpose 2D — GPU カーネル
    pub fn conv_transpose2d_impl(
        &self,
        weight: &CudaTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch(
                "conv_transpose2d requires 4D [N,C_in,H,W]".into(),
            ));
        }
        let w_shape = weight.shape().to_vec();
        let (n, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, kh, kw) = (w_shape[1], w_shape[2], w_shape[3]);
        let h_out = (h_in - 1) * stride.0 - 2 * padding.0 + kh;
        let w_out = (w_in - 1) * stride.1 - 2 * padding.1 + kw;
        let out_shape = vec![n, c_out, h_out, w_out];
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_conv_transpose2d_kernel(
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

    /// Interpolate (resize) — GPU カーネル
    pub fn interpolate_impl(
        &self,
        output_size: (usize, usize),
        mode: &str,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch(
                "interpolate requires 4D [N,C,H,W]".into(),
            ));
        }
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = output_size;
        let mode_int = if mode == "bilinear" { 1 } else { 0 };
        let out_shape = vec![n, c, h_out, w_out];
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_interpolate_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                h_in as i32,
                w_in as i32,
                h_out as i32,
                w_out as i32,
                mode_int,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
