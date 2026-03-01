//! 形状操作 — 全て GPU 上で完結 (to_vec ゼロ)

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_transpose_2d_kernel(
        input: *const f32,
        output: *mut f32,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    );
    fn launch_narrow_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        inner: i32,
        old_dim: i32,
        new_dim: i32,
        start: i32,
        stream: cudaStream_t,
    );
    fn launch_cat_kernel(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        outer: i32,
        inner: i32,
        a_dim: i32,
        b_dim: i32,
        stream: cudaStream_t,
    );
    fn launch_broadcast_to_kernel(
        input: *const f32,
        output: *mut f32,
        target_shape: *const i32,
        src_shape: *const i32,
        ndim: i32,
        total: i32,
        stream: cudaStream_t,
    );
    fn launch_transpose_nd_kernel(
        input: *const f32,
        output: *mut f32,
        old_shape: *const i32,
        old_strides: *const i32,
        new_strides: *const i32,
        ndim: i32,
        total: i32,
        dim0: i32,
        dim1: i32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// リシェイプ — GPU buffer 共有（ゼロコピー）
    pub fn reshape_impl(&self, new_shape: &[usize]) -> BackendResult<CudaTensor> {
        let old_count = self.elem_count();
        let new_count: usize = new_shape.iter().product();
        if old_count != new_count {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot reshape from {:?} ({}) to {:?} ({})",
                self.shape(),
                old_count,
                new_shape,
                new_count
            )));
        }
        Ok(self.view_with_shape(new_shape))
    }

    /// squeeze — GPU buffer 共有（ゼロコピー）
    pub fn squeeze_impl(&self, dim: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        if dim >= shape.len() || shape[dim] != 1 {
            return self.clone_data();
        }
        let mut new_shape = shape;
        new_shape.remove(dim);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        Ok(self.view_with_shape(&new_shape))
    }

    /// unsqueeze — GPU buffer 共有（ゼロコピー）
    pub fn unsqueeze_impl(&self, dim: usize) -> BackendResult<CudaTensor> {
        let mut new_shape = self.shape().to_vec();
        if dim > new_shape.len() {
            return Err(BackendError::ArgumentError(format!(
                "unsqueeze dim {} out of range for ndim {}",
                dim,
                new_shape.len()
            )));
        }
        new_shape.insert(dim, 1);
        Ok(self.view_with_shape(&new_shape))
    }

    /// contiguous — GPU buffer clone
    pub fn contiguous_impl(&self) -> BackendResult<CudaTensor> {
        self.clone_data()
    }

    /// 転置 — GPU カーネル（2D 最適化 + N-D 汎用）
    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "transpose dims ({}, {}) out of range for ndim {}",
                dim0, dim1, ndim
            )));
        }
        if dim0 == dim1 {
            return self.clone_data();
        }

        let mut new_shape = shape.clone();
        new_shape.swap(dim0, dim1);

        // 2D の場合は最適化カーネル
        if ndim == 2 {
            let output = CudaTensor::uninit(&new_shape, DType::F32);
            let stream = crate::stream::get_stream().raw();
            unsafe {
                launch_transpose_2d_kernel(
                    self.buffer.ptr() as *const f32,
                    output.buffer.ptr() as *mut f32,
                    shape[0] as i32,
                    shape[1] as i32,
                    stream,
                );
            }
            crate::stream::sync_stream();
            return Ok(output);
        }

        // N-D 汎用カーネル
        let total = self.elem_count();

        // strides 計算
        let mut old_strides = vec![1i32; ndim];
        for d in (0..ndim - 1).rev() {
            old_strides[d] = old_strides[d + 1] * shape[d + 1] as i32;
        }
        let mut new_strides = vec![1i32; ndim];
        for d in (0..ndim - 1).rev() {
            new_strides[d] = new_strides[d + 1] * new_shape[d + 1] as i32;
        }
        let old_shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();

        // GPU にアップロード
        let shape_gpu = CudaTensor::from_slice(
            unsafe { std::slice::from_raw_parts(old_shape_i32.as_ptr() as *const f32, ndim) },
            &[ndim],
            DType::F32,
        );
        let old_strides_gpu = CudaTensor::from_slice(
            unsafe { std::slice::from_raw_parts(old_strides.as_ptr() as *const f32, ndim) },
            &[ndim],
            DType::F32,
        );
        let new_strides_gpu = CudaTensor::from_slice(
            unsafe { std::slice::from_raw_parts(new_strides.as_ptr() as *const f32, ndim) },
            &[ndim],
            DType::F32,
        );

        let output = CudaTensor::uninit(&new_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_transpose_nd_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                shape_gpu.buffer.ptr() as *const i32,
                old_strides_gpu.buffer.ptr() as *const i32,
                new_strides_gpu.buffer.ptr() as *const i32,
                ndim as i32,
                total as i32,
                dim0 as i32,
                dim1 as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// narrow — GPU カーネル
    pub fn narrow_impl(&self, dim: usize, start: usize, len: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if dim >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "narrow dim {} >= ndim {}",
                dim, ndim
            )));
        }
        if start + len > shape[dim] {
            return Err(BackendError::ArgumentError(format!(
                "narrow: start {} + len {} > dim size {}",
                start, len, shape[dim]
            )));
        }

        let outer: usize = shape[..dim].iter().product::<usize>().max(1);
        let inner: usize = shape[dim + 1..].iter().product::<usize>().max(1);
        let old_dim = shape[dim];

        let mut new_shape = shape.clone();
        new_shape[dim] = len;
        let output = CudaTensor::uninit(&new_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_narrow_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                inner as i32,
                old_dim as i32,
                len as i32,
                start as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// slice（narrow の別名）
    pub fn slice_impl(&self, dim: usize, start: usize, len: usize) -> BackendResult<CudaTensor> {
        self.narrow_impl(dim, start, len)
    }

    /// cat — GPU カーネル
    pub fn cat_impl(&self, other: &CudaTensor, dim: usize) -> BackendResult<CudaTensor> {
        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        if a_shape.len() != b_shape.len() {
            return Err(BackendError::ShapeMismatch(format!(
                "cat: rank mismatch {:?} vs {:?}",
                a_shape, b_shape
            )));
        }
        for (i, (&a, &b)) in a_shape.iter().zip(b_shape.iter()).enumerate() {
            if i != dim && a != b {
                return Err(BackendError::ShapeMismatch(format!(
                    "cat: dim {} mismatch: {} vs {}",
                    i, a, b
                )));
            }
        }

        let outer: usize = a_shape[..dim].iter().product::<usize>().max(1);
        let inner: usize = a_shape[dim + 1..].iter().product::<usize>().max(1);
        let a_dim = a_shape[dim];
        let b_dim = b_shape[dim];

        let mut out_shape = a_shape.clone();
        out_shape[dim] = a_dim + b_dim;
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_cat_kernel(
                self.buffer.ptr() as *const f32,
                other.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                inner as i32,
                a_dim as i32,
                b_dim as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// broadcast_to — GPU カーネル
    pub fn broadcast_to_impl(&self, target_shape: &[usize]) -> BackendResult<CudaTensor> {
        let src_shape = self.shape();
        let ndim = target_shape.len();
        let out_count: usize = target_shape.iter().product();

        let mut padded_src = vec![1i32; ndim];
        let offset = ndim - src_shape.len();
        for (i, &s) in src_shape.iter().enumerate() {
            padded_src[offset + i] = s as i32;
        }
        let target_i32: Vec<i32> = target_shape.iter().map(|&s| s as i32).collect();

        let target_gpu = CudaTensor::from_slice(
            unsafe { std::slice::from_raw_parts(target_i32.as_ptr() as *const f32, ndim) },
            &[ndim],
            DType::F32,
        );
        let src_gpu = CudaTensor::from_slice(
            unsafe { std::slice::from_raw_parts(padded_src.as_ptr() as *const f32, ndim) },
            &[ndim],
            DType::F32,
        );

        let output = CudaTensor::uninit(target_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_broadcast_to_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                target_gpu.buffer.ptr() as *const i32,
                src_gpu.buffer.ptr() as *const i32,
                ndim as i32,
                out_count as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
