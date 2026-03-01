//! 特殊演算 — CUDA カーネルで GPU 上で完結

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_softmax_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_embedding_kernel(
        weight: *const f32,
        indices: *const i64,
        output: *mut f32,
        seq_len: i32,
        embed_dim: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    );
    fn launch_cross_entropy_kernel(
        logits: *const f32,
        targets: *const i64,
        losses: *mut f32,
        n: i32,
        c: i32,
        stream: cudaStream_t,
    );
    fn launch_tril_kernel(
        input: *const f32,
        output: *mut f32,
        rows: i32,
        cols: i32,
        batch: i32,
        diagonal: i32,
        stream: cudaStream_t,
    );
    fn launch_where_cond_kernel(
        cond: *const f32,
        x: *const f32,
        y: *const f32,
        output: *mut f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn launch_one_hot_kernel(
        indices: *const i64,
        output: *mut f32,
        batch: i32,
        classes: i32,
        stream: cudaStream_t,
    );
    fn launch_scatter_add_kernel(
        grad: *const f32,
        indices: *const i64,
        output: *mut f32,
        seq_len: i32,
        dim: i32,
        vocab: i32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// Softmax — GPU カーネル
    pub fn softmax_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        let axis_size = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_softmax_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                axis_size as i32,
                inner as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Embedding lookup — GPU カーネル
    pub fn embedding_impl(&self, indices: &CudaTensor) -> BackendResult<CudaTensor> {
        let weight_shape = self.shape();
        if weight_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(
                "embedding weight must be 2D".into(),
            ));
        }
        let vocab_size = weight_shape[0];
        let embed_dim = weight_shape[1];
        let seq_len = indices.elem_count();

        let mut out_shape = indices.shape().to_vec();
        out_shape.push(embed_dim);
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();

        unsafe {
            launch_embedding_kernel(
                self.buffer.ptr() as *const f32,
                indices.buffer.ptr() as *const i64,
                output.buffer.ptr() as *mut f32,
                seq_len as i32,
                embed_dim as i32,
                vocab_size as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Cross entropy loss — GPU カーネル
    pub fn cross_entropy_impl(&self, target: &CudaTensor) -> BackendResult<CudaTensor> {
        let logits_shape = self.shape();
        if logits_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(
                "cross_entropy logits must be 2D [N, C]".into(),
            ));
        }
        let n = logits_shape[0];
        let c = logits_shape[1];

        // per-sample loss を GPU で計算
        let losses = CudaTensor::uninit(&[n], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_cross_entropy_kernel(
                self.buffer.ptr() as *const f32,
                target.buffer.ptr() as *const i64,
                losses.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                stream,
            );
        }
        crate::stream::sync_stream();

        // mean を GPU で計算
        let total_loss = losses.sumall_impl()? / n as f32;
        Ok(CudaTensor::from_slice(&[total_loss], &[1], DType::F32))
    }

    /// 下三角行列 — GPU カーネル
    pub fn tril_impl(&self, diagonal: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(BackendError::ShapeMismatch("tril requires >= 2D".into()));
        }
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

        let output = CudaTensor::uninit(shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_tril_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                rows as i32,
                cols as i32,
                batch as i32,
                diagonal,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// index_select (CPU — Phase C の後で GPU 化)
    pub fn index_select_impl(
        &self,
        axis: usize,
        indices: &CudaTensor,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let idx_data: Vec<i64> = indices.to_vec();

        let mut out_shape = shape.clone();
        out_shape[axis] = idx_data.len();
        let out_count: usize = out_shape.iter().product();
        let ndim = shape.len();

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            let orig_coord = idx_data[coords[axis]] as usize;
            coords[axis] = orig_coord;
            let src_idx: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            result[out_idx] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, self.dtype()))
    }

    /// where_cond — GPU カーネル
    pub fn where_cond_impl(
        cond: &CudaTensor,
        x: &CudaTensor,
        y: &CudaTensor,
    ) -> BackendResult<CudaTensor> {
        let n = x.elem_count();
        let output = CudaTensor::uninit(x.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_where_cond_kernel(
                cond.buffer.ptr() as *const f32,
                x.buffer.ptr() as *const f32,
                y.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// repeat_interleave (CPU — 使用頻度低)
    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let data = self.to_vec::<f32>();
        let ndim = shape.len();

        let mut out_shape = shape.clone();
        out_shape[axis] *= repeats;
        let out_count: usize = out_shape.iter().product();

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut rem = out_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            coords[axis] /= repeats;
            let src_idx: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            result[out_idx] = data[src_idx];
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, self.dtype()))
    }

    /// one_hot — GPU カーネル
    pub fn one_hot_impl(&self, num_classes: usize) -> BackendResult<CudaTensor> {
        let batch = self.elem_count();
        let out_shape = vec![batch, num_classes];
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_one_hot_kernel(
                self.buffer.ptr() as *const i64,
                output.buffer.ptr() as *mut f32,
                batch as i32,
                num_classes as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// scatter_add — GPU カーネル
    pub fn scatter_add_impl(
        grad: &CudaTensor,
        indices: &CudaTensor,
        vocab_size: usize,
        embed_dim: usize,
    ) -> BackendResult<CudaTensor> {
        let seq_len = indices.elem_count();
        let out_shape = vec![vocab_size, embed_dim];
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_scatter_add_kernel(
                grad.buffer.ptr() as *const f32,
                indices.buffer.ptr() as *const i64,
                output.buffer.ptr() as *mut f32,
                seq_len as i32,
                embed_dim as i32,
                vocab_size as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
