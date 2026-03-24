//! 特殊演算 — 全て GPU カーネルで完結 (to_vec ゼロ)

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
    fn launch_index_select_kernel(
        input: *const f32,
        indices: *const i64,
        output: *mut f32,
        outer: i32,
        inner: i32,
        old_dim: i32,
        n_idx: i32,
        stream: cudaStream_t,
    );
    fn launch_repeat_interleave_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        inner: i32,
        old_dim: i32,
        repeats: i32,
        stream: cudaStream_t,
    );
    // Phase A: masked_fill / fill_
    fn launch_masked_fill_kernel(
        x: *const f32,
        mask: *const f32,
        y: *mut f32,
        n: i32,
        value: f32,
        stream: cudaStream_t,
    );
    fn launch_fill_kernel(y: *mut f32, n: i32, value: f32, stream: cudaStream_t);
}

impl CudaTensor {
    /// F32 テンソルを I64 に変換するヘルパー
    /// .tl は全値を f32 で保持するため、i64 を期待するカーネルに渡す前に変換が必要
    fn ensure_i64(&self) -> Option<CudaTensor> {
        if self.dtype() == DType::F32 {
            let count = self.elem_count();
            let f32_bytes = count * 4;
            let mut f32_host = vec![0f32; count];
            unsafe {
                crate::cuda_sys::cudaMemcpy(
                    f32_host.as_mut_ptr() as *mut std::ffi::c_void,
                    self.buffer.ptr(),
                    f32_bytes,
                    crate::cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
            }
            let i64_host: Vec<i64> = f32_host.iter().map(|&v| v as i64).collect();
            let t = CudaTensor::uninit(&[count], DType::I64);
            let i64_bytes = count * 8;
            unsafe {
                crate::cuda_sys::cudaMemcpy(
                    t.buffer.ptr(),
                    i64_host.as_ptr() as *const std::ffi::c_void,
                    i64_bytes,
                    crate::cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
            }
            Some(t)
        } else {
            None
        }
    }
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
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

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

        // indices が F32 の場合、i64 に変換 (.tl は全値を f32 で保持)
        let i64_indices = if indices.dtype() == DType::F32 {
            // GPU → CPU → convert → GPU (seq_len は小さいので問題なし)
            let f32_bytes = seq_len * 4;
            let mut f32_host = vec![0f32; seq_len];
            unsafe {
                crate::cuda_sys::cudaMemcpy(
                    f32_host.as_mut_ptr() as *mut std::ffi::c_void,
                    indices.buffer.ptr(),
                    f32_bytes,
                    crate::cuda_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
            }
            let i64_host: Vec<i64> = f32_host.iter().map(|&v| v as i64).collect();
            let t = CudaTensor::uninit(&[seq_len], DType::I64);
            let i64_bytes = seq_len * 8;
            unsafe {
                crate::cuda_sys::cudaMemcpy(
                    t.buffer.ptr(),
                    i64_host.as_ptr() as *const std::ffi::c_void,
                    i64_bytes,
                    crate::cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
            }
            Some(t)
        } else {
            None
        };
        let actual_indices = i64_indices.as_ref().unwrap_or(indices);

        let mut out_shape = indices.shape().to_vec();
        out_shape.push(embed_dim);
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_embedding_kernel(
                self.buffer.ptr() as *const f32,
                actual_indices.buffer.ptr() as *const i64,
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

        let i64_targets = target.ensure_i64();
        let actual_targets = i64_targets.as_ref().unwrap_or(target);

        let losses = CudaTensor::uninit(&[n], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_cross_entropy_kernel(
                self.buffer.ptr() as *const f32,
                actual_targets.buffer.ptr() as *const i64,
                losses.buffer.ptr() as *mut f32,
                n as i32,
                c as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
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

    /// index_select — GPU カーネル
    pub fn index_select_impl(
        &self,
        axis: usize,
        indices: &CudaTensor,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let n_idx = indices.elem_count();

        let outer: usize = shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let old_dim = shape[axis];

        let mut out_shape = shape.clone();
        out_shape[axis] = n_idx;
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_index_select_kernel(
                self.buffer.ptr() as *const f32,
                indices.buffer.ptr() as *const i64,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                inner as i32,
                old_dim as i32,
                n_idx as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        let _ = ndim;
        Ok(output)
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

    /// repeat_interleave — GPU カーネル
    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();

        let outer: usize = shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let old_dim = shape[axis];

        let mut out_shape = shape.clone();
        out_shape[axis] *= repeats;
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_repeat_interleave_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                inner as i32,
                old_dim as i32,
                repeats as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// one_hot — GPU カーネル
    pub fn one_hot_impl(&self, num_classes: usize) -> BackendResult<CudaTensor> {
        let batch = self.elem_count();
        let i64_conv = self.ensure_i64();
        let actual = i64_conv.as_ref().unwrap_or(self);
        let out_shape = vec![batch, num_classes];
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_one_hot_kernel(
                actual.buffer.ptr() as *const i64,
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
        let i64_conv = indices.ensure_i64();
        let actual_indices = i64_conv.as_ref().unwrap_or(indices);
        let out_shape = vec![vocab_size, embed_dim];
        let output = CudaTensor::zeros(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_scatter_add_kernel(
                grad.buffer.ptr() as *const f32,
                actual_indices.buffer.ptr() as *const i64,
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

    // ========== Phase A: masked_fill / fill_ ==========

    /// masked_fill — GPU カーネル
    pub fn masked_fill_impl(&self, mask: &CudaTensor, value: f32) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_masked_fill_kernel(
                self.buffer.ptr() as *const f32,
                mask.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                value,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// fill_ (in-place fill) — GPU カーネル
    pub fn fill_impl(&self, value: f32) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(self.shape(), DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_fill_kernel(output.buffer.ptr() as *mut f32, n as i32, value, stream);
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
