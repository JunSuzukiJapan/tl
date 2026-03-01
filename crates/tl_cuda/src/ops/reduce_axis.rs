//! 軸付きリダクション演算 — sum/mean/max/min は GPU カーネル、argmax/argmin は CPU

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

extern "C" {
    fn launch_reduce_axis_sum_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_reduce_axis_max_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_reduce_axis_min_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
}

impl CudaTensor {
    /// 軸パラメータを正規化し (outer, axis_size, inner, out_shape) を返す
    fn resolve_axis(&self, axis: i32) -> BackendResult<(usize, usize, usize, Vec<usize>)> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range for ndim {}",
                axis, ndim
            )));
        }
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Ok((outer, axis_size, inner, out_shape))
    }

    /// 軸リダクションの GPU カーネル共通パターン
    fn reduce_axis_kernel_op(
        &self,
        axis: i32,
        launch: unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, cudaStream_t),
    ) -> BackendResult<CudaTensor> {
        let (outer, axis_size, inner, out_shape) = self.resolve_axis(axis)?;
        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch(
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

    /// 軸に沿った合計 — GPU カーネル
    pub fn sum_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_kernel_op(axis, launch_reduce_axis_sum_kernel)
    }

    /// 軸に沿った平均 — GPU カーネル
    pub fn mean_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let (_, axis_size, _, _) = self.resolve_axis(axis)?;
        let sum = self.sum_impl(axis)?;
        sum.div_scalar_impl(axis_size as f32)
    }

    /// 軸に沿った最大値 — GPU カーネル
    pub fn max_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_kernel_op(axis, launch_reduce_axis_max_kernel)
    }

    /// 軸に沿った最小値 — GPU カーネル
    pub fn min_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_kernel_op(axis, launch_reduce_axis_min_kernel)
    }

    /// 軸に沿った argmax（CPU — インデックス返却が必要なため）
    pub fn argmax_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range for ndim {}",
                axis, ndim
            )));
        }

        let data = self.to_vec::<f32>();
        let axis_size = shape[axis];
        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_count: usize = out_shape.iter().product();

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for k in 0..axis_size {
                let src_idx = self.reduce_src_index(out_idx, k, axis, &shape);
                if data[src_idx] > best_v {
                    best_v = data[src_idx];
                    best_i = k;
                }
            }
            result[out_idx] = best_i as f32;
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::F32))
    }

    /// 軸に沿った argmin（CPU）
    pub fn argmin_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(BackendError::ArgumentError(format!(
                "axis {} out of range for ndim {}",
                axis, ndim
            )));
        }

        let data = self.to_vec::<f32>();
        let axis_size = shape[axis];
        let mut out_shape = shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_count: usize = out_shape.iter().product();

        let mut result = vec![0.0f32; out_count];
        for out_idx in 0..out_count {
            let mut best_i = 0usize;
            let mut best_v = f32::INFINITY;
            for k in 0..axis_size {
                let src_idx = self.reduce_src_index(out_idx, k, axis, &shape);
                if data[src_idx] < best_v {
                    best_v = data[src_idx];
                    best_i = k;
                }
            }
            result[out_idx] = best_i as f32;
        }

        Ok(CudaTensor::from_slice(&result, &out_shape, DType::F32))
    }

    /// reduce 用: 出力 flat index + 軸位置 k → ソース flat index
    fn reduce_src_index(&self, out_flat: usize, k: usize, axis: usize, shape: &[usize]) -> usize {
        let ndim = shape.len();
        let mut out_shape = shape.to_vec();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            return k;
        }

        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = out_flat;
        for d in (0..out_shape.len()).rev() {
            out_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        let mut src_coords = Vec::with_capacity(ndim);
        let mut out_d = 0;
        for d in 0..ndim {
            if d == axis {
                src_coords.push(k);
            } else {
                src_coords.push(out_coords[out_d]);
                out_d += 1;
            }
        }

        let mut idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            idx += src_coords[d] * stride;
            stride *= shape[d];
        }
        idx
    }
}
