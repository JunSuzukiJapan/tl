//! 軸付きリダクション演算 — 全て GPU カーネルで完結 (to_vec ゼロ)

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
    fn launch_argmax_axis_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_argmin_axis_kernel(
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
        let outer: usize = shape[..axis].iter().product::<usize>().max(1);
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

    /// 軸に沿った argmax — GPU カーネル
    pub fn argmax_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_kernel_op(axis, launch_argmax_axis_kernel)
    }

    /// 軸に沿った argmin — GPU カーネル
    pub fn argmin_impl(&self, axis: i32) -> BackendResult<CudaTensor> {
        self.reduce_axis_kernel_op(axis, launch_argmin_axis_kernel)
    }
}
