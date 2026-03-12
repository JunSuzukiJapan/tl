//! リダクション演算 — 全て GPU 上で完結 (to_vec ゼロ)

use crate::cuda_sys::{cudaMemcpy, cudaMemcpyKind, cudaStream_t};
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::BackendResult;

extern "C" {
    fn launch_sum_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_max_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_min_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_argmax_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_argmin_all_kernel(x: *const f32, out: *mut f32, n: i32, stream: cudaStream_t);
    // Phase B: 新規 reduction カーネル
    fn launch_reduce_axis_var_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_reduce_axis_prod_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_cumsum_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        stream: cudaStream_t,
    );
    fn launch_reduce_axis_norm_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        p: f32,
        stream: cudaStream_t,
    );
    fn launch_topk_kernel(
        input: *const f32,
        output: *mut f32,
        outer: i32,
        axis_size: i32,
        inner: i32,
        k: i32,
        stream: cudaStream_t,
    );
    fn launch_sqrt_kernel(x: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// 全要素の合計（スカラー返却）— GPU カーネル + cudaMemcpy (1要素)
    pub fn sumall_impl(&self) -> BackendResult<f32> {
        let n = self.elem_count();
        let output = CudaTensor::zeros(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_sum_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        // 1要素だけ cudaMemcpy D→H
        let mut result = 0.0f32;
        unsafe {
            cudaMemcpy(
                &mut result as *mut f32 as *mut std::ffi::c_void,
                output.buffer.ptr(),
                std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
        }
        Ok(result)
    }

    /// 全要素の平均（スカラー返却）
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        let sum = self.sumall_impl()?;
        Ok(sum / self.elem_count() as f32)
    }

    /// 全要素の合計（テンソル返却: shape=[1]）
    pub fn sum_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::zeros(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_sum_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の平均（テンソル返却: shape=[1]）
    pub fn mean_all_tensor_impl(&self) -> BackendResult<CudaTensor> {
        let sum_tensor = self.sum_all_tensor_impl()?;
        let len = self.elem_count() as f32;
        sum_tensor.div_scalar_impl(len)
    }

    /// 全要素の最大値 — GPU カーネル
    pub fn max_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::from_slice(&[f32::NEG_INFINITY], &[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_max_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の最小値 — GPU カーネル
    pub fn min_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::from_slice(&[f32::INFINITY], &[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_min_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の argmax — GPU カーネル
    pub fn argmax_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_argmax_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 全要素の argmin — GPU カーネル
    pub fn argmin_all_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_argmin_all_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    // ========== Phase B: 新規 Reduction 操作 ==========

    /// 分散 (全要素) — GPU カーネル
    pub fn var_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_reduce_axis_var_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                1, // outer=1
                n as i32,
                1, // inner=1
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 標準偏差 (全要素)
    pub fn std_impl(&self) -> BackendResult<CudaTensor> {
        let var = self.var_impl()?;
        // sqrt_impl は activation.rs 経由で利用可能ではないかもしれないので、
        // unary.rs の sqrt を使う
        let n = var.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_sqrt_kernel(
                var.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                n as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 積 (全要素) — GPU カーネル
    pub fn prod_impl(&self) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_reduce_axis_prod_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                1,
                n as i32,
                1,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// 累積和 — GPU カーネル
    pub fn cumsum_impl(&self, dim: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let axis = if dim < rank { dim } else { rank - 1 };
        let axis_size = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        let output = CudaTensor::uninit(&shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_cumsum_kernel(
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

    /// Lp ノルム — GPU カーネル
    pub fn norm_impl(&self, p: f32) -> BackendResult<CudaTensor> {
        let n = self.elem_count();
        let output = CudaTensor::uninit(&[1], DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_reduce_axis_norm_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                1,
                n as i32,
                1,
                p,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }

    /// Top-k — GPU カーネル
    pub fn topk_impl(&self, k: usize) -> BackendResult<CudaTensor> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let axis_size = shape[rank - 1];
        let outer: usize = shape[..rank - 1].iter().product::<usize>().max(1);
        let inner = 1usize;
        let k_clamped = k.min(64).min(axis_size);

        let mut out_shape = shape[..rank - 1].to_vec();
        out_shape.push(k_clamped);

        let output = CudaTensor::uninit(&out_shape, DType::F32);
        let stream = crate::stream::get_stream().raw();
        unsafe {
            launch_topk_kernel(
                self.buffer.ptr() as *const f32,
                output.buffer.ptr() as *mut f32,
                outer as i32,
                axis_size as i32,
                inner as i32,
                k_clamped as i32,
                stream,
            );
        }
        crate::stream::sync_stream();
        Ok(output)
    }
}
