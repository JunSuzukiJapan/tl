//! 二項演算（要素ごと）— 同一 shape は CUDA カーネル、broadcast は CPU フォールバック

use crate::cuda_sys::cudaStream_t;
use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

/// NumPy スタイルの broadcast shape 計算
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> BackendResult<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        let out_dim = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        };
        result[max_rank - 1 - i] = out_dim;
    }
    Ok(result)
}

/// broadcast 用: flat index をソーステンソルの index に変換
fn broadcast_index(flat_idx: usize, out_shape: &[usize], src_shape: &[usize]) -> usize {
    let rank = out_shape.len();
    let mut idx = flat_idx;
    let mut src_idx = 0;
    let mut src_stride = 1;

    for d in (0..rank).rev() {
        let out_dim = out_shape[d];
        let coord = idx % out_dim;
        idx /= out_dim;

        let src_dim = if d >= rank - src_shape.len() {
            src_shape[d - (rank - src_shape.len())]
        } else {
            1
        };

        if src_dim > 1 {
            src_idx += coord * src_stride;
        }
        src_stride *= src_dim;
    }
    src_idx
}

// CUDA カーネル FFI
type BinaryKernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, cudaStream_t);

extern "C" {
    fn launch_add_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_sub_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_mul_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_div_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_pow_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_rem_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_eq_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_ne_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_lt_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_le_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_gt_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
    fn launch_ge_kernel(a: *const f32, b: *const f32, y: *mut f32, n: i32, stream: cudaStream_t);
}

impl CudaTensor {
    /// 二項演算: 同一 shape → GPU カーネル、broadcast → CPU フォールバック
    fn binary_op_gpu<F: Fn(f32, f32) -> f32>(
        &self,
        other: &CudaTensor,
        kernel: BinaryKernelFn,
        cpu_op: F,
    ) -> BackendResult<CudaTensor> {
        if self.dtype() != other.dtype() {
            return Err(BackendError::TypeMismatch(format!(
                "DType mismatch: {:?} vs {:?}",
                self.dtype(),
                other.dtype()
            )));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape == other_shape {
            // 同一 shape → GPU カーネル (to_vec なし)
            let n = self.elem_count();
            let output = CudaTensor::uninit(self_shape, DType::F32);
            let stream = crate::stream::get_stream().raw();
            unsafe {
                kernel(
                    self.buffer.ptr() as *const f32,
                    other.buffer.ptr() as *const f32,
                    output.buffer.ptr() as *mut f32,
                    n as i32,
                    stream,
                );
            }
            crate::stream::sync_stream();
            Ok(output)
        } else {
            // broadcast → CPU フォールバック
            let out_shape = broadcast_shape(self_shape, other_shape)?;
            let out_count = out_shape.iter().product::<usize>();
            let a_data = self.to_vec::<f32>();
            let b_data = other.to_vec::<f32>();

            let result_data: Vec<f32> = (0..out_count)
                .map(|i| {
                    let a_idx = broadcast_index(i, &out_shape, self_shape);
                    let b_idx = broadcast_index(i, &out_shape, other_shape);
                    cpu_op(a_data[a_idx], b_data[b_idx])
                })
                .collect();

            Ok(CudaTensor::from_slice(
                &result_data,
                &out_shape,
                self.dtype(),
            ))
        }
    }

    /// 要素ごとの加算
    pub fn add_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_add_kernel, |a, b| a + b)
    }

    /// 要素ごとの減算
    pub fn sub_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_sub_kernel, |a, b| a - b)
    }

    /// 要素ごとの乗算
    pub fn mul_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_mul_kernel, |a, b| a * b)
    }

    /// 要素ごとの除算
    pub fn div_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_div_kernel, |a, b| a / b)
    }

    /// 要素ごとのべき乗
    pub fn pow_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_pow_kernel, |a, b| a.powf(b))
    }

    /// 要素ごとの剰余
    pub fn rem_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_rem_kernel, |a, b| a % b)
    }

    // ========== 比較演算 ==========

    pub fn eq_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_eq_kernel, |a, b| {
            if (a - b).abs() < 1e-6 {
                1.0
            } else {
                0.0
            }
        })
    }
    pub fn ne_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(other, launch_ne_kernel, |a, b| {
            if (a - b).abs() >= 1e-6 {
                1.0
            } else {
                0.0
            }
        })
    }
    pub fn lt_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(
            other,
            launch_lt_kernel,
            |a, b| if a < b { 1.0 } else { 0.0 },
        )
    }
    pub fn le_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(
            other,
            launch_le_kernel,
            |a, b| if a <= b { 1.0 } else { 0.0 },
        )
    }
    pub fn gt_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(
            other,
            launch_gt_kernel,
            |a, b| if a > b { 1.0 } else { 0.0 },
        )
    }
    pub fn ge_impl(&self, other: &CudaTensor) -> BackendResult<CudaTensor> {
        self.binary_op_gpu(
            other,
            launch_ge_kernel,
            |a, b| if a >= b { 1.0 } else { 0.0 },
        )
    }
}
