//! CPU 版 演算の勾配関数

use super::GradFn;
use crate::tensor::CpuTensor;
use tl_backend::GpuOps;

/// ブロードキャスト勾配集約:
/// grad の shape を target_shape に合わせるため、ブロードキャストで追加された次元に沿って sum する。
fn reduce_grad_for_broadcast(grad: &CpuTensor, target_shape: &[usize]) -> CpuTensor {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.shallow_clone();
    }
    
    let mut result = grad.shallow_clone();
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    
    // target_shape が短い場合、先頭の余分な次元を sum で潰す
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum_impl(0); // dim 0 に沿って sum (keep_dim=false で次元が減る)
        }
    }
    
    // 同じ次元数で、target が 1 の次元を sum + keep_dim
    let result_shape = result.shape().to_vec();
    let min_ndim = result_shape.len().min(target_shape.len());
    for d in 0..min_ndim {
        if target_shape[d] == 1 && result_shape[d] > 1 {
            result = result.sum_impl(d as i32);
            // sum_impl は keep_dim=false なので、shape を復元
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(d, 1);
            result = result.reshape_impl(&new_shape);
        }
    }
    
    // 最終的な shape を target_shape に合わせて reshape
    if result.shape() != target_shape {
        result = result.reshape_impl(target_shape);
    }
    
    result
}

/// 加算の勾配
pub struct AddBackward {
    pub a: *mut CpuTensor,
    pub b: *mut CpuTensor,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        vec![grad_output.shallow_clone(), grad_output.shallow_clone()]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a, self.b]
    }
}

/// 減算の勾配
pub struct SubBackward {
    pub a: *mut CpuTensor,
    pub b: *mut CpuTensor,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        vec![grad_output.shallow_clone(), grad_output.neg()]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a, self.b]
    }
}

/// 乗算の勾配
pub struct MulBackward {
    pub a: *mut CpuTensor,
    pub b: *mut CpuTensor,
    pub a_data: CpuTensor,
    pub b_data: CpuTensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.mul(&self.b_data), &self.a_shape);
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul(&self.a_data), &self.b_shape);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a, self.b]
    }
}

/// 除算の勾配
pub struct DivBackward {
    pub a: *mut CpuTensor,
    pub b: *mut CpuTensor,
    pub a_data: CpuTensor,
    pub b_data: CpuTensor,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let grad_a = grad_output.div(&self.b_data);
        let b_sq = self.b_data.mul(&self.b_data);
        let grad_b = grad_output.mul(&self.a_data).neg().div(&b_sq);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a, self.b]
    }
}

/// べき乗の勾配
pub struct PowBackward {
    pub a: *mut CpuTensor,
    pub a_data: CpuTensor,
    pub b_data: CpuTensor,
    pub output: CpuTensor,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let ones = CpuTensor::ones(self.b_data.shape(), self.b_data.dtype());
        let b_minus_1 = self.b_data.sub(&ones);
        let grad_a = grad_output.mul(&self.b_data).mul(&self.a_data.pow(&b_minus_1));
        vec![grad_a]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// sumall の勾配
pub struct SumallBackward {
    pub a: *mut CpuTensor,
    pub shape: Vec<usize>,
}

impl GradFn for SumallBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let grad_val = grad_output.to_vec::<f32>()[0];
        let count: usize = self.shape.iter().product();
        let grads = vec![grad_val; count];
        vec![CpuTensor::from_slice(&grads, &self.shape, grad_output.dtype())]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// ReLU の勾配
pub struct ReluBackward {
    pub a: *mut CpuTensor,
    pub a_data: CpuTensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let data: Vec<f32> = self.a_data.to_vec();
        let grad: Vec<f32> = grad_output.to_vec();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(a, g)| if *a > 0.0 { *g } else { 0.0 })
            .collect();
        vec![CpuTensor::from_slice(&result, self.a_data.shape(), self.a_data.dtype())]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// Softmax の勾配
pub struct SoftmaxBackward {
    pub a: *mut CpuTensor,
    pub output: CpuTensor,
    pub axis: i32,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let s: Vec<f32> = self.output.to_vec();
        let g: Vec<f32> = grad_output.to_vec();
        let shape = self.output.shape();
        if shape.len() == 2 && self.axis == 1 {
            let rows = shape[0];
            let cols = shape[1];
            let mut result = vec![0.0f32; s.len()];
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                let s_row = &s[start..end];
                let g_row = &g[start..end];
                let sg_sum: f32 = s_row.iter().zip(g_row.iter()).map(|(si, gi)| si * gi).sum();
                for j in 0..cols {
                    result[start + j] = s_row[j] * (g_row[j] - sg_sum);
                }
            }
            vec![CpuTensor::from_slice(&result, shape, self.output.dtype())]
        } else {
            let sg_sum: f32 = s.iter().zip(g.iter()).map(|(si, gi)| si * gi).sum();
            let result: Vec<f32> = s.iter().zip(g.iter())
                .map(|(si, gi)| si * (gi - sg_sum))
                .collect();
            vec![CpuTensor::from_slice(&result, shape, self.output.dtype())]
        }
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// matmul の勾配
pub struct MatmulBackward {
    pub a: *mut CpuTensor,
    pub b: *mut CpuTensor,
    pub a_data: CpuTensor,
    pub b_data: CpuTensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let grad_a = grad_output.matmul_impl(&self.b_data.transpose_impl(0, 1));
        let grad_b = self.a_data.transpose_impl(0, 1).matmul_impl(grad_output);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a, self.b]
    }
}

/// sigmoid の勾配
pub struct SigmoidBackward {
    pub a: *mut CpuTensor,
    pub output: CpuTensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let ones = CpuTensor::ones(self.output.shape(), self.output.dtype());
        let one_minus_s = ones.sub(&self.output);
        let grad = grad_output.mul(&self.output).mul(&one_minus_s);
        vec![grad]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// exp の勾配
pub struct ExpBackward {
    pub a: *mut CpuTensor,
    pub output: CpuTensor,
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        vec![grad_output.mul(&self.output)]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// log の勾配
pub struct LogBackward {
    pub a: *mut CpuTensor,
    pub a_data: CpuTensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        vec![grad_output.div(&self.a_data)]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

/// sum(dim) の勾配
pub struct SumDimBackward {
    pub a: *mut CpuTensor,
    pub input_shape: Vec<usize>,
    pub axis: i32,
}

impl GradFn for SumDimBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        let axis = if self.axis < 0 {
            (self.input_shape.len() as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        let grad_data: Vec<f32> = grad_output.to_vec();
        let numel: usize = self.input_shape.iter().product();
        let mut result = vec![0.0f32; numel];
        let outer: usize = self.input_shape[..axis].iter().product();
        let inner: usize = self.input_shape[axis + 1..].iter().product();
        let axis_size = self.input_shape[axis];
        for i in 0..outer {
            for k in 0..inner {
                let grad_val = grad_data[i * inner + k];
                for j in 0..axis_size {
                    result[i * axis_size * inner + j * inner + k] = grad_val;
                }
            }
        }
        vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.a]
    }
}

// unsafe impl Send/Sync for raw pointer types
unsafe impl Send for AddBackward {}
unsafe impl Sync for AddBackward {}
unsafe impl Send for SubBackward {}
unsafe impl Sync for SubBackward {}
unsafe impl Send for MulBackward {}
unsafe impl Sync for MulBackward {}
unsafe impl Send for DivBackward {}
unsafe impl Sync for DivBackward {}
unsafe impl Send for PowBackward {}
unsafe impl Sync for PowBackward {}
unsafe impl Send for SumallBackward {}
unsafe impl Sync for SumallBackward {}
unsafe impl Send for ReluBackward {}
unsafe impl Sync for ReluBackward {}
unsafe impl Send for SoftmaxBackward {}
unsafe impl Sync for SoftmaxBackward {}
unsafe impl Send for MatmulBackward {}
unsafe impl Sync for MatmulBackward {}
unsafe impl Send for SigmoidBackward {}
unsafe impl Sync for SigmoidBackward {}
unsafe impl Send for ExpBackward {}
unsafe impl Sync for ExpBackward {}
unsafe impl Send for LogBackward {}
unsafe impl Sync for LogBackward {}
unsafe impl Send for SumDimBackward {}
unsafe impl Sync for SumDimBackward {}

/// reshape の勾配
pub struct ReshapeBackward {
    pub input: *mut CpuTensor,
    pub input_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &CpuTensor) -> Vec<CpuTensor> {
        // grad を元のshapeに戻す
        vec![grad_output.reshape_impl(&self.input_shape)]
    }
    fn inputs(&self) -> Vec<*mut CpuTensor> {
        vec![self.input]
    }
}

unsafe impl Send for ReshapeBackward {}
unsafe impl Sync for ReshapeBackward {}
