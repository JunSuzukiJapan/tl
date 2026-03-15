//! CPU 版 演算の勾配関数

use super::GradFn;
use crate::scalar::TensorScalar;
use crate::tensor::{CpuTensor, TensorRef};


/// ブロードキャスト勾配集約:
/// grad の shape を target_shape に合わせるため、ブロードキャストで追加された次元に沿って sum する。
fn reduce_grad_for_broadcast<T: TensorScalar>(grad: &CpuTensor<T>, target_shape: &[usize]) -> CpuTensor<T> {
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
            result = result.sum_impl(0).expect("Autograd sum failed (reduce_grad)"); // dim 0 に沿って sum (keep_dim=false で次元が減る)
        }
    }
    
    // 同じ次元数で、target が 1 の次元を sum + keep_dim
    let result_shape = result.shape().to_vec();
    let min_ndim = result_shape.len().min(target_shape.len());
    for d in 0..min_ndim {
        if target_shape[d] == 1 && result_shape[d] > 1 {
            result = result.sum_impl(d as i32).expect("Autograd sum failed (reduce_grad)");
            // sum_impl は keep_dim=false なので、shape を復元
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(d, 1);
            result = result.reshape_impl(&new_shape).expect("Autograd reshape failed (reduce_grad)");
        }
    }
    
    // 最終的な shape を target_shape に合わせて reshape
    if result.shape() != target_shape {
        result = result.reshape_impl(target_shape).expect("Autograd reshape failed (reduce_grad broadcast)");
    }
    
    result
}

/// 加算の勾配
pub struct AddBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape);
        let grad_b = reduce_grad_for_broadcast(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 減算の勾配
pub struct SubBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for SubBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape);
        let grad_b = reduce_grad_for_broadcast(&grad_output.neg_impl().expect("Autograd neg failed"), &self.b_shape);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 乗算の勾配
pub struct MulBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_data: CpuTensor<T>,
    pub b_data: CpuTensor<T>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for MulBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.b_data).expect("Autograd mul failed"), &self.a_shape);
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.a_data).expect("Autograd mul failed"), &self.b_shape);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 除算の勾配
pub struct DivBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_data: CpuTensor<T>,
    pub b_data: CpuTensor<T>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for DivBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.div_impl(&self.b_data).expect("Autograd div failed"), &self.a_shape);
        let b_sq = self.b_data.mul_impl(&self.b_data).expect("Autograd mul failed");
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.a_data).expect("Autograd mul failed").neg_impl().expect("Autograd neg failed").div_impl(&b_sq).expect("Autograd div failed"), &self.b_shape);
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// べき乗の勾配
pub struct PowBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub a_data: CpuTensor<T>,
    pub b_data: CpuTensor<T>,
    pub output: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for PowBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let ones = CpuTensor::<T>::ones(self.b_data.shape(), self.b_data.dtype());
        let b_minus_1 = self.b_data.sub_impl(&ones).expect("Autograd sub failed");
        let grad_a = grad_output.mul_impl(&self.b_data).expect("Autograd mul failed").mul_impl(&self.a_data.pow_impl(&b_minus_1).expect("Autograd pow failed")).expect("Autograd mul failed");
        vec![grad_a]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// sumall の勾配
pub struct SumallBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for SumallBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_val = grad_output.to_vec_t()[0];
        let count: usize = self.shape.iter().product();
        let grads = vec![grad_val; count];
        vec![CpuTensor::from_slice(&grads, &self.shape, grad_output.dtype())]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// ReLU の勾配
pub struct ReluBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub a_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for ReluBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let data = self.a_data.to_vec_t();
        let grad = grad_output.to_vec_t();
        let result: Vec<T> = data.iter().zip(grad.iter())
            .map(|(a, g)| if *a > T::zero() { *g } else { T::zero() })
            .collect();
        vec![CpuTensor::from_slice(&result, self.a_data.shape(), self.a_data.dtype())]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// Softmax の勾配
pub struct SoftmaxBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub output: CpuTensor<T>,
    pub axis: i32,
}

impl<T: TensorScalar> GradFn<T> for SoftmaxBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let s = self.output.to_vec_t();
        let g = grad_output.to_vec_t();
        let shape = self.output.shape();
        if shape.len() == 2 && self.axis == 1 {
            let rows = shape[0];
            let cols = shape[1];
            let mut result = vec![T::zero(); s.len()];
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                let s_row = &s[start..end];
                let g_row = &g[start..end];
                let mut sg_sum = T::zero();
                for (si, gi) in s_row.iter().zip(g_row.iter()) {
                    sg_sum = sg_sum + *si * *gi;
                }
                for j in 0..cols {
                    result[start + j] = s_row[j] * (g_row[j] - sg_sum);
                }
            }
            vec![CpuTensor::from_slice(&result, shape, self.output.dtype())]
        } else {
            let mut sg_sum = T::zero();
            for (si, gi) in s.iter().zip(g.iter()) {
                sg_sum = sg_sum + *si * *gi;
            }
            let result: Vec<T> = s.iter().zip(g.iter())
                .map(|(si, gi)| *si * (*gi - sg_sum))
                .collect();
            vec![CpuTensor::from_slice(&result, shape, self.output.dtype())]
        }
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// matmul の勾配
pub struct MatmulBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_data: CpuTensor<T>,
    pub b_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for MatmulBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let grad_a = grad_output.matmul_impl(&self.b_data.transpose_impl(0, 1).expect("Autograd transpose failed")).expect("Autograd matmul failed");
        let grad_b = self.a_data.transpose_impl(0, 1).expect("Autograd transpose failed").matmul_impl(grad_output).expect("Autograd matmul failed");
        vec![grad_a, grad_b]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// sigmoid の勾配
pub struct SigmoidBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub output: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for SigmoidBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let ones = CpuTensor::<T>::ones(self.output.shape(), self.output.dtype());
        let one_minus_s = ones.sub_impl(&self.output).expect("Autograd sub failed");
        let grad = grad_output.mul_impl(&self.output).expect("Autograd mul failed").mul_impl(&one_minus_s).expect("Autograd mul failed");
        vec![grad]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// exp の勾配
pub struct ExpBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub output: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for ExpBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.mul_impl(&self.output).expect("Autograd mul failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// log の勾配
pub struct LogBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub a_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for LogBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.div_impl(&self.a_data).expect("Autograd div failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// sum(dim) の勾配
pub struct SumDimBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub input_shape: Vec<usize>,
    pub axis: i32,
}

impl<T: TensorScalar> GradFn<T> for SumDimBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        let ndim = self.input_shape.len();
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        if ndim == 0 || axis >= ndim {
            let grad_val = grad_output.to_vec_t().first().copied().unwrap_or(T::zero());
            let numel: usize = self.input_shape.iter().product::<usize>().max(1);
            let result = vec![grad_val; numel];
            return vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())];
        }
        let grad_data = grad_output.to_vec_t();
        let numel: usize = self.input_shape.iter().product();
        let mut result = vec![T::zero(); numel];
        let outer: usize = self.input_shape[..axis].iter().product();
        let inner: usize = self.input_shape[axis + 1..].iter().product();
        let axis_size = self.input_shape[axis];
        for i in 0..outer {
            for k in 0..inner {
                let gi = i * inner + k;
                let grad_val = if gi < grad_data.len() { grad_data[gi] } else { T::zero() };
                for j in 0..axis_size {
                    result[i * axis_size * inner + j * inner + k] = grad_val;
                }
            }
        }
        vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// reshape の勾配
pub struct ReshapeBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for ReshapeBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.reshape_impl(&self.input_shape).expect("Autograd reshape failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// スカラ加算の勾配 (t + s) -> grad_t = grad_output
pub struct AddScalarBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
}

impl<T: TensorScalar> GradFn<T> for AddScalarBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.shallow_clone()]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// スカラ減算の勾配 (t - s) -> grad_t = grad_output
pub struct SubScalarBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
}

impl<T: TensorScalar> GradFn<T> for SubScalarBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.shallow_clone()]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// スカラ乗算の勾配 (t * s) -> grad_t = grad_output * s
pub struct MulScalarBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub s: T,
}

impl<T: TensorScalar> GradFn<T> for MulScalarBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.mul_scalar_impl(self.s).expect("Autograd mul_scalar failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// スカラ除算の勾配 (t / s) -> grad_t = grad_output / s
pub struct DivScalarBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub s: T,
}

impl<T: TensorScalar> GradFn<T> for DivScalarBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.div_scalar_impl(self.s).expect("Autograd div_scalar failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// 符号反転の勾配 (-t) -> grad_t = -grad_output
pub struct NegBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
}

impl<T: TensorScalar> GradFn<T> for NegBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> Vec<CpuTensor<T>> {
        vec![grad_output.neg_impl().expect("Autograd neg failed")]
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

// UnsafeCell<CpuTensor<T>> を含む TensorRef<T> を保持する構造体に対して
// Send + Sync を手動実装。CpuTensor<T> 自体が unsafe impl Send + Sync を持つため安全。
unsafe impl<T: TensorScalar> Send for AddBackward<T> {}
unsafe impl<T: TensorScalar> Sync for AddBackward<T> {}
unsafe impl<T: TensorScalar> Send for SubBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SubBackward<T> {}
unsafe impl<T: TensorScalar> Send for MulBackward<T> {}
unsafe impl<T: TensorScalar> Sync for MulBackward<T> {}
unsafe impl<T: TensorScalar> Send for DivBackward<T> {}
unsafe impl<T: TensorScalar> Sync for DivBackward<T> {}
unsafe impl<T: TensorScalar> Send for PowBackward<T> {}
unsafe impl<T: TensorScalar> Sync for PowBackward<T> {}
unsafe impl<T: TensorScalar> Send for SumallBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SumallBackward<T> {}
unsafe impl<T: TensorScalar> Send for ReluBackward<T> {}
unsafe impl<T: TensorScalar> Sync for ReluBackward<T> {}
unsafe impl<T: TensorScalar> Send for SoftmaxBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SoftmaxBackward<T> {}
unsafe impl<T: TensorScalar> Send for MatmulBackward<T> {}
unsafe impl<T: TensorScalar> Sync for MatmulBackward<T> {}
unsafe impl<T: TensorScalar> Send for SigmoidBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SigmoidBackward<T> {}
unsafe impl<T: TensorScalar> Send for ExpBackward<T> {}
unsafe impl<T: TensorScalar> Sync for ExpBackward<T> {}
unsafe impl<T: TensorScalar> Send for LogBackward<T> {}
unsafe impl<T: TensorScalar> Sync for LogBackward<T> {}
unsafe impl<T: TensorScalar> Send for SumDimBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SumDimBackward<T> {}
unsafe impl<T: TensorScalar> Send for ReshapeBackward<T> {}
unsafe impl<T: TensorScalar> Sync for ReshapeBackward<T> {}
unsafe impl<T: TensorScalar> Send for AddScalarBackward<T> {}
unsafe impl<T: TensorScalar> Sync for AddScalarBackward<T> {}
unsafe impl<T: TensorScalar> Send for SubScalarBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SubScalarBackward<T> {}
unsafe impl<T: TensorScalar> Send for MulScalarBackward<T> {}
unsafe impl<T: TensorScalar> Sync for MulScalarBackward<T> {}
unsafe impl<T: TensorScalar> Send for DivScalarBackward<T> {}
unsafe impl<T: TensorScalar> Sync for DivScalarBackward<T> {}
unsafe impl<T: TensorScalar> Send for NegBackward<T> {}
unsafe impl<T: TensorScalar> Sync for NegBackward<T> {}
