//! CPU 版 演算の勾配関数

use super::GradFn;
use crate::scalar::TensorScalar;
use crate::tensor::{CpuTensor, TensorRef};
use tl_backend::BackendResult;


/// ブロードキャスト勾配集約:
/// grad の shape を target_shape に合わせるため、ブロードキャストで追加された次元に沿って sum する。
fn reduce_grad_for_broadcast<T: TensorScalar>(grad: &CpuTensor<T>, target_shape: &[usize]) -> BackendResult<CpuTensor<T>> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.shallow_clone());
    }
    
    let mut result = grad.shallow_clone();
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    
    // target_shape が短い場合、先頭の余分な次元を sum で潰す
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum_impl(0)?; // dim 0 に沿って sum (keep_dim=false で次元が減る)
        }
    }
    
    // 同じ次元数で、target が 1 の次元を sum + keep_dim
    let result_shape = result.shape().to_vec();
    let min_ndim = result_shape.len().min(target_shape.len());
    for d in 0..min_ndim {
        if target_shape[d] == 1 && result_shape[d] > 1 {
            result = result.sum_impl(d as i32)?;
            // sum_impl は keep_dim=false なので、shape を復元
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(d, 1);
            result = result.reshape_impl(&new_shape)?;
        }
    }
    
    // 最終的な shape を target_shape に合わせて reshape
    if result.shape() != target_shape {
        result = result.reshape_impl(target_shape)?;
    }
    
    Ok(result)
}

/// 加算の勾配
pub struct AddBackward<T: TensorScalar> {
    pub a: TensorRef<T>,
    pub b: TensorRef<T>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(grad_output, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_output.neg_impl()?, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.b_data)?, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.a_data)?, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.div_impl(&self.b_data)?, &self.a_shape)?;
        let b_sq = self.b_data.mul_impl(&self.b_data)?;
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.a_data)?.neg_impl()?.div_impl(&b_sq)?, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let ones = CpuTensor::<T>::ones(self.b_data.shape(), self.b_data.dtype());
        let b_minus_1 = self.b_data.sub_impl(&ones)?;
        let grad_a = grad_output.mul_impl(&self.b_data)?.mul_impl(&self.a_data.pow_impl(&b_minus_1)?)?;
        Ok(vec![grad_a])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_val = grad_output.to_vec_t()[0];
        let count: usize = self.shape.iter().product();
        let grads = vec![grad_val; count];
        Ok(vec![CpuTensor::from_slice(&grads, &self.shape, grad_output.dtype())])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let data = self.a_data.to_vec_t();
        let grad = grad_output.to_vec_t();
        let result: Vec<T> = data.iter().zip(grad.iter())
            .map(|(a, g)| if *a > T::zero() { *g } else { T::zero() })
            .collect();
        Ok(vec![CpuTensor::from_slice(&result, self.a_data.shape(), self.a_data.dtype())])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let s = self.output.to_vec_t();
        let g = grad_output.to_vec_t();
        let shape = self.output.shape();
        let ndim = shape.len();

        // 正規化された axis
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();

        let mut result = vec![T::zero(); s.len()];

        // 各 softmax スライス (outer × inner) ごとに独立して勾配計算
        for o in 0..outer {
            for k in 0..inner {
                // sg_sum = sum_j(s[j] * g[j]) for this slice
                let mut sg_sum = T::zero();
                for j in 0..axis_size {
                    let idx = o * axis_size * inner + j * inner + k;
                    sg_sum = sg_sum + s[idx] * g[idx];
                }
                // result[j] = s[j] * (g[j] - sg_sum)
                for j in 0..axis_size {
                    let idx = o * axis_size * inner + j * inner + k;
                    result[idx] = s[idx] * (g[idx] - sg_sum);
                }
            }
        }

        Ok(vec![CpuTensor::from_slice(&result, shape, self.output.dtype())])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let a_ndim = self.a_data.shape().len();
        let b_ndim = self.b_data.shape().len();

        // grad_a = grad_output @ b^T  (transpose last 2 dims of b)
        let b_t = if b_ndim >= 2 {
            self.b_data.transpose_impl(b_ndim - 2, b_ndim - 1)?
        } else {
            self.b_data.shallow_clone()
        };
        let grad_a = grad_output.matmul_impl(&b_t)?;

        // grad_b = a^T @ grad_output  (transpose last 2 dims of a)
        let a_t = if a_ndim >= 2 {
            self.a_data.transpose_impl(a_ndim - 2, a_ndim - 1)?
        } else {
            self.a_data.shallow_clone()
        };
        let grad_b = a_t.matmul_impl(grad_output)?;

        // grad_b の shape が b の shape と異なる場合 (batch dims), sum で集約
        let b_shape = self.b_data.shape();
        let gb_shape = grad_b.shape();
        let grad_b = if gb_shape != b_shape && gb_shape.len() > b_shape.len() {
            // batch dims を sum で削除 (例: [1,64,13] -> [64,13])
            let extra = gb_shape.len() - b_shape.len();
            let mut g = grad_b;
            for _ in 0..extra {
                g = g.sum_impl(0)?;
            }
            g
        } else {
            grad_b
        };

        Ok(vec![grad_a, grad_b])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let ones = CpuTensor::<T>::ones(self.output.shape(), self.output.dtype());
        let one_minus_s = ones.sub_impl(&self.output)?;
        let grad = grad_output.mul_impl(&self.output)?.mul_impl(&one_minus_s)?;
        Ok(vec![grad])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.mul_impl(&self.output)?])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.div_impl(&self.a_data)?])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
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
            return Ok(vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())]);
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
        Ok(vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.reshape_impl(&self.input_shape)?])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.shallow_clone()])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.shallow_clone()])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.mul_scalar_impl(self.s)?])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.div_scalar_impl(self.s)?])
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
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.neg_impl()?])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.a.clone()]
    }
}

/// cross_entropy の勾配
/// CrossEntropy(logits, targets) = -sum(one_hot(targets) * log_softmax(logits))
/// grad_logits = softmax(logits) - one_hot(targets)
pub struct CrossEntropyBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub softmax_output: CpuTensor<T>,
    pub targets: CpuTensor<T>,
    pub num_classes: usize,
}

impl<T: TensorScalar> GradFn<T> for CrossEntropyBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let softmax_data = self.softmax_output.to_vec_t();
        let target_data = self.targets.to_vec_t();
        let batch_size = target_data.len();
        let num_classes = self.num_classes;
        let mut grad = softmax_data.clone();
        // grad = softmax - one_hot(targets)
        for i in 0..batch_size {
            let target_idx = T::to_usize(target_data[i]);
            if target_idx < num_classes {
                grad[i * num_classes + target_idx] = grad[i * num_classes + target_idx] - T::one();
            }
        }
        // scale by grad_output and batch_size
        let scale = grad_output.to_vec_t()[0];
        let inv_batch = T::from_f64(1.0 / batch_size as f64);
        let grad: Vec<T> = grad.iter().map(|g| *g * scale * inv_batch).collect();
        let shape = self.softmax_output.shape().to_vec();
        Ok(vec![CpuTensor::from_slice(&grad, &shape, self.softmax_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// transpose の勾配: 逆転置
pub struct TransposeBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub dim0: usize,
    pub dim1: usize,
}

impl<T: TensorScalar> GradFn<T> for TransposeBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        // 逆転置 = 同じ次元を再度転置
        Ok(vec![grad_output.transpose_impl(self.dim0, self.dim1)?])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// embedding の勾配
pub struct EmbeddingBackward<T: TensorScalar> {
    pub weight: TensorRef<T>,
    pub indices: Vec<i64>,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl<T: TensorScalar> GradFn<T> for EmbeddingBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let grad_data = grad_output.to_vec_t();
        let mut weight_grad = vec![T::zero(); self.vocab_size * self.embed_dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            if (idx as usize) < self.vocab_size {
                let src_start = i * self.embed_dim;
                let dst_start = (idx as usize) * self.embed_dim;
                for d in 0..self.embed_dim {
                    if src_start + d < grad_data.len() {
                        weight_grad[dst_start + d] = weight_grad[dst_start + d] + grad_data[src_start + d];
                    }
                }
            }
        }
        Ok(vec![CpuTensor::from_slice(&weight_grad, &[self.vocab_size, self.embed_dim], grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.weight.clone()]
    }
}

/// layer_norm の勾配
pub struct LayerNormBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_data: CpuTensor<T>,
    pub weight: TensorRef<T>,
    pub weight_data: CpuTensor<T>,
    pub bias: TensorRef<T>,
    pub eps: f64,
}

impl<T: TensorScalar> GradFn<T> for LayerNormBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let x = self.input_data.to_vec_t();
        let w = self.weight_data.to_vec_t();
        let dy = grad_output.to_vec_t();
        let shape = self.input_data.shape();
        let ndim = shape.len();
        let d = shape[ndim - 1]; // normalized dimension
        let n = x.len() / d; // batch size (all dims except last)

        let mut dx = vec![T::zero(); x.len()];
        let mut dw = vec![T::zero(); d]; // dγ: weight の勾配
        let mut db = vec![T::zero(); d]; // dβ: bias の勾配

        for i in 0..n {
            let start = i * d;
            let end = start + d;
            let row = &x[start..end];
            let dy_row = &dy[start..end];

            // mean
            let mut sum = T::zero();
            for &v in row { sum = sum + v; }
            let mean = sum * T::from_f64(1.0 / d as f64);

            // variance
            let mut var_sum = T::zero();
            for &v in row {
                let diff = v - mean;
                var_sum = var_sum + diff * diff;
            }
            let var = var_sum * T::from_f64(1.0 / d as f64);
            let std_inv = T::from_f64(1.0 / (T::to_f64(var) + self.eps).sqrt());

            // x_hat = (x - mean) / std
            let mut x_hat = vec![T::zero(); d];
            for j in 0..d {
                x_hat[j] = (row[j] - mean) * std_inv;
            }

            // dγ += dy * x_hat, dβ += dy (summed over batch)
            for j in 0..d {
                dw[j] = dw[j] + dy_row[j] * x_hat[j];
                db[j] = db[j] + dy_row[j];
            }

            // dx: grad through layer_norm
            let mut dy_sum = T::zero();
            let mut dy_xhat_sum = T::zero();
            for j in 0..d {
                let wdy = w[j] * dy_row[j];
                dy_sum = dy_sum + wdy;
                dy_xhat_sum = dy_xhat_sum + wdy * x_hat[j];
            }

            let inv_d = T::from_f64(1.0 / d as f64);
            for j in 0..d {
                let wdy = w[j] * dy_row[j];
                dx[start + j] = std_inv * (wdy - inv_d * (dy_sum + x_hat[j] * dy_xhat_sum));
            }
        }

        let dtype = self.input_data.dtype();
        Ok(vec![
            CpuTensor::from_slice(&dx, shape, dtype),
            CpuTensor::from_slice(&dw, &[d], dtype),
            CpuTensor::from_slice(&db, &[d], dtype),
        ])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone(), self.weight.clone(), self.bias.clone()]
    }
}

/// tril の勾配: 下三角マスクをそのまま適用
pub struct TrilBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub diagonal: i32,
}

impl<T: TensorScalar> GradFn<T> for TrilBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.tril_impl(self.diagonal)?])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// slice の勾配: ゼロパディングで元 shape に scatter
pub struct SliceBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl<T: TensorScalar> GradFn<T> for SliceBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let total: usize = self.input_shape.iter().product();
        let mut result = vec![T::zero(); total];
        let grad_data = grad_output.to_vec_t();

        let outer: usize = self.input_shape[..self.dim].iter().product();
        let inner: usize = self.input_shape[self.dim + 1..].iter().product();
        let dim_size = self.input_shape[self.dim];

        let mut gi = 0;
        for o in 0..outer.max(1) {
            let mut idx = self.start;
            while idx < self.end && idx < dim_size {
                for k in 0..inner {
                    let src = gi * inner + k;
                    let dst = o * dim_size * inner + idx * inner + k;
                    if src < grad_data.len() && dst < result.len() {
                        result[dst] = grad_data[src];
                    }
                }
                gi += 1;
                idx += self.step;
            }
        }
        Ok(vec![CpuTensor::from_slice(&result, &self.input_shape, grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// tanh の勾配: 1 - tanh²(x)
pub struct TanhBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub output_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for TanhBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let out = self.output_data.to_vec_t();
        let dy = grad_output.to_vec_t();
        let grad: Vec<T> = out.iter().zip(dy.iter())
            .map(|(&o, &g)| g * (T::one() - o * o))
            .collect();
        Ok(vec![CpuTensor::from_slice(&grad, self.output_data.shape(), grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// gelu の勾配
pub struct GeluBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for GeluBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let x = self.input_data.to_vec_t();
        let dy = grad_output.to_vec_t();
        let sqrt2_inv = T::from_f64(1.0 / std::f64::consts::SQRT_2);
        let coeff = T::from_f64(std::f64::consts::FRAC_2_SQRT_PI * 0.5);
        let half = T::from_f64(0.5);
        let grad: Vec<T> = x.iter().zip(dy.iter()).map(|(&xi, &dyi)| {
            let cdf = half * (T::one() + (xi * sqrt2_inv).tanh());
            let pdf = coeff * (-(xi * xi) * half).exp();
            dyi * (cdf + xi * pdf)
        }).collect();
        Ok(vec![CpuTensor::from_slice(&grad, self.input_data.shape(), grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// sqrt の勾配: 0.5 / sqrt(x)
pub struct SqrtBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub output_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for SqrtBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let out = self.output_data.to_vec_t();
        let dy = grad_output.to_vec_t();
        let half = T::from_f64(0.5);
        let eps = T::from_f64(1e-12);
        let grad: Vec<T> = out.iter().zip(dy.iter())
            .map(|(&o, &g)| g * half / (o + eps))
            .collect();
        Ok(vec![CpuTensor::from_slice(&grad, self.output_data.shape(), grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// mean の勾配: 1/n
pub struct MeanBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for MeanBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let n: usize = self.input_shape.iter().product();
        let scale = grad_output.to_vec_t()[0] * T::from_f64(1.0 / n as f64);
        let grad = vec![scale; n];
        Ok(vec![CpuTensor::from_slice(&grad, &self.input_shape, grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// squeeze の勾配: unsqueeze
pub struct SqueezeBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for SqueezeBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.reshape_impl(&self.input_shape)?])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// unsqueeze の勾配: squeeze
pub struct UnsqueezeBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_shape: Vec<usize>,
}

impl<T: TensorScalar> GradFn<T> for UnsqueezeBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        Ok(vec![grad_output.reshape_impl(&self.input_shape)?])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
    }
}

/// silu (x * sigmoid(x)) の勾配
pub struct SiluBackward<T: TensorScalar> {
    pub input: TensorRef<T>,
    pub input_data: CpuTensor<T>,
}

impl<T: TensorScalar> GradFn<T> for SiluBackward<T> {
    fn backward(&self, grad_output: &CpuTensor<T>) -> BackendResult<Vec<CpuTensor<T>>> {
        let x = self.input_data.to_vec_t();
        let dy = grad_output.to_vec_t();
        let grad: Vec<T> = x.iter().zip(dy.iter()).map(|(&xi, &dyi)| {
            let sig = T::one() / (T::one() + (-xi).exp());
            dyi * (sig * (T::one() + xi * (T::one() - sig)))
        }).collect();
        Ok(vec![CpuTensor::from_slice(&grad, self.input_data.shape(), grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef<T>> {
        vec![self.input.clone()]
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
unsafe impl<T: TensorScalar> Send for CrossEntropyBackward<T> {}
unsafe impl<T: TensorScalar> Sync for CrossEntropyBackward<T> {}
unsafe impl<T: TensorScalar> Send for TransposeBackward<T> {}
unsafe impl<T: TensorScalar> Sync for TransposeBackward<T> {}
unsafe impl<T: TensorScalar> Send for EmbeddingBackward<T> {}
unsafe impl<T: TensorScalar> Sync for EmbeddingBackward<T> {}
unsafe impl<T: TensorScalar> Send for LayerNormBackward<T> {}
unsafe impl<T: TensorScalar> Sync for LayerNormBackward<T> {}
unsafe impl<T: TensorScalar> Send for TrilBackward<T> {}
unsafe impl<T: TensorScalar> Sync for TrilBackward<T> {}
unsafe impl<T: TensorScalar> Send for SliceBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SliceBackward<T> {}
unsafe impl<T: TensorScalar> Send for TanhBackward<T> {}
unsafe impl<T: TensorScalar> Sync for TanhBackward<T> {}
unsafe impl<T: TensorScalar> Send for GeluBackward<T> {}
unsafe impl<T: TensorScalar> Sync for GeluBackward<T> {}
unsafe impl<T: TensorScalar> Send for SqrtBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SqrtBackward<T> {}
unsafe impl<T: TensorScalar> Send for MeanBackward<T> {}
unsafe impl<T: TensorScalar> Sync for MeanBackward<T> {}
unsafe impl<T: TensorScalar> Send for SqueezeBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SqueezeBackward<T> {}
unsafe impl<T: TensorScalar> Send for UnsqueezeBackward<T> {}
unsafe impl<T: TensorScalar> Sync for UnsqueezeBackward<T> {}
unsafe impl<T: TensorScalar> Send for SiluBackward<T> {}
unsafe impl<T: TensorScalar> Sync for SiluBackward<T> {}


