//! 演算の勾配関数（V5.0: Arc ベース TensorRef）

use super::GradFn;
use crate::tensor::{MetalTensor, TensorRef};
use crate::DType;
use tl_backend::{BackendResult};

/// ブロードキャスト勾配集約:
/// grad の shape を target_shape に合わせるため、ブロードキャストで追加された次元に沿って sum する。
fn reduce_grad_for_broadcast(grad: &MetalTensor, target_shape: &[usize]) -> BackendResult<MetalTensor> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.shallow_clone());
    }
    
    let mut result = grad.shallow_clone();
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum_impl(0)?;
        }
    }
    
    // 同じ次元数で、target が 1 の次元を sum + keep_dim
    let result_shape = result.shape().to_vec();
    let min_ndim = result_shape.len().min(target_shape.len());
    for d in 0..min_ndim {
        if target_shape[d] == 1 && result_shape[d] > 1 {
            result = result.sum_impl(d as i32)?;
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(d, 1);
            result = result.reshape_impl(&new_shape)?;
        }
    }
    
    if result.shape() != target_shape {
        result = result.reshape_impl(target_shape)?;
    }
    
    Ok(result)
}

/// 加算の勾配
pub struct AddBackward {
    pub a: TensorRef,
    pub b: TensorRef,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(grad_output, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 減算の勾配
pub struct SubBackward {
    pub a: TensorRef,
    pub b: TensorRef,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let grad_a = reduce_grad_for_broadcast(grad_output, &self.a_shape)?;
        let grad_b_neg = grad_output.neg_impl()?;
        let grad_b = reduce_grad_for_broadcast(&grad_b_neg, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 乗算の勾配
pub struct MulBackward {
    pub a: TensorRef,
    pub b: TensorRef,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.b_data)?, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_output.mul_impl(&self.a_data)?, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 除算の勾配
pub struct DivBackward {
    pub a: TensorRef,
    pub b: TensorRef,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let grad_a = reduce_grad_for_broadcast(&grad_output.div_impl(&self.b_data)?, &self.a_shape)?;
        let b_sq = self.b_data.mul_impl(&self.b_data)?;
        let grad_b_tmp = grad_output.mul_impl(&self.a_data)?.neg_impl()?.div_impl(&b_sq)?;
        let grad_b = reduce_grad_for_broadcast(&grad_b_tmp, &self.b_shape)?;
        Ok(vec![grad_a, grad_b])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// べき乗の勾配 (a^b)
pub struct PowBackward {
    pub a: TensorRef,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
    pub output: MetalTensor,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let ones = MetalTensor::ones(self.b_data.shape(), self.b_data.dtype());
        let b_minus_1 = self.b_data.sub_impl(&ones)?;
        let grad_a = grad_output.mul_impl(&self.b_data)?.mul_impl(&self.a_data.pow_impl(&b_minus_1)?)?;
        Ok(vec![grad_a])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// sumall の勾配
pub struct SumallBackward {
    pub a: TensorRef,
    pub shape: Vec<usize>,
}

impl GradFn for SumallBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let grad_val = grad_output.to_vec::<f32>().first().copied().unwrap_or(0.0);
        let count: usize = self.shape.iter().product();
        let grads = vec![grad_val; count];
        Ok(vec![MetalTensor::from_slice(&grads, &self.shape, grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// ReLU の勾配
pub struct ReluBackward {
    pub a: TensorRef,
    pub a_data: MetalTensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let data: Vec<f32> = self.a_data.to_vec();
        let grad: Vec<f32> = grad_output.to_vec();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(a, g)| if *a > 0.0 { *g } else { 0.0 })
            .collect();
        Ok(vec![MetalTensor::from_slice(&result, self.a_data.shape(), self.a_data.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// Softmax の勾配
pub struct SoftmaxBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
    pub axis: i32,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
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
            Ok(vec![MetalTensor::from_slice(&result, shape, self.output.dtype())])
        } else {
            let sg_sum: f32 = s.iter().zip(g.iter()).map(|(si, gi)| si * gi).sum();
            let result: Vec<f32> = s.iter().zip(g.iter())
                .map(|(si, gi)| si * (gi - sg_sum))
                .collect();
            Ok(vec![MetalTensor::from_slice(&result, shape, self.output.dtype())])
        }
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// matmul の勾配
pub struct MatmulBackward {
    pub a: TensorRef,
    pub b: TensorRef,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let a_ndim = self.a_data.shape().len();
        let b_ndim = self.b_data.shape().len();

        let b_t = if b_ndim >= 2 {
            self.b_data.transpose_impl(b_ndim - 2, b_ndim - 1)?
        } else {
            self.b_data.transpose_impl(0, 1)?
        };
        let grad_a_raw = grad_output.matmul_impl(&b_t)?;

        let a_t = if a_ndim >= 2 {
            self.a_data.transpose_impl(a_ndim - 2, a_ndim - 1)?
        } else {
            self.a_data.transpose_impl(0, 1)?
        };
        let grad_b_raw = a_t.matmul_impl(grad_output)?;
        
        let a_shape = unsafe { &*self.a.get() }.shape().to_vec();
        let b_shape = unsafe { &*self.b.get() }.shape().to_vec();
        
        let grad_a = reduce_grad_for_broadcast(&grad_a_raw, &a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_b_raw, &b_shape)?;
        
        Ok(vec![grad_a, grad_b])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// sigmoid の勾配
pub struct SigmoidBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let ones = MetalTensor::ones(self.output.shape(), self.output.dtype());
        let one_minus_s = ones.sub_impl(&self.output)?;
        let grad = grad_output.mul_impl(&self.output)?.mul_impl(&one_minus_s)?;
        Ok(vec![grad])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// exp の勾配
pub struct ExpBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.mul_impl(&self.output)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// log の勾配
pub struct LogBackward {
    pub a: TensorRef,
    pub a_data: MetalTensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.div_impl(&self.a_data)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// sum(dim) の勾配
pub struct SumDimBackward {
    pub a: TensorRef,
    pub input_shape: Vec<usize>,
    pub axis: i32,
}

impl GradFn for SumDimBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let ndim = self.input_shape.len();
        let axis = if self.axis < 0 {
            (ndim as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        if ndim == 0 || axis >= ndim {
            let grad_val = grad_output.to_vec::<f32>().first().copied().unwrap_or(0.0);
            let numel: usize = self.input_shape.iter().product::<usize>().max(1);
            let result = vec![grad_val; numel];
            return Ok(vec![MetalTensor::from_slice(&result, &self.input_shape, grad_output.dtype())]);
        }
        
        let grad_data: Vec<f32> = grad_output.to_vec();
        let numel: usize = self.input_shape.iter().product();
        let mut result = vec![0.0f32; numel];
        
        let outer_size: usize = self.input_shape[..axis].iter().product();
        let inner_size: usize = self.input_shape[axis + 1..].iter().product();
        let axis_size = self.input_shape[axis];
        
        for i in 0..outer_size {
            for k in 0..inner_size {
                let gi = i * inner_size + k;
                let grad_val = if gi < grad_data.len() { grad_data[gi] } else { 0.0 };
                for j in 0..axis_size {
                    result[i * axis_size * inner_size + j * inner_size + k] = grad_val;
                }
            }
        }
        
        Ok(vec![MetalTensor::from_slice(&result, &self.input_shape, grad_output.dtype())])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// reshape の勾配
pub struct ReshapeBackward {
    pub input: TensorRef,
    pub input_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.reshape_impl(&self.input_shape)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// 符号反転の勾配 (-t) -> grad_t = -grad_output
pub struct NegBackward {
    pub a: TensorRef,
}

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.neg_impl()?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// スカラ加算の勾配 (t + s) -> grad_t = grad_output
pub struct AddScalarBackward {
    pub a: TensorRef,
}

impl GradFn for AddScalarBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// スカラ減算の勾配 (t - s) -> grad_t = grad_output
pub struct SubScalarBackward {
    pub a: TensorRef,
}

impl GradFn for SubScalarBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// スカラ乗算の勾配 (t * s) -> grad_t = grad_output * s
pub struct MulScalarBackward {
    pub a: TensorRef,
    pub s: f32,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.mul_scalar_impl(self.s)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// スカラ除算の勾配 (t / s) -> grad_t = grad_output / s
pub struct DivScalarBackward {
    pub a: TensorRef,
    pub s: f32,
}

impl GradFn for DivScalarBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.div_scalar_impl(self.s)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// Embedding の勾配
pub struct EmbeddingBackward {
    pub weight: TensorRef,
    pub indices: TensorRef,
    pub num_embeddings: usize,
    pub embed_dim: usize,
}

impl GradFn for EmbeddingBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let indices_tensor = unsafe { &*self.indices.get() };
        let grad_data: Vec<f32> = grad_output.to_vec();
        let idx_data: Vec<f32> = indices_tensor.to_vec();
        let n = idx_data.len();

        let mut gw_data = vec![0.0f32; self.num_embeddings * self.embed_dim];
        for i in 0..n {
            let idx = idx_data[i] as usize;
            if idx < self.num_embeddings {
                for j in 0..self.embed_dim {
                    gw_data[idx * self.embed_dim + j] += grad_data[i * self.embed_dim + j];
                }
            }
        }
        let grad_weight = MetalTensor::from_slice(&gw_data, &[self.num_embeddings, self.embed_dim], DType::F32);
        Ok(vec![grad_weight])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.weight.clone()]
    }
}

/// tanh の勾配: grad_t = grad_output * (1 - tanh(x)^2)
pub struct TanhBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let one_minus_sq = self.output.mul_impl(&self.output)?.neg_impl()?.add_scalar_impl(1.0)?;
        Ok(vec![grad_output.mul_impl(&one_minus_sq)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// sqrt の勾配
pub struct SqrtBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
}

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let half_over_sqrt = self.output.pow_scalar_impl(-1.0)?.mul_scalar_impl(0.5)?;
        Ok(vec![grad_output.mul_impl(&half_over_sqrt)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// transpose の勾配
pub struct TransposeBackward {
    pub a: TensorRef,
    pub dim0: usize,
    pub dim1: usize,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.transpose_impl(self.dim0, self.dim1)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// cross_entropy の勾配
pub struct CrossEntropyBackward {
    pub logits: TensorRef,
    pub labels: TensorRef,
}

impl GradFn for CrossEntropyBackward {
    fn backward(&self, _grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let logits = unsafe { &*self.logits.get() };
        let labels = unsafe { &*self.labels.get() };
        let sm = logits.softmax_impl(-1)?;
        let diff = sm.sub_impl(labels)?;
        let batch = logits.shape()[0] as f32;
        Ok(vec![diff.div_scalar_impl(batch)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.logits.clone()]
    }
}

/// mean (全要素) の勾配
pub struct MeanAllBackward {
    pub a: TensorRef,
    pub shape: Vec<usize>,
}

impl GradFn for MeanAllBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let numel: usize = self.shape.iter().product();
        let ones = MetalTensor::ones(&self.shape, DType::F32);
        let scalar: Vec<f32> = grad_output.to_vec();
        Ok(vec![ones.mul_scalar_impl(scalar[0] / numel as f32)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// abs の勾配
pub struct AbsBackward {
    pub a: TensorRef,
}

impl GradFn for AbsBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let input = unsafe { &*self.a.get() };
        let data: Vec<f32> = input.to_vec();
        let sign_data: Vec<f32> = data.iter().map(|&v| {
            if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
        }).collect();
        let sign_tensor = MetalTensor::from_slice(&sign_data, input.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&sign_tensor)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// GELU の勾配
pub struct GeluBackward {
    pub a: TensorRef,
}

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let input = unsafe { &*self.a.get() };
        let data: Vec<f32> = input.to_vec();
        let sqrt_2: f32 = std::f32::consts::SQRT_2;
        let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();

        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let t = 1.0 / (1.0 + 0.3275911 * (x / sqrt_2).abs());
            let erf_approx = 1.0 - (0.254829592 * t - 0.284496736 * t * t
                + 1.421413741 * t * t * t - 1.453152027 * t * t * t * t
                + 1.061405429 * t * t * t * t * t) * (-(x / sqrt_2) * (x / sqrt_2)).exp();
            let erf_val = if x >= 0.0 { erf_approx } else { -erf_approx };
            let cdf = 0.5 * (1.0 + erf_val);
            let pdf = inv_sqrt_2pi * (-0.5 * x * x).exp();
            cdf + x * pdf
        }).collect();
        let grad_tensor = MetalTensor::from_slice(&grad_data, input.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&grad_tensor)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// SiLU (Swish) の勾配
pub struct SiluBackward {
    pub a: TensorRef,
}

impl GradFn for SiluBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let input = unsafe { &*self.a.get() };
        let data: Vec<f32> = input.to_vec();
        let grad_data: Vec<f32> = data.iter().map(|&x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            sig * (1.0 + x * (1.0 - sig))
        }).collect();
        let grad_tensor = MetalTensor::from_slice(&grad_data, input.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&grad_tensor)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// mean(dim) の勾配
pub struct MeanDimBackward {
    pub a: TensorRef,
    pub dim: usize,
    pub input_shape: Vec<usize>,
}

impl GradFn for MeanDimBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let dim_size = self.input_shape[self.dim] as f32;
        let ones = MetalTensor::ones(&self.input_shape, DType::F32);
        let expanded = ones.mul_impl(grad_output)?;
        Ok(vec![expanded.div_scalar_impl(dim_size)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// layer_norm の勾配 (簡易近似)
pub struct LayerNormBackward {
    pub input: TensorRef,
    pub weight: Option<TensorRef>,
    pub eps: f32,
}

impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let x = unsafe { &*self.input.get() };
        let shape = x.shape();
        let last_dim = *shape.last().unwrap();
        let n = last_dim as f32;
        let data: Vec<f32> = x.to_vec();
        let grad_data: Vec<f32> = grad_output.to_vec();
        let total = data.len();
        let num_groups = total / last_dim;

        let mut result = vec![0.0f32; total];

        for g in 0..num_groups {
            let start = g * last_dim;
            let end = start + last_dim;
            let group = &data[start..end];

            let mean: f32 = group.iter().sum::<f32>() / n;
            let var: f32 = group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            let g_out = &grad_data[start..end];

            let weighted_grad: Vec<f32> = if let Some(ref w_ref) = self.weight {
                let w = unsafe { &*w_ref.get() };
                let w_data: Vec<f32> = w.to_vec();
                g_out.iter().enumerate().map(|(i, &g)| g * w_data[i % w_data.len()]).collect()
            } else {
                g_out.to_vec()
            };

            let sum_g: f32 = weighted_grad.iter().sum();
            let x_hat: Vec<f32> = group.iter().map(|&v| (v - mean) * std_inv).collect();
            let sum_gx: f32 = weighted_grad.iter().zip(x_hat.iter()).map(|(&g, &x)| g * x).sum();

            for i in 0..last_dim {
                result[start + i] = std_inv * (weighted_grad[i] - sum_g / n - x_hat[i] * sum_gx / n);
            }
        }

        Ok(vec![MetalTensor::from_slice(&result, shape, DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// scale (定数倍) の勾配
pub struct ScaleBackward {
    pub a: TensorRef,
    pub s: f32,
}

impl GradFn for ScaleBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.mul_scalar_impl(self.s)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// Conv2D の勾配
pub struct Conv2dBackward {
    pub input: TensorRef,
    pub weight: TensorRef,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl GradFn for Conv2dBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let input = unsafe { &*self.input.get() };
        let weight = unsafe { &*self.weight.get() };
        let in_shape = input.shape();
        let w_shape = weight.shape();
        let g_shape = grad_output.shape();

        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (_, _, h_out, w_out) = (g_shape[0], g_shape[1], g_shape[2], g_shape[3]);
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let input_data: Vec<f32> = input.to_vec();
        let weight_data: Vec<f32> = weight.to_vec();
        let grad_data: Vec<f32> = grad_output.to_vec();

        let mut grad_input = vec![0.0f32; n * c_in * h_in * w_in];
        let mut grad_weight = vec![0.0f32; c_out * c_in * kh * kw];

        for batch in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g_idx = batch * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                        let g_val = grad_data[g_idx];

                        for ic in 0..c_in {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = oh * stride_h + khi;
                                    let iw = ow * stride_w + kwi;
                                    if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                        let in_idx = batch * c_in * h_in * w_in + ic * h_in * w_in + (ih - pad_h) * w_in + (iw - pad_w);
                                        let k_idx = oc * c_in * kh * kw + ic * kh * kw + khi * kw + kwi;

                                        grad_input[in_idx] += g_val * weight_data[k_idx];
                                        grad_weight[k_idx] += g_val * input_data[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gi = MetalTensor::from_slice(&grad_input, &[n, c_in, h_in, w_in], DType::F32);
        let gw = MetalTensor::from_slice(&grad_weight, &[c_out, c_in, kh, kw], DType::F32);
        Ok(vec![gi, gw])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone(), self.weight.clone()]
    }
}

/// BatchNorm の勾配
pub struct BatchNormBackward {
    pub input: TensorRef,
    pub weight: TensorRef,
    pub running_mean: TensorRef,
    pub running_var: TensorRef,
    pub eps: f32,
}

impl GradFn for BatchNormBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let x = unsafe { &*self.input.get() };
        let gamma = unsafe { &*self.weight.get() };
        let mean = unsafe { &*self.running_mean.get() };
        let var = unsafe { &*self.running_var.get() };
        let shape = x.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;

        let x_data: Vec<f32> = x.to_vec();
        let gamma_data: Vec<f32> = gamma.to_vec();
        let mean_data: Vec<f32> = mean.to_vec();
        let var_data: Vec<f32> = var.to_vec();
        let grad_data: Vec<f32> = grad_output.to_vec();

        let mut grad_input = vec![0.0f32; n * c * spatial];
        let mut dgamma = vec![0.0f32; c];
        let mut dbeta = vec![0.0f32; c];

        for ch in 0..c {
            let inv_std = 1.0 / (var_data[ch] + self.eps).sqrt();
            let g = gamma_data[ch];

            for batch in 0..n {
                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    let x_hat = (x_data[idx] - mean_data[ch]) * inv_std;
                    grad_input[idx] = grad_data[idx] * g * inv_std;
                    dgamma[ch] += grad_data[idx] * x_hat;
                    dbeta[ch] += grad_data[idx];
                }
            }
        }

        Ok(vec![MetalTensor::from_slice(&grad_input, &[n, c, h, w], DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// Dropout の勾配
pub struct DropoutBackward {
    pub a: TensorRef,
    pub output: MetalTensor,
    pub p: f32,
}

impl GradFn for DropoutBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let scale = 1.0 / (1.0 - self.p);
        let out_data: Vec<f32> = self.output.to_vec();
        let grad_data: Vec<f32> = grad_output.to_vec();
        let result: Vec<f32> = out_data.iter().zip(grad_data.iter()).map(|(&o, &g)| {
            if o != 0.0 { g * scale } else { 0.0 }
        }).collect();
        let shape = grad_output.shape();
        Ok(vec![MetalTensor::from_slice(&result, shape, DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

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
unsafe impl Send for ReshapeBackward {}
unsafe impl Sync for ReshapeBackward {}
unsafe impl Send for NegBackward {}
unsafe impl Sync for NegBackward {}
unsafe impl Send for AddScalarBackward {}
unsafe impl Sync for AddScalarBackward {}
unsafe impl Send for SubScalarBackward {}
unsafe impl Sync for SubScalarBackward {}
unsafe impl Send for MulScalarBackward {}
unsafe impl Sync for MulScalarBackward {}
unsafe impl Send for DivScalarBackward {}
unsafe impl Sync for DivScalarBackward {}
unsafe impl Send for EmbeddingBackward {}
unsafe impl Sync for EmbeddingBackward {}
unsafe impl Send for TanhBackward {}
unsafe impl Sync for TanhBackward {}
unsafe impl Send for SqrtBackward {}
unsafe impl Sync for SqrtBackward {}
unsafe impl Send for TransposeBackward {}
unsafe impl Sync for TransposeBackward {}
unsafe impl Send for CrossEntropyBackward {}
unsafe impl Sync for CrossEntropyBackward {}
unsafe impl Send for MeanAllBackward {}
unsafe impl Sync for MeanAllBackward {}
unsafe impl Send for AbsBackward {}
unsafe impl Sync for AbsBackward {}
unsafe impl Send for GeluBackward {}
unsafe impl Sync for GeluBackward {}
unsafe impl Send for SiluBackward {}
unsafe impl Sync for SiluBackward {}
unsafe impl Send for MeanDimBackward {}
unsafe impl Sync for MeanDimBackward {}
unsafe impl Send for LayerNormBackward {}
unsafe impl Sync for LayerNormBackward {}
unsafe impl Send for ScaleBackward {}
unsafe impl Sync for ScaleBackward {}
unsafe impl Send for Conv2dBackward {}
unsafe impl Sync for Conv2dBackward {}
unsafe impl Send for BatchNormBackward {}
unsafe impl Sync for BatchNormBackward {}
unsafe impl Send for DropoutBackward {}
unsafe impl Sync for DropoutBackward {}
