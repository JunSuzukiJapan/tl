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
        let ones = MetalTensor::ones(self.b_data.shape(), self.b_data.dtype())?;
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
        // GPU 完結: ones(input_shape) * grad_output (broadcast)
        let ones = MetalTensor::ones(&self.shape, DType::F32)?;
        Ok(vec![ones.mul_impl(grad_output)?])
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
        // GPU 完結: mask = (input > 0), grad * mask
        let zeros = MetalTensor::zeros(self.a_data.shape(), DType::F32);
        let mask = self.a_data.gt_impl(&zeros)?;
        Ok(vec![grad_output.mul_impl(&mask)?])
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
        // GPU 完結: s * (grad - sum(grad * s, dim=axis))
        let s = &self.output;
        let gs = grad_output.mul_impl(s)?;
        // dim に沿って sum → broadcast
        let sum_gs = gs.sum_impl(self.axis)?;
        let sum_unsqueezed = sum_gs.unsqueeze_impl(self.axis as usize)?;
        let sum_broad = sum_unsqueezed.broadcast_to_impl(s.shape())?;
        let diff = grad_output.sub_impl(&sum_broad)?;
        Ok(vec![s.mul_impl(&diff)?])
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
        let ones = MetalTensor::ones(self.output.shape(), self.output.dtype())?;
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
        // GPU 完結: unsqueeze(axis) → broadcast_to(input_shape)
        let expanded = grad_output.unsqueeze_impl(axis)?;
        Ok(vec![expanded.broadcast_to_impl(&self.input_shape)?])
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
        // GPU 完結: scatter_add パターン
        // indices から grad_output の行を vocab_size x embed_dim の勾配行列に加算
        let idx = unsafe { &*self.indices.get() };
        let grad_weight = MetalTensor::zeros(&[self.num_embeddings, self.embed_dim], DType::F32);
        
        // Embedding backward は scatter_add が必要なため、
        // 現時点では sync_stream + contents() でバッファ直接操作で実装
        // (to_vec と違い、全要素の Vec コピーは発生しない)
        crate::command_stream::sync_stream();
        let n = idx.elem_count();
        unsafe {
            let idx_ptr = idx.buffer().contents() as *const f32;
            let grad_ptr = grad_output.buffer().contents() as *const f32;
            let out_ptr = grad_weight.buffer().contents() as *mut f32;
            for i in 0..n {
                let target_row = *idx_ptr.add(i) as usize;
                if target_row < self.num_embeddings {
                    for j in 0..self.embed_dim {
                        *out_ptr.add(target_row * self.embed_dim + j) += *grad_ptr.add(i * self.embed_dim + j);
                    }
                }
            }
        }
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
        let logits_shape = logits.shape();
        let num_classes = if logits_shape.len() >= 2 { *logits_shape.last().unwrap() } else { logits.elem_count() };
        let batch_size = labels.elem_count();

        // softmax(logits) を計算
        let sm = logits.softmax_impl(-1)?;

        // integer labels → one_hot → softmax - one_hot
        // GPU sync してバッファ直接操作
        crate::command_stream::sync_stream();
        let label_data: Vec<f32> = labels.to_vec();
        let mut grad_data: Vec<f32> = sm.to_vec();

        for i in 0..batch_size {
            let idx = label_data[i] as usize;
            if idx < num_classes {
                grad_data[i * num_classes + idx] -= 1.0;
            }
        }

        // grad / batch_size
        let scale = 1.0 / batch_size as f32;
        for v in grad_data.iter_mut() {
            *v *= scale;
        }

        let grad = MetalTensor::from_slice(&grad_data, logits_shape, DType::F32);
        Ok(vec![grad])
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
        // GPU 完結: ones(input_shape) * grad_output * (1/n)
        let numel = self.shape.iter().product::<usize>() as f32;
        let ones = MetalTensor::ones(&self.shape, DType::F32)?;
        let scaled = grad_output.mul_scalar_impl(1.0 / numel)?;
        Ok(vec![ones.mul_impl(&scaled)?])
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
        // GPU 完結: sign(x) = gt(x,0) - lt(x,0) → かけ算
        let input = unsafe { &*self.a.get() };
        let zeros = MetalTensor::zeros(input.shape(), DType::F32);
        let pos = input.gt_impl(&zeros)?; // 1.0 where x > 0
        let neg = input.lt_impl(&zeros)?; // 1.0 where x < 0
        let sign = pos.sub_impl(&neg)?;   // 1, 0, -1
        Ok(vec![grad_output.mul_impl(&sign)?])
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
        // GPU 完結: CUDA版と同一パターン
        let x = unsafe { &*self.a.get() };
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        // inner = k * (x + 0.044715 * x^3)
        let x3 = x.pow_scalar_impl(3.0)?;
        let inner_arg = x.add_impl(&x3.mul_scalar_impl(0.044715)?)?;
        let inner = inner_arg.mul_scalar_impl(k)?;
        // cdf = 0.5 * (1 + tanh(inner))
        let tanh_inner = inner.tanh_impl()?;
        let cdf = tanh_inner.add_scalar_impl(1.0)?.mul_scalar_impl(0.5)?;
        // pdf = k * (1 - tanh²(inner)) * (1 + 3*0.044715*x²)
        let tanh_sq = tanh_inner.mul_impl(&tanh_inner)?;
        let one_minus_tanh_sq = tanh_sq.neg_impl()?.add_scalar_impl(1.0)?;
        let x2 = x.mul_impl(x)?;
        let coeff = x2.mul_scalar_impl(3.0 * 0.044715)?.add_scalar_impl(1.0)?;
        let pdf = one_minus_tanh_sq.mul_impl(&coeff)?.mul_scalar_impl(k)?;
        // grad_gelu = cdf + x * 0.5 * pdf
        let grad_gelu = cdf.add_impl(&x.mul_impl(&pdf)?.mul_scalar_impl(0.5)?)?;
        Ok(vec![grad_output.mul_impl(&grad_gelu)?])
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
        // GPU 完結: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let x = unsafe { &*self.a.get() };
        let sig = x.sigmoid_impl()?;
        let one_minus_sig = sig.neg_impl()?.add_scalar_impl(1.0)?;
        let x_term = x.mul_impl(&one_minus_sig)?;
        let bracket = x_term.add_scalar_impl(1.0)?;
        let grad_silu = sig.mul_impl(&bracket)?;
        Ok(vec![grad_output.mul_impl(&grad_silu)?])
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
        let ones = MetalTensor::ones(&self.input_shape, DType::F32)?;
        let expanded = ones.mul_impl(grad_output)?;
        Ok(vec![expanded.div_scalar_impl(dim_size)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// layer_norm の勾配
pub struct LayerNormBackward {
    pub input: TensorRef,
    pub weight: Option<TensorRef>,
    pub bias: Option<TensorRef>,
    pub eps: f32,
}

impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        let x = unsafe { &*self.input.get() };
        let shape = x.shape();
        let _ndim = shape.len();
        let _last_dim = *shape.last().unwrap_or(&1);
        
        // normalized = (x - mean) / sqrt(var + eps)
        // 簡易実装: grad_input ≈ grad_output (layer_norm は近似的に identity に近い)
        let grad_input = grad_output.shallow_clone();
        
        let mut grads = vec![grad_input];
        
        if let Some(ref _w_ref) = self.weight {
            // grad_weight: sum over batch dims of (grad_output * normalized_x)
            // 簡易近似: sum(grad_output, batch_dims)
            let go_shape = grad_output.shape();
            if go_shape.len() >= 2 {
                // [batch, seq, dim] → sum over [batch, seq] → [dim]
                let mut g = grad_output.shallow_clone();
                for _ in 0..(go_shape.len() - 1) {
                    g = g.sum_impl(0)?;
                }
                grads.push(g.clone());  // grad_weight
                grads.push(g);          // grad_bias (same shape)
            } else {
                grads.push(grad_output.shallow_clone());
                grads.push(grad_output.shallow_clone());
            }
        }
        
        Ok(grads)
    }
    fn inputs(&self) -> Vec<TensorRef> {
        let mut inputs = vec![self.input.clone()];
        if let Some(ref w) = self.weight {
            inputs.push(w.clone());
        }
        if let Some(ref b) = self.bias {
            inputs.push(b.clone());
        }
        inputs
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
        // 簡易: grad をそのまま通す（CUDA 版と同一パターン）
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
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
        // 簡易: grad をそのまま通す（CUDA 版と同一パターン）
        Ok(vec![grad_output.shallow_clone()])
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
        // GPU 完結: mask = (output != 0), grad * mask * scale
        let scale = 1.0 / (1.0 - self.p);
        let zeros = MetalTensor::zeros(self.output.shape(), DType::F32);
        // output != 0 → 1.0、output == 0 → 0.0
        let mask = self.output.ne_impl(&zeros)?;
        let scaled_grad = grad_output.mul_scalar_impl(scale)?;
        Ok(vec![scaled_grad.mul_impl(&mask)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone()]
    }
}

/// tril の勾配: 下三角マスクを勾配にも適用
pub struct TrilBackward {
    pub a: TensorRef,
    pub diagonal: i32,
}

impl GradFn for TrilBackward {
    fn backward(&self, grad_output: &MetalTensor) -> BackendResult<Vec<MetalTensor>> {
        Ok(vec![grad_output.tril_impl(self.diagonal)?])
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
unsafe impl Send for TrilBackward {}
unsafe impl Sync for TrilBackward {}

