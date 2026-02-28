//! CUDA Autograd 演算
//!
//! 各 forward 演算に対応する backward（勾配計算）を実装。

use crate::autograd::GradFn;
use crate::tensor::{CudaTensor, TensorRef};
use crate::DType;
use std::cell::UnsafeCell;
use std::sync::Arc;
use tl_backend::BackendResult;

/// TensorRef は Arc<UnsafeCell<CudaTensor>> なので Send/Sync の自動実装がない。
/// backward 構造体は GradFn: Send + Sync を要求するため手動で unsafe impl する。
macro_rules! unsafe_send_sync {
    ($($t:ty),*) => { $(
        unsafe impl Send for $t {}
        unsafe impl Sync for $t {}
    )* };
}

unsafe_send_sync!(
    AddBackward,
    SubBackward,
    MulBackward,
    DivBackward,
    PowBackward,
    MatmulBackward,
    NegBackward,
    AbsBackward,
    ExpBackward,
    LogBackward,
    SqrtBackward,
    TanhBackward,
    SigmoidBackward,
    ReluBackward,
    GeluBackward,
    SiluBackward,
    SumallBackward,
    MeanAllBackward,
    SumDimBackward,
    MeanDimBackward,
    ReshapeBackward,
    TransposeBackward,
    SoftmaxBackward,
    CrossEntropyBackward,
    EmbeddingBackward,
    LayerNormBackward,
    AddScalarBackward,
    SubScalarBackward,
    MulScalarBackward,
    DivScalarBackward,
    PowScalarBackward,
    ScaleBackward,
    Conv2dBackward,
    BatchNormBackward,
    DropoutBackward
);

/// TensorRef ヘルパー
fn make_ref(t: &CudaTensor) -> TensorRef {
    Arc::new(UnsafeCell::new(t.shallow_clone()))
}

fn get_ref(r: &TensorRef) -> &CudaTensor {
    unsafe { &*r.get() }
}

// ========== 基本二項演算 ==========

/// d(a+b)/da = 1, d(a+b)/db = 1
pub struct AddBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl AddBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for AddBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![
            grad_output.shallow_clone(),
            grad_output.shallow_clone(),
        ])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// d(a-b)/da = 1, d(a-b)/db = -1
pub struct SubBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl SubBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for SubBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let neg_grad = grad_output.neg_impl()?;
        Ok(vec![grad_output.shallow_clone(), neg_grad])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// d(a*b)/da = b, d(a*b)/db = a
pub struct MulBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl MulBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for MulBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let ga = grad_output.mul_impl(get_ref(&self.b))?;
        let gb = grad_output.mul_impl(get_ref(&self.a))?;
        Ok(vec![ga, gb])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// d(a/b)/da = 1/b, d(a/b)/db = -a/b²
pub struct DivBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl DivBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for DivBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let b = get_ref(&self.b);
        let a = get_ref(&self.a);
        let ga = grad_output.div_impl(b)?;
        let b_sq = b.mul_impl(b)?;
        let neg_a = a.neg_impl()?;
        let gb = grad_output.mul_impl(&neg_a.div_impl(&b_sq)?)?;
        Ok(vec![ga, gb])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// d(a^b)/da = b * a^(b-1), simplified: grad * b * a^(b-1)
pub struct PowBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl PowBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for PowBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let a = get_ref(&self.a);
        let b = get_ref(&self.b);
        let one = CudaTensor::ones(b.shape(), DType::F32);
        let b_minus_1 = b.sub_impl(&one)?;
        let a_pow = a.pow_impl(&b_minus_1)?;
        let ga = grad_output.mul_impl(&b.mul_impl(&a_pow)?)?;
        // grad for exponent: d/db = a^b * ln(a)
        let a_pow_b = a.pow_impl(b)?;
        let ln_a = a.log_impl()?;
        let gb = grad_output.mul_impl(&a_pow_b.mul_impl(&ln_a)?)?;
        Ok(vec![ga, gb])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad
pub struct MatmulBackward {
    pub a: TensorRef,
    pub b: TensorRef,
}
impl MatmulBackward {
    pub fn new(a: &CudaTensor, b: &CudaTensor) -> Self {
        Self {
            a: make_ref(a),
            b: make_ref(b),
        }
    }
}
impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let a = get_ref(&self.a);
        let b = get_ref(&self.b);
        let a_ndim = a.shape().len();
        let b_ndim = b.shape().len();
        let bt = b.transpose_impl(b_ndim - 2, b_ndim - 1)?;
        let at = a.transpose_impl(a_ndim - 2, a_ndim - 1)?;
        let ga = grad_output.matmul_impl(&bt)?;
        let gb = at.matmul_impl(grad_output)?;
        Ok(vec![ga, gb])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.a.clone(), self.b.clone()]
    }
}

// ========== 単項演算 ==========

/// d(-x)/dx = -1
pub struct NegBackward {
    pub input: TensorRef,
}
impl NegBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for NegBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.neg_impl()?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d|x|/dx = sign(x)
pub struct AbsBackward {
    pub input: TensorRef,
}
impl AbsBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for AbsBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let data = x.to_vec::<f32>();
        let sign: Vec<f32> = data
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        let sign_t = CudaTensor::from_slice(&sign, x.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&sign_t)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(exp(x))/dx = exp(x)
pub struct ExpBackward {
    pub input: TensorRef,
    pub output: TensorRef,
}
impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.mul_impl(get_ref(&self.output))?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(ln(x))/dx = 1/x
pub struct LogBackward {
    pub input: TensorRef,
}
impl LogBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for LogBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.div_impl(get_ref(&self.input))?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(sqrt(x))/dx = 1/(2*sqrt(x))
pub struct SqrtBackward {
    pub input: TensorRef,
    pub output: TensorRef,
}
impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let two_sqrt = get_ref(&self.output).scale_impl(2.0)?;
        Ok(vec![grad_output.div_impl(&two_sqrt)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(tanh(x))/dx = 1 - tanh²(x)
pub struct TanhBackward {
    pub input: TensorRef,
    pub output: TensorRef,
}
impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let out = get_ref(&self.output);
        let sq = out.mul_impl(out)?;
        let one = CudaTensor::ones(out.shape(), DType::F32);
        let factor = one.sub_impl(&sq)?;
        Ok(vec![grad_output.mul_impl(&factor)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
pub struct SigmoidBackward {
    pub input: TensorRef,
    pub output: TensorRef,
}
impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let out = get_ref(&self.output);
        let one = CudaTensor::ones(out.shape(), DType::F32);
        let factor = out.mul_impl(&one.sub_impl(out)?)?;
        Ok(vec![grad_output.mul_impl(&factor)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(relu(x))/dx = (x > 0) ? 1 : 0
pub struct ReluBackward {
    pub input: TensorRef,
}
impl ReluBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let data = x.to_vec::<f32>();
        let mask: Vec<f32> = data
            .iter()
            .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mask_t = CudaTensor::from_slice(&mask, x.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&mask_t)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// GELU backward (approximate)
pub struct GeluBackward {
    pub input: TensorRef,
}
impl GeluBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let data = x.to_vec::<f32>();
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        let grad_data: Vec<f32> = data
            .iter()
            .map(|&xi| {
                let cdf = 0.5 * (1.0 + (k * (xi + 0.044715 * xi * xi * xi)).tanh());
                let pdf = k
                    * (1.0 - (k * (xi + 0.044715 * xi * xi * xi)).tanh().powi(2))
                    * (1.0 + 3.0 * 0.044715 * xi * xi);
                cdf + xi * 0.5 * pdf
            })
            .collect();
        let gt = CudaTensor::from_slice(&grad_data, x.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&gt)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub struct SiluBackward {
    pub input: TensorRef,
}
impl SiluBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for SiluBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let data = x.to_vec::<f32>();
        let grad_data: Vec<f32> = data
            .iter()
            .map(|&xi| {
                let sig = 1.0 / (1.0 + (-xi).exp());
                sig * (1.0 + xi * (1.0 - sig))
            })
            .collect();
        let gt = CudaTensor::from_slice(&grad_data, x.shape(), DType::F32);
        Ok(vec![grad_output.mul_impl(&gt)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

// ========== リダクション ==========

/// d(sum(x))/dx = ones_like(x)
pub struct SumallBackward {
    pub input: TensorRef,
}
impl SumallBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for SumallBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let g_val = grad_output.to_vec::<f32>()[0];
        let data = vec![g_val; x.elem_count()];
        Ok(vec![CudaTensor::from_slice(&data, x.shape(), DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(mean(x))/dx = 1/n
pub struct MeanAllBackward {
    pub input: TensorRef,
}
impl MeanAllBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for MeanAllBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let n = x.elem_count() as f32;
        let g_val = grad_output.to_vec::<f32>()[0] / n;
        let data = vec![g_val; x.elem_count()];
        Ok(vec![CudaTensor::from_slice(&data, x.shape(), DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct SumDimBackward {
    pub input: TensorRef,
    pub dim: usize,
}
impl GradFn for SumDimBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let input_shape = x.shape();
        let dim = self.dim;
        // grad_output は sum(dim) で dim 次元が消えている
        // 元の shape に戻すため、dim 次元に沿って勾配を複製する
        //
        // 例: input=[3,4,5].sum(2) → grad=[3,4]
        //   result[i,j,k] = grad[i,j]  (k は無関係)
        let grad_data = grad_output.to_vec::<f32>();
        let out_count: usize = input_shape.iter().product();
        let mut result = vec![0.0f32; out_count];
        let ndim = input_shape.len();

        // input_shape の strides
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * input_shape[d + 1];
        }
        // grad_shape の strides (dim 次元を除いた shape)
        let grad_shape = grad_output.shape();
        let grad_ndim = grad_shape.len();
        let mut grad_strides = vec![1usize; grad_ndim];
        for d in (0..grad_ndim.saturating_sub(1)).rev() {
            grad_strides[d] = grad_strides[d + 1] * grad_shape[d + 1];
        }

        for i in 0..out_count {
            let mut rem = i;
            let mut grad_idx = 0;
            let mut gd = 0; // grad dim のインデックス
            for d in 0..ndim {
                let coord = rem / strides[d];
                rem %= strides[d];
                if d == dim {
                    // sum された dim → grad にはこの次元がない → スキップ
                    continue;
                }
                if gd < grad_ndim {
                    grad_idx += coord * grad_strides[gd];
                    gd += 1;
                }
            }
            if grad_idx < grad_data.len() {
                result[i] = grad_data[grad_idx];
            }
        }

        Ok(vec![CudaTensor::from_slice(
            &result,
            input_shape,
            DType::F32,
        )])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct MeanDimBackward {
    pub input: TensorRef,
    pub dim: usize,
}
impl MeanDimBackward {
    pub fn new(t: &CudaTensor, dim: usize) -> Self {
        Self {
            input: make_ref(t),
            dim,
        }
    }
}
impl GradFn for MeanDimBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let n = x.shape()[self.dim] as f32;
        let scaled = grad_output.scale_impl(1.0 / n)?;
        Ok(vec![scaled.broadcast_to_impl(x.shape())?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

// ========== 形状操作 ==========

pub struct ReshapeBackward {
    pub input: TensorRef,
}
impl ReshapeBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.reshape_impl(get_ref(&self.input).shape())?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct TransposeBackward {
    pub dim0: usize,
    pub dim1: usize,
    pub input: TensorRef,
}
impl TransposeBackward {
    pub fn new(t: &CudaTensor, d0: usize, d1: usize) -> Self {
        Self {
            input: make_ref(t),
            dim0: d0,
            dim1: d1,
        }
    }
}
impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.transpose_impl(self.dim0, self.dim1)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

// ========== 特殊 ==========

pub struct SoftmaxBackward {
    pub input: TensorRef,
    pub output: TensorRef,
    pub dim: i32,
}
impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        // softmax grad: s * (grad - sum(grad * s, dim=d))
        // sum は softmax が適用された dim に沿って計算
        let s = get_ref(&self.output);
        let gs = grad_output.mul_impl(s)?;

        // dim に沿って sum → unsqueeze → broadcast
        let sum_gs = gs.sum_impl(self.dim)?;
        let sum_unsqueezed = sum_gs.unsqueeze_impl(self.dim as usize)?;
        let sum_broad = sum_unsqueezed.broadcast_to_impl(s.shape())?;
        let diff = grad_output.sub_impl(&sum_broad)?;
        Ok(vec![s.mul_impl(&diff)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct CrossEntropyBackward {
    pub logits: TensorRef,
    pub targets: TensorRef,
}
impl CrossEntropyBackward {
    pub fn new(logits: &CudaTensor, targets: &CudaTensor) -> Self {
        Self {
            logits: make_ref(logits),
            targets: make_ref(targets),
        }
    }
}
impl GradFn for CrossEntropyBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let logits = get_ref(&self.logits);
        let targets = get_ref(&self.targets);
        // softmax(logits) - one_hot(targets)
        let probs = logits.softmax_impl(-1)?;
        let mut grad_data = probs.to_vec::<f32>();
        let target_data = targets.to_vec::<i64>();
        let batch = target_data.len();
        let classes = grad_data.len() / batch;
        for i in 0..batch {
            let t = target_data[i] as usize;
            if t < classes {
                grad_data[i * classes + t] -= 1.0;
            }
        }
        let scale = grad_output.to_vec::<f32>()[0] / batch as f32;
        let grad: Vec<f32> = grad_data.iter().map(|&x| x * scale).collect();
        Ok(vec![CudaTensor::from_slice(
            &grad,
            logits.shape(),
            DType::F32,
        )])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.logits.clone()]
    }
}

pub struct EmbeddingBackward {
    pub weight: TensorRef,
    pub indices: TensorRef,
}
impl EmbeddingBackward {
    pub fn new(w: &CudaTensor, idx: &CudaTensor) -> Self {
        Self {
            weight: make_ref(w),
            indices: make_ref(idx),
        }
    }
}
impl GradFn for EmbeddingBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let w = get_ref(&self.weight);
        let idx = get_ref(&self.indices);
        let mut grad_w = vec![0.0f32; w.elem_count()];
        let grad_data = grad_output.to_vec::<f32>();
        let idx_data = idx.to_vec::<i64>();
        let embed_dim = w.shape()[1];
        for (i, &token) in idx_data.iter().enumerate() {
            let t = token as usize;
            for j in 0..embed_dim {
                grad_w[t * embed_dim + j] += grad_data[i * embed_dim + j];
            }
        }
        Ok(vec![CudaTensor::from_slice(&grad_w, w.shape(), DType::F32)])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.weight.clone()]
    }
}

pub struct LayerNormBackward {
    pub input: TensorRef,
}
impl LayerNormBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        // 簡易: grad をそのまま通す
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

// ========== スカラー演算 ==========

/// d(x+s)/dx = 1
pub struct AddScalarBackward {
    pub input: TensorRef,
}
impl AddScalarBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for AddScalarBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct SubScalarBackward {
    pub input: TensorRef,
}
impl SubScalarBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for SubScalarBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(x*s)/dx = s
pub struct MulScalarBackward {
    pub input: TensorRef,
    pub scalar: f32,
}
impl MulScalarBackward {
    pub fn new(t: &CudaTensor, s: f32) -> Self {
        Self {
            input: make_ref(t),
            scalar: s,
        }
    }
}
impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.scale_impl(self.scalar)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(x/s)/dx = 1/s
pub struct DivScalarBackward {
    pub input: TensorRef,
    pub scalar: f32,
}
impl DivScalarBackward {
    pub fn new(t: &CudaTensor, s: f32) -> Self {
        Self {
            input: make_ref(t),
            scalar: s,
        }
    }
}
impl GradFn for DivScalarBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.scale_impl(1.0 / self.scalar)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(x*s)/dx = s  (scale)
pub struct ScaleBackward {
    pub input: TensorRef,
    pub scalar: f32,
}
impl ScaleBackward {
    pub fn new(t: &CudaTensor, s: f32) -> Self {
        Self {
            input: make_ref(t),
            scalar: s,
        }
    }
}
impl GradFn for ScaleBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.scale_impl(self.scalar)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

/// d(x^n)/dx = n * x^(n-1)
pub struct PowScalarBackward {
    pub input: TensorRef,
    pub scalar: f32,
}
impl PowScalarBackward {
    pub fn new(t: &CudaTensor, s: f32) -> Self {
        Self {
            input: make_ref(t),
            scalar: s,
        }
    }
}
impl GradFn for PowScalarBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        let x = get_ref(&self.input);
        let n = self.scalar;
        let x_pow = x.pow_scalar_impl(n - 1.0)?;
        let coeff = x_pow.scale_impl(n)?;
        Ok(vec![grad_output.mul_impl(&coeff)?])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

// ========== NN ==========

pub struct Conv2dBackward {
    pub input: TensorRef,
}
impl Conv2dBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for Conv2dBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct BatchNormBackward {
    pub input: TensorRef,
}
impl BatchNormBackward {
    pub fn new(t: &CudaTensor) -> Self {
        Self { input: make_ref(t) }
    }
}
impl GradFn for BatchNormBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.input.clone()]
    }
}

pub struct DropoutBackward {
    pub mask: TensorRef,
}
impl DropoutBackward {
    pub fn new(mask: &CudaTensor) -> Self {
        Self {
            mask: make_ref(mask),
        }
    }
}
impl GradFn for DropoutBackward {
    fn backward(&self, grad_output: &CudaTensor) -> BackendResult<Vec<CudaTensor>> {
        Ok(vec![grad_output.shallow_clone()])
    }
    fn inputs(&self) -> Vec<TensorRef> {
        vec![self.mask.clone()]
    }
}
