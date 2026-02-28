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

/// broadcast backward: grad shape を input shape に reduce する
/// GPU 上で sum_impl + unsqueeze_impl を使い broadcast 次元を reduce
fn reduce_grad_to_shape(grad: &CudaTensor, target_shape: &[usize]) -> BackendResult<CudaTensor> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.shallow_clone());
    }

    let mut result = grad.shallow_clone();
    let mut cur_shape = grad_shape.to_vec();

    // ランク差があれば先頭次元を sum して消す
    while cur_shape.len() > target_shape.len() {
        result = result.sum_impl(0)?;
        cur_shape.remove(0);
    }

    // 同ランクで broadcast された次元 (size==1) を sum + unsqueeze
    for d in 0..target_shape.len() {
        if target_shape[d] == 1 && cur_shape[d] > 1 {
            result = result.sum_impl(d as i32)?;
            result = result.unsqueeze_impl(d)?;
            cur_shape[d] = 1;
        }
    }

    Ok(result)
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
        let ga = reduce_grad_to_shape(grad_output, get_ref(&self.a).shape())?;
        let gb = reduce_grad_to_shape(grad_output, get_ref(&self.b).shape())?;
        Ok(vec![ga, gb])
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
        let ga = reduce_grad_to_shape(grad_output, get_ref(&self.a).shape())?;
        let gb = reduce_grad_to_shape(&neg_grad, get_ref(&self.b).shape())?;
        Ok(vec![ga, gb])
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
        let ga = reduce_grad_to_shape(&ga, get_ref(&self.a).shape())?;
        let gb = reduce_grad_to_shape(&gb, get_ref(&self.b).shape())?;
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
        let ga = reduce_grad_to_shape(&ga, a.shape())?;
        let gb = reduce_grad_to_shape(&gb, b.shape())?;
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
        // sign(x) = gt(x,0) - lt(x,0)  → 1, 0, -1
        let zeros = CudaTensor::zeros(x.shape(), DType::F32);
        let pos = x.gt_impl(&zeros)?; // 1.0 where x > 0
        let neg = x.lt_impl(&zeros)?; // 1.0 where x < 0
        let sign = pos.sub_impl(&neg)?; // 1, 0, -1
        Ok(vec![grad_output.mul_impl(&sign)?])
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
        // mask = (x > 0) → 1.0 or 0.0
        let zeros = CudaTensor::zeros(x.shape(), DType::F32);
        let mask = x.gt_impl(&zeros)?;
        Ok(vec![grad_output.mul_impl(&mask)?])
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
        // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let sig = x.sigmoid_impl()?;
        let one_minus_sig = sig.neg_impl()?.add_scalar_impl(1.0)?;
        let x_term = x.mul_impl(&one_minus_sig)?;
        let bracket = x_term.add_scalar_impl(1.0)?;
        let grad_silu = sig.mul_impl(&bracket)?;
        Ok(vec![grad_output.mul_impl(&grad_silu)?])
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
        // grad_output (scalar [1]) を input shape に broadcast
        let ones = CudaTensor::ones(x.shape(), DType::F32);
        Ok(vec![ones.mul_impl(grad_output)?])
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
        // grad_output / n を input shape に broadcast
        let ones = CudaTensor::ones(x.shape(), DType::F32);
        let scaled = grad_output.mul_scalar_impl(1.0 / n)?;
        Ok(vec![ones.mul_impl(&scaled)?])
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
        // grad_output は sum(dim) で dim 次元が消えている
        // unsqueeze(dim) → broadcast_to(input_shape) で元の shape に展開
        let expanded = grad_output.unsqueeze_impl(self.dim)?;
        Ok(vec![expanded.broadcast_to_impl(input_shape)?])
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
