//! GPU バックエンドディスパッチャ
//!
//! トレイト経由で Metal (将来的には CUDA) バックエンドを呼び出す。

#[cfg(feature = "tl_metal_backend")]
use tl_backend::{GpuTensor, GpuOps, DType as BackendDType};
#[cfg(feature = "tl_metal_backend")]
use tl_metal::MetalTensor;

use candle_core::{DType, Tensor, Device};

/// Candle DType から Backend DType への変換
#[cfg(feature = "tl_metal_backend")]
fn candle_to_backend_dtype(dtype: DType) -> BackendDType {
    match dtype {
        DType::F32 => BackendDType::F32,
        DType::F16 => BackendDType::F16,
        DType::I64 => BackendDType::I64,
        DType::U8 => BackendDType::U8,
        _ => BackendDType::F32,
    }
}

/// Backend DType から Candle DType への変換
#[cfg(feature = "tl_metal_backend")]
fn backend_to_candle_dtype(dtype: BackendDType) -> DType {
    match dtype {
        BackendDType::F32 => DType::F32,
        BackendDType::F16 => DType::F16,
        BackendDType::I64 => DType::I64,
        BackendDType::U8 => DType::U8,
        _ => DType::F32,
    }
}

/// Candle Tensor から MetalTensor への変換
#[cfg(feature = "tl_metal_backend")]
pub fn to_metal(tensor: &Tensor) -> Result<MetalTensor, String> {
    let shape = tensor.dims();
    let dtype = candle_to_backend_dtype(tensor.dtype());
    
    // F32 として取得
    let data = tensor
        .flatten_all()
        .map_err(|e| e.to_string())?
        .to_vec1::<f32>()
        .map_err(|e| e.to_string())?;
    
    Ok(MetalTensor::from_slice(&data, shape, tl_metal::DType::F32))
}

/// MetalTensor から Candle Tensor への変換
#[cfg(feature = "tl_metal_backend")]
pub fn from_metal(metal: &MetalTensor, device: &Device) -> Result<Tensor, String> {
    let data: Vec<f32> = metal.to_vec_f32();
    let shape = metal.shape();
    Tensor::from_vec(data, shape, device).map_err(|e| e.to_string())
}

// ========== 二項演算 ==========

/// Metal バックエンドで加算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_add(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    // Shape が異なる場合は Candle にフォールバック
    if a.dims() != b.dims() {
        return Err("Shape mismatch, fallback to Candle".into());
    }
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::add(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで減算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sub(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    if a.dims() != b.dims() {
        return Err("Shape mismatch, fallback to Candle".into());
    }
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::sub(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで乗算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_mul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    if a.dims() != b.dims() {
        return Err("Shape mismatch, fallback to Candle".into());
    }
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::mul(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで除算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_div(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    if a.dims() != b.dims() {
        return Err("Shape mismatch, fallback to Candle".into());
    }
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::div(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで matmul
#[cfg(feature = "tl_metal_backend")]
pub fn metal_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::matmul(&ma, &mb);
    from_metal(&result, device)
}

// ========== 単項演算 ==========

/// Metal バックエンドで exp
#[cfg(feature = "tl_metal_backend")]
pub fn metal_exp(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::exp(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで relu
#[cfg(feature = "tl_metal_backend")]
pub fn metal_relu(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::relu(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで softmax
#[cfg(feature = "tl_metal_backend")]
pub fn metal_softmax(t: &Tensor, dim: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::softmax(&mt, dim);
    from_metal(&result, device)
}

/// Metal バックエンドで sigmoid
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sigmoid(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::sigmoid(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで tanh
#[cfg(feature = "tl_metal_backend")]
pub fn metal_tanh(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::tanh(&mt);
    from_metal(&result, device)
}

// ========== Reduce 演算 ==========

/// Metal バックエンドで sumall
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sumall(t: &Tensor, device: &Device) -> Result<f32, String> {
    let mt = to_metal(t)?;
    Ok(GpuOps::sumall(&mt))
}

/// Metal バックエンドで sum(axis)
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sum(t: &Tensor, axis: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::sum(&mt, axis);
    from_metal(&result, device)
}

/// Metal バックエンドで argmax
#[cfg(feature = "tl_metal_backend")]
pub fn metal_argmax(t: &Tensor, axis: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::argmax(&mt, axis);
    from_metal(&result, device)
}

// ========== 形状操作 ==========

/// Metal バックエンドで reshape
#[cfg(feature = "tl_metal_backend")]
pub fn metal_reshape(t: &Tensor, shape: &[usize], device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::reshape(&mt, shape);
    from_metal(&result, device)
}

/// Metal バックエンドで transpose
#[cfg(feature = "tl_metal_backend")]
pub fn metal_transpose(t: &Tensor, dim0: usize, dim1: usize, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::transpose(&mt, dim0, dim1);
    from_metal(&result, device)
}

// ========== 生成 ==========

/// Metal バックエンドで randn
#[cfg(feature = "tl_metal_backend")]
pub fn metal_randn(shape: &[usize], device: &Device) -> Result<Tensor, String> {
    let mt = MetalTensor::randn(shape, tl_metal::DType::F32);
    from_metal(&mt, device)
}

/// Metal バックエンドで zeros
#[cfg(feature = "tl_metal_backend")]
pub fn metal_zeros(shape: &[usize], device: &Device) -> Result<Tensor, String> {
    let mt = MetalTensor::zeros(shape, tl_metal::DType::F32);
    from_metal(&mt, device)
}

/// Metal バックエンドで ones
#[cfg(feature = "tl_metal_backend")]
pub fn metal_ones(shape: &[usize], device: &Device) -> Result<Tensor, String> {
    let mt = MetalTensor::ones(shape, tl_metal::DType::F32);
    from_metal(&mt, device)
}

// ========== 追加単項演算 ==========

/// Metal バックエンドで neg
#[cfg(feature = "tl_metal_backend")]
pub fn metal_neg(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::neg(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで log
#[cfg(feature = "tl_metal_backend")]
pub fn metal_log(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::log(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで sqrt
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sqrt(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::sqrt(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで abs
#[cfg(feature = "tl_metal_backend")]
pub fn metal_abs(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::abs(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで sin
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sin(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::sin(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで cos
#[cfg(feature = "tl_metal_backend")]
pub fn metal_cos(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::cos(&mt);
    from_metal(&result, device)
}

/// Metal バックエンドで gelu
#[cfg(feature = "tl_metal_backend")]
pub fn metal_gelu(t: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::gelu(&mt);
    from_metal(&result, device)
}

// ========== 二項演算追加 ==========

/// Metal バックエンドで pow
#[cfg(feature = "tl_metal_backend")]
pub fn metal_pow(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::pow(&ma, &mb);
    from_metal(&result, device)
}

// ========== スカラー演算 ==========

/// Metal バックエンドで add_scalar
#[cfg(feature = "tl_metal_backend")]
pub fn metal_add_scalar(t: &Tensor, scalar: f32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::add_scalar(&mt, scalar);
    from_metal(&result, device)
}

/// Metal バックエンドで mul_scalar
#[cfg(feature = "tl_metal_backend")]
pub fn metal_mul_scalar(t: &Tensor, scalar: f32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::mul_scalar(&mt, scalar);
    from_metal(&result, device)
}

/// Metal バックエンドで clamp
#[cfg(feature = "tl_metal_backend")]
pub fn metal_clamp(t: &Tensor, min: f32, max: f32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::clamp(&mt, min, max);
    from_metal(&result, device)
}

// ========== Reduce 追加 ==========

/// Metal バックエンドで mean
#[cfg(feature = "tl_metal_backend")]
pub fn metal_mean(t: &Tensor, axis: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::mean(&mt, axis);
    from_metal(&result, device)
}

/// Metal バックエンドで max
#[cfg(feature = "tl_metal_backend")]
pub fn metal_max(t: &Tensor, axis: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::max(&mt, axis);
    from_metal(&result, device)
}

/// Metal バックエンドで min
#[cfg(feature = "tl_metal_backend")]
pub fn metal_min(t: &Tensor, axis: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::min(&mt, axis);
    from_metal(&result, device)
}

// ========== 形状操作追加 ==========

/// Metal バックエンドで squeeze
#[cfg(feature = "tl_metal_backend")]
pub fn metal_squeeze(t: &Tensor, dim: usize, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::squeeze(&mt, dim);
    from_metal(&result, device)
}

/// Metal バックエンドで unsqueeze
#[cfg(feature = "tl_metal_backend")]
pub fn metal_unsqueeze(t: &Tensor, dim: usize, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::unsqueeze(&mt, dim);
    from_metal(&result, device)
}

/// Metal バックエンドで narrow
#[cfg(feature = "tl_metal_backend")]
pub fn metal_narrow(t: &Tensor, axis: usize, start: usize, len: usize, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::narrow(&mt, axis, start, len);
    from_metal(&result, device)
}

/// Metal バックエンドで broadcast_to
#[cfg(feature = "tl_metal_backend")]
pub fn metal_broadcast_to(t: &Tensor, shape: &[usize], device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::broadcast_to(&mt, shape);
    from_metal(&result, device)
}

// ========== 特殊演算 ==========

/// Metal バックエンドで embedding
#[cfg(feature = "tl_metal_backend")]
pub fn metal_embedding(weight: &Tensor, indices: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mw = to_metal(weight)?;
    let mi = to_metal(indices)?;
    let result = GpuOps::embedding(&mw, &mi);
    from_metal(&result, device)
}

/// Metal バックエンドで tril
#[cfg(feature = "tl_metal_backend")]
pub fn metal_tril(t: &Tensor, diagonal: i32, device: &Device) -> Result<Tensor, String> {
    let mt = to_metal(t)?;
    let result = GpuOps::tril(&mt, diagonal);
    from_metal(&result, device)
}

/// Metal バックエンドで cross_entropy
#[cfg(feature = "tl_metal_backend")]
pub fn metal_cross_entropy(pred: &Tensor, target: &Tensor, device: &Device) -> Result<Tensor, String> {
    let mp = to_metal(pred)?;
    let mt = to_metal(target)?;
    let result = GpuOps::cross_entropy(&mp, &mt);
    from_metal(&result, device)
}

