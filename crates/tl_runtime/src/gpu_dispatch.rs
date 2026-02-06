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
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::add(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで減算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_sub(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::sub(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで乗算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_mul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
    let ma = to_metal(a)?;
    let mb = to_metal(b)?;
    let result = GpuOps::mul(&ma, &mb);
    from_metal(&result, device)
}

/// Metal バックエンドで除算
#[cfg(feature = "tl_metal_backend")]
pub fn metal_div(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor, String> {
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
