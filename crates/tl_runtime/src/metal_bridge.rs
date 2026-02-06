//! tl_metal バックエンドへのブリッジ
//!
//! OpaqueTensor と MetalTensor の間の変換を提供する。

#[cfg(feature = "tl_metal_backend")]
use tl_metal::{DType as MetalDType, MetalTensor};

use candle_core::{DType, Tensor};

/// Candle DType から tl_metal DType への変換
#[cfg(feature = "tl_metal_backend")]
fn candle_to_metal_dtype(dtype: DType) -> MetalDType {
    match dtype {
        DType::F32 => MetalDType::F32,
        DType::F16 => MetalDType::F16,
        DType::I64 => MetalDType::I64,
        DType::U8 => MetalDType::U8,
        _ => unimplemented!("Unsupported dtype: {:?}", dtype),
    }
}

/// Candle Tensor から MetalTensor を作成
#[cfg(feature = "tl_metal_backend")]
pub fn tensor_to_metal(tensor: &Tensor) -> Result<MetalTensor, String> {
    let shape = tensor.dims();
    let dtype = candle_to_metal_dtype(tensor.dtype());
    
    // Tensor データを CPU に取得
    let data = tensor
        .flatten_all()
        .map_err(|e| e.to_string())?
        .to_vec1::<f32>()
        .map_err(|e| e.to_string())?;
    
    Ok(MetalTensor::from_slice(&data, shape, dtype))
}

/// MetalTensor から Candle Tensor を作成
#[cfg(feature = "tl_metal_backend")]
pub fn metal_to_tensor(metal_tensor: &MetalTensor, device: &candle_core::Device) -> Result<Tensor, String> {
    let data: Vec<f32> = metal_tensor.to_vec();
    let shape = metal_tensor.shape();
    
    Tensor::from_vec(data, shape, device).map_err(|e| e.to_string())
}

/// tl_metal バックエンドを使用して二項演算を実行
#[cfg(feature = "tl_metal_backend")]
pub fn metal_binary_op(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    device: &candle_core::Device,
) -> Result<Tensor, String> {
    let ma = tensor_to_metal(a)?;
    let mb = tensor_to_metal(b)?;
    
    let result = match op {
        "add" => ma.add(&mb),
        "mul" => ma.mul(&mb),
        _ => return Err(format!("Unknown op: {}", op)),
    };
    
    metal_to_tensor(&result, device)
}

/// tl_metal バックエンドを使用して matmul を実行
#[cfg(feature = "tl_metal_backend")]
pub fn metal_matmul(
    a: &Tensor,
    b: &Tensor,
    device: &candle_core::Device,
) -> Result<Tensor, String> {
    let ma = tensor_to_metal(a)?;
    let mb = tensor_to_metal(b)?;
    
    let result = ma.matmul(&mb);
    
    metal_to_tensor(&result, device)
}
