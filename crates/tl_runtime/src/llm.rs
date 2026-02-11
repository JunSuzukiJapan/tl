//! llm スタブモジュール
//!
//! tensor_ops_ext と system からの re-export でコンパイラの互換性を維持。

use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};

// tensor_ops_ext からの re-export
// tensor_ops_ext からの re-export
pub use crate::tl_tensor_silu;

// system からの kv_cache 関数を re-export
pub use crate::system::{
    tl_kv_cache_new, tl_kv_cache_free,
    tl_kv_cache_get_k, tl_kv_cache_get_v,
    tl_kv_cache_update,
};

// tokenizer 関数を re-export（存在する関数のみ）
pub use crate::tokenizer::{
    tl_tokenizer_new, tl_tokenizer_encode, tl_tokenizer_decode,
};



// tensor_ops_ext から LLM 関連関数を re-export (silu は lib.rs でエクスポート済み)
pub use crate::{
    tl_tensor_rms_norm, 
    tl_tensor_sample,
    tl_tensor_rope_new_cos, tl_tensor_rope_new_sin, tl_tensor_apply_rope,
};

// tensor_cat は lib.rs で定義済み - llm モジュールからは呼び出しラッパー

/// tensor_cat2 - 2 テンソル連結（dim=0）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat2(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    crate::tl_tensor_cat(a, b, 0)
}

/// tensor_cat_4d - 4D テンソル連結
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_4d(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    crate::tl_tensor_cat(a, b, dim)
}

// tl_tensor_cat を lib.rs から re-export
pub use crate::tl_tensor_cat;



/// RoPE cos キャッシュ作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_rope_cos_cache_new(
    seq_len: i64,
    head_dim: i64,
    base: f64,
) -> *mut OpaqueTensor {
    let mut cos_data = Vec::with_capacity((seq_len * head_dim / 2) as usize);
    for pos in 0..seq_len {
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / (base as f32).powf((2.0 * i as f32) / head_dim as f32);
            cos_data.push((pos as f32 * freq).cos());
        }
    }
    let shape = vec![seq_len as usize, (head_dim / 2) as usize];
    let result = MetalTensor::from_slice(&cos_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

/// RoPE sin キャッシュ作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_rope_sin_cache_new(
    seq_len: i64,
    head_dim: i64,
    base: f64,
) -> *mut OpaqueTensor {
    let mut sin_data = Vec::with_capacity((seq_len * head_dim / 2) as usize);
    for pos in 0..seq_len {
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / (base as f32).powf((2.0 * i as f32) / head_dim as f32);
            sin_data.push((pos as f32 * freq).sin());
        }
    }
    let shape = vec![seq_len as usize, (head_dim / 2) as usize];
    let result = MetalTensor::from_slice(&sin_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

/// RMS Norm
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rms_norm_llm(t: *mut OpaqueTensor, eps: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    let mean_sq: f32 = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;
    let rms = (mean_sq + eps as f32).sqrt();
    let result_data: Vec<f32> = data.iter().map(|&x| x / rms).collect();
    let result = MetalTensor::from_slice(&result_data, tensor.shape(), DType::F32);
    Box::into_raw(Box::new(result))
}

// tl_gguf_load と tl_tokenizer_new_from_gguf は system.rs で定義
