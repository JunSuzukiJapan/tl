//! llm スタブモジュール
//!
//! tensor_ops_ext と system からの re-export でコンパイラの互換性を維持。

use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};

// tl_metal::ffi_ops から直接使用
use tl_metal::ffi_ops::tl_metal_cat;

// system からの kv_cache 関数を re-export
pub use crate::system::{
    tl_kv_cache_new, tl_kv_cache_free,
    tl_kv_cache_get_k, tl_kv_cache_get_v,
    tl_kv_cache_update,
};

// tokenizer 関数を re-export（存在する関数のみ）
pub use crate::tokenizer::{
    tl_tokenizer_new, tl_tokenizer_encode, tl_tokenizer_decode, tl_tokenizer_encode_chat,
};


// tensor_cat は lib.rs で定義済み - llm モジュールからは呼び出しラッパー

/// tensor_cat2 - 2 テンソル連結（dim=0）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat2(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    tl_metal_cat(a, b, 0)
}

/// tensor_cat_4d - 4D テンソル連結
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_4d(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_metal_cat(a, b, dim)
}





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
    crate::make_metal_tensor(result)
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
    crate::make_metal_tensor(result)
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
    crate::make_metal_tensor(result)
}

// tl_gguf_load と tl_tokenizer_new_from_gguf は system.rs で定義

// --- KVCache ABI Wrappers for Compiler Auto-Generation ---
// Compiler generates calls using pointer to struct (ptr), while runtime expects i64 handle directly.
// These wrappers bridge the gap.

#[unsafe(no_mangle)]
pub extern "C" fn tl_kvcache_free(ptr: *const i64) {
    if ptr.is_null() { return; }
    let handle = unsafe { *ptr };
    tl_kv_cache_free(handle);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kvcache_new(sret: *mut i64, layers: i64) {
    let handle = tl_kv_cache_new(layers);
    unsafe { *sret = handle };
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kvcache_get_k(ptr: *const i64, layer: i64) -> *mut OpaqueTensor {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let handle = unsafe { *ptr };
    tl_kv_cache_get_k(handle, layer)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kvcache_get_v(ptr: *const i64, layer: i64) -> *mut OpaqueTensor {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let handle = unsafe { *ptr };
    tl_kv_cache_get_v(handle, layer)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kvcache_update(ptr: *const i64, layer: i64, k: *mut OpaqueTensor, v: *mut OpaqueTensor) {
    if ptr.is_null() { return; }
    let handle = unsafe { *ptr };
    tl_kv_cache_update(handle, layer, k, v);
}
