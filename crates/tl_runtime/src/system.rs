//! System 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::io::Write;

/// スリープ（ミリ秒）
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_sleep(ms: i64) {
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }
}

/// 現在時刻（Unix タイムスタンプ、秒）
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 標準入力から行を読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_read_line() -> *mut StringStruct {
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(_) => {
            let trimmed = input.trim_end();
            let c_str = CString::new(trimmed).unwrap_or_else(|_| CString::new("").unwrap());
            let ptr = c_str.into_raw();
            unsafe {
                let len = libc::strlen(ptr) as i64;
                let layout = std::alloc::Layout::new::<StringStruct>();
                let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
                (*struct_ptr).ptr = ptr;
                (*struct_ptr).len = len;
                struct_ptr
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// プロンプト表示して入力を読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_prompt(prompt: *const c_char) -> *mut StringStruct {
    if !prompt.is_null() {
        let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy() };
        print!("{}", prompt_str);
        let _ = std::io::stdout().flush();
    }
    tl_read_line()
}

/// デバイス設定（Metal のみなのでスタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_set_device(_device_id: i64) {
    // Metal バックエンドでは単一デバイスのため何もしない
}

/// VarBuilder 関連（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get(_name: *mut StringStruct) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder not yet supported in Metal backend");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get_from_tensor(_tensor: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder not yet supported in Metal backend");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_grad(_name: *mut StringStruct) -> *mut crate::OpaqueTensor {
    eprintln!("Warning: VarBuilder gradients not yet supported in Metal backend");
    std::ptr::null_mut()
}

/// パラメータ関連（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_add_parameter(_name: *mut StringStruct, _t: *mut crate::OpaqueTensor) {
    // スタブ
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_register_parameter(_t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    // スタブ - そのまま返す
    _t
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_save_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter saving not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_load_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter loading not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_update_all_params(lr: f32) {
    // REGISTRY 廃止: テンソルのパラメータ更新は TL コード側で直接実行
    let _ = lr;
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_clear_grads() {
    // REGISTRY 廃止: テンソルの勾配クリアはスコープ離脱時に自動
}

/// GGUF ロード（candle-core 実装）
#[unsafe(no_mangle)]
pub extern "C" fn tl_gguf_load(path: *mut StringStruct) -> *mut crate::tensor_map::OpaqueTensorMap {
    use candle_core::quantized::gguf_file;
    use std::collections::HashMap;
    use std::sync::Arc;

    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
             return std::ptr::null_mut();
        }
        let path_slice = std::slice::from_raw_parts((*path).ptr as *const u8, (*path).len as usize);
        let path_str = String::from_utf8_lossy(path_slice).into_owned();
        let expanded = crate::file_io::expand_path(&path_str);

        // GGUF ファイルを読み込む
        let mut file = match std::fs::File::open(&expanded) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Failed to open GGUF file {:?}: {}", expanded, e);
                return std::ptr::null_mut();
            }
        };

        let content = match gguf_file::Content::read(&mut file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: Failed to parse GGUF file {:?}: {}", expanded, e);
                return std::ptr::null_mut();
            }
        };

        eprintln!(
            "[tl_gguf_load] GGUF contains {} tensors",
            content.tensor_infos.len()
        );

        // Metal デバイスを使用
        let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);

        // テンソルをロードし、OpaqueTensorMap に格納
        let entries = HashMap::new();
        let mut qtensor_map = HashMap::new();

        for (name, _) in content.tensor_infos.iter() {
            match content.tensor(&mut file, name, &device) {
                Ok(qtensor) => {
                    // QTensor を量子化テンソルマップに保存
                    qtensor_map.insert(name.clone(), Arc::new(qtensor));
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to load tensor '{}': {}",
                        name, e
                    );
                }
            }
        }

        let _total = entries.len() + qtensor_map.len();
        eprintln!(
            "[tl_gguf_load] Loaded {} regular + {} quantized tensors",
            entries.len(),
            qtensor_map.len()
        );

        let map = crate::tensor_map::OpaqueTensorMap {
            entries,
            qtensors: qtensor_map,
        };

        Box::into_raw(Box::new(map))
    }
}

/// QTensor 解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_free(ptr: usize) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(
            ptr as *mut candle_core::quantized::QTensor,
        );
    }
}

/// QTensor matmul: input_tensor × quantized_weight
/// QTensor をデクォンタイズしてから通常の matmul を実行
/// コンパイラシグネチャ: (void_ptr, void_ptr) -> void_ptr
#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_matmul(
    input: *mut crate::OpaqueTensor,
    weight: *mut candle_core::quantized::QTensor,
) -> *mut crate::OpaqueTensor {
    if input.is_null() || weight.is_null() {
        eprintln!("Error: tl_qtensor_matmul received null argument");
        return std::ptr::null_mut();
    }

    unsafe {
        let input_tensor = &*input;
        let qtensor = &*weight;

        let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);

        // QTensor をデクォンタイズ
        let weight_tensor = match qtensor.dequantize(&device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error: tl_qtensor_matmul dequantize failed: {}", e);
                return std::ptr::null_mut();
            }
        };

        // MetalTensor -> candle Tensor
        let input_data: Vec<f32> = input_tensor.to_vec();
        let input_shape: Vec<usize> = tl_metal::MetalTensor::shape(input_tensor).to_vec();

        let candle_input = match candle_core::Tensor::from_vec(
            input_data,
            input_shape.as_slice(),
            &device,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error: Failed to create candle tensor: {}", e);
                return std::ptr::null_mut();
            }
        };

        // matmul: input × weight^T
        let result = match candle_input.matmul(&weight_tensor.t().unwrap_or(weight_tensor)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error: tl_qtensor_matmul matmul failed: {}", e);
                return std::ptr::null_mut();
            }
        };

        // candle Tensor -> MetalTensor
        let result_shape: Vec<usize> = result.dims().to_vec();
        let result_data: Vec<f32> = match result.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error: Failed to extract result data: {}", e);
                return std::ptr::null_mut();
            }
        };
        let metal_result =
            tl_metal::MetalTensor::from_slice(&result_data, &result_shape, tl_metal::DType::F32);
        Box::into_raw(Box::new(metal_result))
    }
}

/// KV Cache 関連 — layer ベース実装
/// LLVM 宣言: new(layers: i64) → i64, get_k/get_v(ptr: i64, layer: i64) → Tensor,
///            update(ptr: i64, layer: i64, k: ptr, v: ptr), free(ptr: i64)
pub struct OpaqueKVCache {
    pub layers: Vec<(Option<*mut crate::OpaqueTensor>, Option<*mut crate::OpaqueTensor>)>,
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_new(num_layers: i64) -> i64 {
    let n = num_layers.max(1) as usize;
    let cache = Box::new(OpaqueKVCache {
        layers: vec![(None, None); n],
    });
    Box::into_raw(cache) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_free(cache_ptr: i64) {
    if cache_ptr == 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(cache_ptr as *mut OpaqueKVCache);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_k(cache_ptr: i64, layer: i64) -> *mut crate::OpaqueTensor {
    if cache_ptr == 0 {
        return std::ptr::null_mut();
    }
    let cache = unsafe { &*(cache_ptr as *mut OpaqueKVCache) };
    let idx = layer as usize;
    if idx < cache.layers.len() {
        cache.layers[idx].0.unwrap_or(std::ptr::null_mut())
    } else {
        std::ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_v(cache_ptr: i64, layer: i64) -> *mut crate::OpaqueTensor {
    if cache_ptr == 0 {
        return std::ptr::null_mut();
    }
    let cache = unsafe { &*(cache_ptr as *mut OpaqueKVCache) };
    let idx = layer as usize;
    if idx < cache.layers.len() {
        cache.layers[idx].1.unwrap_or(std::ptr::null_mut())
    } else {
        std::ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_update(
    cache_ptr: i64,
    layer: i64,
    k: *mut crate::OpaqueTensor,
    v: *mut crate::OpaqueTensor,
) {
    if cache_ptr == 0 {
        return;
    }
    let cache = unsafe { &mut *(cache_ptr as *mut OpaqueKVCache) };
    let idx = layer as usize;
    // layer がキャパシティを超えた場合は拡張
    while cache.layers.len() <= idx {
        cache.layers.push((None, None));
    }
    cache.layers[idx] = (Some(k), Some(v));
}


// ========== 追加 System 関数 ==========

// tl_checkpoint と tl_trace_mem は memory_ffi.rs で定義済み

// tl_hash_string は string_ffi.rs で定義済み

// tl_http_get と tl_http_download は file_io.rs で定義済み

// tl_download_file は file_io.rs で定義済み

/// Metal 同期
#[unsafe(no_mangle)]
pub extern "C" fn tl_metal_sync() {
    // Metal バックエンドの同期（現在は何もしない）
}

// tl_register_tensor は memory_ffi.rs で定義済み

/// ランタイムエラー報告
#[unsafe(no_mangle)]
pub extern "C" fn tl_report_runtime_error(msg: *mut StringStruct) {
    if !msg.is_null() {
        unsafe {
            if !(*msg).ptr.is_null() {
                let c_str = CStr::from_ptr((*msg).ptr);
                eprintln!("Runtime error: {}", c_str.to_string_lossy());
            }
        }
    }
}

/// ランタイムエラーハンドル
#[unsafe(no_mangle)]
pub extern "C" fn tl_handle_runtime_error(msg: *mut StringStruct) {
    tl_report_runtime_error(msg);
}

// tl_log_alloc と tl_log_free は memory_ffi.rs で定義済み

// tl_query は knowledge_base.rs で定義済み
