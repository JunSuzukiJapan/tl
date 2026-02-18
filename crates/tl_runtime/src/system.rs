//! System 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{CStr, CString};
use std::io::Write;


/// スリープ（ミリ秒）
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
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
/// 標準入力から行を読み込み (prompt があれば表示)
/// 引数は StringStruct* (TLのString型)
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> String*
pub extern "C" fn tl_read_line(prompt: *mut StringStruct) -> *mut StringStruct {
    if !prompt.is_null() {
        unsafe {
            // StringStruct { ptr: *mut c_char, len: i64 }
            if !(*prompt).ptr.is_null() {
                let prompt_str = CStr::from_ptr((*prompt).ptr).to_string_lossy();
                if !prompt_str.is_empty() {
                    print!("{}", prompt_str);
                    let _ = std::io::stdout().flush();
                }
            }
        }
    } else {
        let _ = std::io::stdout().flush();
    }

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
/// プロンプト表示して入力を読み込み (Legacy wrapper)
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> String*
pub extern "C" fn tl_prompt(prompt: *mut StringStruct) -> *mut StringStruct {
    // tl_read_line がプロンプトを処理するようになったのでそのまま移譲
    tl_read_line(prompt)
}

/// デバイス設定（Metal のみなのでスタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_set_device(_device_id: i64) {
    // Metal バックエンドでは単一デバイスのため何もしない
}

/// VarBuilder 関連（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> *mut crate::OpaqueTensor
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
/// @ffi_sig (String*) -> *mut crate::OpaqueTensor
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
/// @ffi_sig (OpaqueTensor) -> *mut crate::OpaqueTensor
pub extern "C" fn tl_register_parameter(_t: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    // スタブ - そのまま返す
    _t
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_save_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter saving not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_load_all_params(_path: *mut StringStruct) {
    eprintln!("Warning: Parameter loading not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_update_all_params(lr: f32) {
    // REGISTRY 廃止: テンソルのパラメータ更新は TL コード側で直接実行
    let _ = lr;
}

#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_clear_grads() {
    // No-op: デバッグカウンタは削除済み
}

/// QTensor 解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_free(ptr: usize) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(
            ptr as *mut crate::quantized::QTensor,
        );
    }
}

/// QTensor matmul: input_tensor × quantized_weight
/// GPU: 融合カーネル (mul_mv_q4_K_f32) で Q4_K データから直接 matmul
/// CPU: フォールバック (dequantize → F32 matmul)
/// コンパイラシグネチャ: (void_ptr, void_ptr) -> void_ptr
#[unsafe(no_mangle)]
/// @ffi_sig (OpaqueTensor, QTensor) -> *mut crate::OpaqueTensor
pub extern "C" fn tl_qtensor_matmul(
    input: *mut crate::OpaqueTensor,
    weight: *mut crate::quantized::QTensor,
) -> *mut crate::OpaqueTensor {
    if input.is_null() || weight.is_null() {
        eprintln!("Error: tl_qtensor_matmul received null argument");
        return std::ptr::null_mut();
    }

    unsafe {
        let qtensor = &*weight;
        let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");

        if is_cpu {
            // CPU フォールバック: dequantize + transpose + matmul
            let weight_ptr = match qtensor.dequantize_to_tensor() {
                Ok(t) => t as *mut std::ffi::c_void,
                Err(e) => {
                    eprintln!("Error: tl_qtensor_matmul dequantize failed: {}", e);
                    return std::ptr::null_mut();
                }
            };
            // Transposed キャッシュ
            {
                let cache_guard = qtensor.cache_transposed.lock().unwrap();
                if let Some(ptr_val) = *cache_guard {
                    let transposed = ptr_val as *mut std::ffi::c_void;
                    return crate::device_ffi::tl_device_tensor_matmul(
                        input as *mut std::ffi::c_void, transposed
                    ) as *mut crate::OpaqueTensor;
                }
            }
            let transposed = crate::device_ffi::tl_device_tensor_transpose_2d(weight_ptr);
            {
                let mut cache_guard = qtensor.cache_transposed.lock().unwrap();
                *cache_guard = Some(transposed as usize);
            }
            return crate::device_ffi::tl_device_tensor_matmul(
                input as *mut std::ffi::c_void, transposed
            ) as *mut crate::OpaqueTensor;
        }

        // ========== GPU パス ==========
        let shape = &qtensor.shape;
        assert!(shape.len() == 2, "QTensor must be 2D, got {}D", shape.len());
        let n = shape[0]; // output features
        let k = shape[1]; // input features

        // 融合カーネル対応の量子化タイプかチェック
        let use_fused = matches!(
            qtensor.ggml_type,
            crate::quantized::GGMLType::Q4_K | crate::quantized::GGMLType::Q6_K
        );

        if !use_fused {
            // 未対応型: 既存方式 (dequantize → transpose → matmul)
            {
                let cache_guard = qtensor.cache_transposed.lock().unwrap();
                if let Some(ptr_val) = *cache_guard {
                    let transposed = ptr_val as *mut std::ffi::c_void;
                    return crate::device_ffi::tl_device_tensor_matmul(
                        input as *mut std::ffi::c_void, transposed
                    ) as *mut crate::OpaqueTensor;
                }
            }
            let weight_ptr = match qtensor.dequantize_to_tensor() {
                Ok(t) => t as *mut std::ffi::c_void,
                Err(e) => {
                    eprintln!("Error: tl_qtensor_matmul dequantize failed: {}", e);
                    return std::ptr::null_mut();
                }
            };
            let transposed = crate::device_ffi::tl_device_tensor_transpose_2d(weight_ptr);
            {
                let mut cache_guard = qtensor.cache_transposed.lock().unwrap();
                *cache_guard = Some(transposed as usize);
            }
            return crate::device_ffi::tl_device_tensor_matmul(
                input as *mut std::ffi::c_void, transposed
            ) as *mut crate::OpaqueTensor;
        }

        // ========== 融合カーネルパス (Q4_K / Q6_K) ==========
        // GPU raw buffer キャッシュチェック
        let gpu_raw_ptr = {
            let cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
            *cache_guard
        };

        let w_raw_ptr = if let Some(ptr_val) = gpu_raw_ptr {
            ptr_val as *const tl_metal::MetalTensor
        } else {
            let raw_bytes = &qtensor.data;
            let raw_shape = &[raw_bytes.len()];
            let gpu_tensor = tl_metal::MetalTensor::from_slice(
                raw_bytes, raw_shape, tl_metal::DType::U8
            );
            let ptr = Box::into_raw(Box::new(gpu_tensor));
            {
                let mut cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
                *cache_guard = Some(ptr as usize);
            }
            ptr as *const tl_metal::MetalTensor
        };

        let input_mt = &*(input as *const tl_metal::MetalTensor);
        let w_raw_mt = &*w_raw_ptr;

        let result = match qtensor.ggml_type {
            crate::quantized::GGMLType::Q4_K => input_mt.mul_mv_q4_k(w_raw_mt, n, k),
            crate::quantized::GGMLType::Q6_K => input_mt.mul_mv_q6_k(w_raw_mt, n, k),
            _ => unreachable!(),
        };
        
        let result_tensor = match result {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error: tl_qtensor_matmul failed: {}", e);
                return std::ptr::null_mut();
            }
        };
        
        crate::make_metal_tensor(result_tensor)
    }
}




/// KV Cache 関連 — layer ベース実装
/// LLVM 宣言: new(layers: i64) → i64, get_k/get_v(ptr: i64, layer: i64) → Tensor,
///            update(ptr: i64, layer: i64, k: ptr, v: ptr), free(ptr: i64)
pub struct OpaqueKVCache {
    pub layers: Vec<(Option<*mut crate::OpaqueTensor>, Option<*mut crate::OpaqueTensor>)>,
}

impl Drop for OpaqueKVCache {
    fn drop(&mut self) {
        for (k_opt, v_opt) in &self.layers {
            if let Some(k) = k_opt {
                crate::memory_ffi::tl_tensor_release_safe(*k);
            }
            if let Some(v) = v_opt {
                crate::memory_ffi::tl_tensor_release_safe(*v);
            }
        }
    }
}

#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> i64
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
/// @ffi_sig (i64, i64) -> *mut crate::OpaqueTensor
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
/// @ffi_sig (i64, i64) -> *mut crate::OpaqueTensor
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
/// @ffi_sig (i64, i64, OpaqueTensor, OpaqueTensor) -> void
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
    // テンソルをcloneしてcacheが独立コピーを保持する
    // (JITの自動メモリ管理で元テンソルが解放されてもcache内は有効)
    let k_clone = if !k.is_null() {
        Some(crate::device_ffi::tl_device_tensor_clone(
            k as *mut std::ffi::c_void,
        ) as *mut crate::OpaqueTensor)
    } else {
        None
    };
    let v_clone = if !v.is_null() {
        Some(crate::device_ffi::tl_device_tensor_clone(
            v as *mut std::ffi::c_void,
        ) as *mut crate::OpaqueTensor)
    } else {
        None
    };
    cache.layers[idx] = (k_clone, v_clone);
}


// ========== 追加 System 関数 ==========

// tl_checkpoint と tl_trace_mem は memory_ffi.rs で定義済み

// tl_hash_string は string_ffi.rs で定義済み

// tl_http_get と tl_http_download は file_io.rs で定義済み

// tl_download_file は file_io.rs で定義済み

/// Metal 同期 — プロセス終了前に GPU 処理の完了を保証
#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_metal_sync() {
    // デバイスが既に初期化されている場合のみ同期
    // get_device() を呼ぶと未初期化でもデバイスが作られるため、
    // try_get_device() で確認してから同期する
    if let Some(device) = tl_metal::device::try_get_device() {
        let cmd = device.command_queue().new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

// tl_register_tensor は memory_ffi.rs で定義済み

/// ランタイムエラー報告
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
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
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_handle_runtime_error(msg: *mut StringStruct) {
    tl_report_runtime_error(msg);
}

// tl_log_alloc と tl_log_free は memory_ffi.rs で定義済み

// tl_query は knowledge_base.rs で定義済み
