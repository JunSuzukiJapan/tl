//! System 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{c_char, CStr, CString};
use std::io::Write;

/// スリープ（秒、f64）— Rust の std::thread::sleep(Duration::from_secs_f64()) に準拠
#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> void
pub extern "C" fn tl_system_sleep(secs: f64) {
    if secs > 0.0 {
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
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

/// VarBuilder: グローバルパラメータレジストリ
/// 名前→テンソルポインタの HashMap で管理。
/// get は初回呼び出し時に randn(req_grad=true) で GradTensor を作成し、
/// 以降は同じポインタを返す。
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::LazyLock;

/// *mut OpaqueTensor を Send/Sync 対応にするラッパー。
/// JIT はシングルスレッド実行なので、FFI 境界を越えるポインタ共有は安全。
#[derive(Clone, Copy)]
struct TensorPtr(*mut crate::OpaqueTensor);
unsafe impl Send for TensorPtr {}
unsafe impl Sync for TensorPtr {}

static VAR_REGISTRY: LazyLock<Mutex<HashMap<String, TensorPtr>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[unsafe(no_mangle)]
/// @ffi_sig (*const i8, i64, *const i64) -> *mut OpaqueTensor
pub extern "C" fn tl_varbuilder_get(
    name: *const std::ffi::c_char,
    rank: i64,
    shape: *const i64,
) -> *mut crate::OpaqueTensor {
    if name.is_null() || shape.is_null() || rank <= 0 {
        eprintln!("[VarBuilder::get] Invalid arguments");
        return std::ptr::null_mut();
    }

    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .to_string();

    // レジストリを確認: 既存のパラメータがあればそのまま返す
    {
        let registry = VAR_REGISTRY.lock().unwrap();
        if let Some(&TensorPtr(ptr)) = registry.get(&name_str) {
            return ptr;
        }
    }

    // 新規作成: randn(shape, req_grad=true) で GradTensor を作成
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank as usize) };
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&d| d as usize).collect();

    let ptr = crate::device_ffi::tl_device_tensor_randn_debug(
        rank as usize,
        shape_usize.as_ptr(),
        0,     // seed = 0 (random)
        true,  // req_grad = true → GradTensor
    ) as *mut crate::OpaqueTensor;

    if ptr.is_null() {
        eprintln!("[VarBuilder::get] Failed to create tensor for '{}'", name_str);
        return std::ptr::null_mut();
    }

    // レジストリに登録
    {
        let mut registry = VAR_REGISTRY.lock().unwrap();
        registry.insert(name_str, TensorPtr(ptr));
    }

    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get_from_tensor(
    _name: *const std::ffi::c_char,
    _tensor: *mut crate::OpaqueTensor,
) -> *mut crate::OpaqueTensor {
    // shape テンソルからの取得（未使用パス）
    eprintln!("Warning: tl_varbuilder_get_from_tensor not yet implemented");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
/// @ffi_sig (*const i8) -> *mut OpaqueTensor
pub extern "C" fn tl_varbuilder_grad(
    name: *const std::ffi::c_char,
) -> *mut crate::OpaqueTensor {
    if name.is_null() {
        eprintln!("[VarBuilder::grad] Null name");
        return std::ptr::null_mut();
    }

    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .to_string();

    // レジストリからテンソルを取得
    let tensor_ptr = {
        let registry = VAR_REGISTRY.lock().unwrap();
        registry.get(&name_str).copied()
    };

    match tensor_ptr {
        Some(TensorPtr(ptr)) if !ptr.is_null() => {
            // テンソルの勾配を取得: device_ffi の grad 関数を使う
            crate::device_ffi::tl_device_tensor_grad(ptr as *mut std::ffi::c_void)
                as *mut crate::OpaqueTensor
        }
        _ => {
            eprintln!("[VarBuilder::grad] Parameter '{}' not found in registry", name_str);
            std::ptr::null_mut()
        }
    }
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
    // 毎ステップの最後にメモリプールを解放し、未使用メモリをOSに返却する
    crate::device_ffi::tl_device_mem_purge();
}

/// QTensor 解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_free(ptr: usize) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let _ = std::sync::Arc::from_raw(ptr as *mut crate::quantized::QTensor);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_retain(ptr: *mut crate::quantized::QTensor) -> *mut crate::quantized::QTensor {
    if ptr.is_null() {
        return ptr;
    }
    unsafe {
        let arc = std::sync::Arc::from_raw(ptr);
        let cloned = std::sync::Arc::clone(&arc);
        let _ = std::sync::Arc::into_raw(arc);
        std::sync::Arc::into_raw(cloned) as *mut crate::quantized::QTensor
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_release_safe(ptr: *mut crate::quantized::QTensor) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = std::sync::Arc::from_raw(ptr);
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
                        input as *mut std::ffi::c_void,
                        transposed,
                    ) as *mut crate::OpaqueTensor;
                }
            }
            let transposed = crate::device_ffi::tl_device_tensor_transpose_2d(weight_ptr);
            {
                let mut cache_guard = qtensor.cache_transposed.lock().unwrap();
                *cache_guard = Some(transposed as usize);
            }
            return crate::device_ffi::tl_device_tensor_matmul(
                input as *mut std::ffi::c_void,
                transposed,
            ) as *mut crate::OpaqueTensor;
        }

        // ========== GPU パス ==========
        let shape = &qtensor.shape;
        assert!(shape.len() == 2, "QTensor must be 2D, got {}D", shape.len());
        let _n = shape[0]; // output features
        let _k = shape[1]; // input features

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
                        input as *mut std::ffi::c_void,
                        transposed,
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
                input as *mut std::ffi::c_void,
                transposed,
            ) as *mut crate::OpaqueTensor;
        }

        // ========== 融合カーネルパス (Q4_K / Q6_K) ==========
        // Metal 固有: mul_mv_q4_k / mul_mv_q6_k カーネル
        #[cfg(target_os = "macos")]
        {
            // GPU raw buffer キャッシュチェック
            let gpu_raw_ptr = {
                let cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
                *cache_guard
            };

            let w_raw_ptr = if let Some(ptr_val) = gpu_raw_ptr {
                ptr_val as *const tl_metal::MetalTensor
            } else {
                // device_ffi ヘルパーで U8 テンソルを GPU にアップロード
                let raw_bytes = &qtensor.data;
                let raw_shape = &[raw_bytes.len()];
                let ptr = crate::device_ffi::create_runtime_tensor_u8(raw_bytes, raw_shape);
                {
                    let mut cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
                    *cache_guard = Some(ptr as usize);
                }
                ptr as *const tl_metal::MetalTensor
            };

            let input_mt = &*(input as *const tl_metal::MetalTensor);
            let w_raw_mt = &*w_raw_ptr;

            let result = match qtensor.ggml_type {
                crate::quantized::GGMLType::Q4_K => input_mt.mul_mv_q4_k(w_raw_mt, _n, _k),
                crate::quantized::GGMLType::Q6_K => input_mt.mul_mv_q6_k(w_raw_mt, _n, _k),
                _ => unreachable!(),
            };

            let result_tensor = match result {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: tl_qtensor_matmul failed: {}", e);
                    return std::ptr::null_mut();
                }
            };

            return crate::make_tensor(result_tensor);
        }

        // CUDA: 融合カーネル (Q4_K / Q6_K) またはフォールバック
        #[cfg(not(target_os = "macos"))]
        {
            if use_fused {
                // ========== CUDA 融合カーネルパス ==========
                // raw Q4_K/Q6_K bytes を GPU にアップロード (キャッシュ)
                let gpu_raw_ptr = {
                    let cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
                    *cache_guard
                };
                let w_raw_ptr = if let Some(ptr_val) = gpu_raw_ptr {
                    ptr_val as *mut std::ffi::c_void
                } else {
                    let raw_bytes = &qtensor.data;
                    let raw_shape = &[raw_bytes.len()];
                    let ptr = crate::device_ffi::create_runtime_tensor_u8(raw_bytes, raw_shape);
                    {
                        let mut cache_guard = qtensor.gpu_raw_cache.lock().unwrap();
                        *cache_guard = Some(ptr as usize);
                    }
                    ptr
                };

                // 融合 matmul カーネル呼び出し
                let result = match qtensor.ggml_type {
                    crate::quantized::GGMLType::Q4_K => crate::device_ffi::cuda_mul_mv_q4_k(
                        input as *mut std::ffi::c_void,
                        w_raw_ptr,
                        _n as i64,
                        _k as i64,
                    ),
                    crate::quantized::GGMLType::Q6_K => crate::device_ffi::cuda_mul_mv_q6_k(
                        input as *mut std::ffi::c_void,
                        w_raw_ptr,
                        _n as i64,
                        _k as i64,
                    ),
                    _ => unreachable!(),
                };

                return result as *mut crate::OpaqueTensor;
            }

            // ========== 非融合フォールバック (dequantize → matmul) ==========
            let weight_ptr = match qtensor.dequantize_to_tensor() {
                Ok(t) => t as *mut std::ffi::c_void,
                Err(e) => {
                    eprintln!("Error: tl_qtensor_matmul dequantize failed: {}", e);
                    return std::ptr::null_mut();
                }
            };
            {
                let cache_guard = qtensor.cache_transposed.lock().unwrap();
                if let Some(ptr_val) = *cache_guard {
                    let transposed = ptr_val as *mut std::ffi::c_void;
                    return crate::device_ffi::tl_device_tensor_matmul(
                        input as *mut std::ffi::c_void,
                        transposed,
                    ) as *mut crate::OpaqueTensor;
                }
            }
            let transposed = crate::device_ffi::tl_device_tensor_transpose_2d(weight_ptr);
            {
                let mut cache_guard = qtensor.cache_transposed.lock().unwrap();
                *cache_guard = Some(transposed as usize);
            }
            return crate::device_ffi::tl_device_tensor_matmul(
                input as *mut std::ffi::c_void,
                transposed,
            ) as *mut crate::OpaqueTensor;
        }
    }
}

/// KV Cache 関連 — layer ベース実装
/// LLVM 宣言: new(layers: i64) → i64, get_k/get_v(ptr: i64, layer: i64) → Tensor,
///            update(ptr: i64, layer: i64, k: ptr, v: ptr), free(ptr: i64)
pub struct OpaqueKVCache {
    pub layers: Vec<(
        Option<*mut crate::OpaqueTensor>,
        Option<*mut crate::OpaqueTensor>,
    )>,
}

/// テンソルポインタに対して Arc RC+1 を行い、新しいポインタを返す。
/// CPU/GPU 両対応: is_cpu() で正しい型にキャストしてから Arc 操作を行う。
/// macOS では OpaqueTensor = MetalTensor だが、--device cpu 時の実テンソルは
/// Arc<UnsafeCell<CpuTensor<f32>>> のため、型を正しく選択する必要がある。
fn tensor_arc_retain(ptr: *mut crate::OpaqueTensor) -> *mut crate::OpaqueTensor {
    if ptr.is_null() {
        return ptr;
    }
    if crate::device_ffi::is_cpu() {
        // CPU: テンソルは Arc<UnsafeCell<CpuTensor<f32>>> で管理
        unsafe {
            let arc = std::sync::Arc::from_raw(
                ptr as *const std::cell::UnsafeCell<tl_cpu::CpuTensor<f32>>,
            );
            let cloned = std::sync::Arc::clone(&arc);
            let _ = std::sync::Arc::into_raw(arc);
            std::sync::Arc::into_raw(cloned) as *mut crate::OpaqueTensor
        }
    } else {
        // GPU: テンソルは Arc<UnsafeCell<OpaqueTensor>> で管理
        unsafe {
            let arc = std::sync::Arc::from_raw(
                ptr as *const std::cell::UnsafeCell<crate::OpaqueTensor>,
            );
            let cloned = std::sync::Arc::clone(&arc);
            let _ = std::sync::Arc::into_raw(arc);
            std::sync::Arc::into_raw(cloned) as *mut crate::OpaqueTensor
        }
    }
}

/// CPU/GPU 両対応のテンソル解放ヘルパー。
/// Arc の参照カウントを -1 する。RC=0 になればテンソルが Drop される。
/// memory_ffi::tl_tensor_release_safe 経由に統一し、CPU メモリ統計も正しく更新。
fn release_tensor_safe(ptr: *mut crate::OpaqueTensor) {
    if ptr.is_null() {
        return;
    }
    // 全デバイス統一: memory_ffi 経由で解放（CPU 統計カウンタも更新される）
    crate::memory_ffi::tl_tensor_release_safe(ptr);
}

impl Drop for OpaqueKVCache {
    fn drop(&mut self) {
        for (k_opt, v_opt) in &self.layers {
            if let Some(k) = k_opt {
                if !k.is_null() {
                    release_tensor_safe(*k);
                }
            }
            if let Some(v) = v_opt {
                if !v.is_null() {
                    release_tensor_safe(*v);
                }
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
/// @ffi_sig (i64) -> i64
pub extern "C" fn tl_kv_cache_len(cache_ptr: i64) -> i64 {
    if cache_ptr == 0 {
        return 0;
    }
    let cache = unsafe { &*(cache_ptr as *mut OpaqueKVCache) };
    cache.layers.len() as i64
}

#[unsafe(no_mangle)]
/// @ffi_sig (i64, i64) -> void
pub extern "C" fn tl_kv_cache_resize(cache_ptr: i64, max_len: i64) {
    if cache_ptr == 0 || max_len < 0 {
        return;
    }
    let cache = unsafe { &mut *(cache_ptr as *mut OpaqueKVCache) };
    let new_len = max_len as usize;
    
    // 縮小される場合、切り捨てられる要素を明示的に解放する
    if new_len < cache.layers.len() {
        for (k_opt, v_opt) in cache.layers.drain(new_len..) {
            if let Some(k) = k_opt {
                if !k.is_null() {
                    release_tensor_safe(k);
                }
            }
            if let Some(v) = v_opt {
                if !v.is_null() {
                    release_tensor_safe(v);
                }
            }
        }
    } else if new_len > cache.layers.len() {
        cache.layers.resize(new_len, (None, None));
    }
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
        match cache.layers[idx].0 {
            Some(ptr) if !ptr.is_null() => {
                // Arc RC+1: JIT が戻り値を auto-release しても cache 側のコピーは有効
                tensor_arc_retain(ptr)
            }
            _ => std::ptr::null_mut(),
        }
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
        match cache.layers[idx].1 {
            Some(ptr) if !ptr.is_null() => {
                // Arc RC+1: JIT が戻り値を auto-release しても cache 側のコピーは有効
                tensor_arc_retain(ptr)
            }
            _ => std::ptr::null_mut(),
        }
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
    // 古い値を解放
    if let Some(old_k) = cache.layers[idx].0 {
        if !old_k.is_null() {
            release_tensor_safe(old_k);
        }
    }
    if let Some(old_v) = cache.layers[idx].1 {
        if !old_v.is_null() {
            release_tensor_safe(old_v);
        }
    }
    // Arc RC+1 でキャッシュに保存（JITの自動解放とは独立した参照）
    let k_retained = if !k.is_null() {
        Some(tensor_arc_retain(k))
    } else {
        None
    };
    let v_retained = if !v.is_null() {
        Some(tensor_arc_retain(v))
    } else {
        None
    };
    cache.layers[idx] = (k_retained, v_retained);
}

#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
/// KVCache 全レイヤーのキャッシュをクリア（テンソル解放 + None リセット）
pub extern "C" fn tl_kv_cache_clear(cache_ptr: i64) {
    if cache_ptr == 0 {
        return;
    }
    let cache = unsafe { &mut *(cache_ptr as *mut OpaqueKVCache) };
    for (k_opt, v_opt) in cache.layers.iter_mut() {
        if let Some(k) = k_opt.take() {
            if !k.is_null() {
                release_tensor_safe(k);
            }
        }
        if let Some(v) = v_opt.take() {
            if !v.is_null() {
                release_tensor_safe(v);
            }
        }
    }
}

// ========== メモリ統計 ==========

#[unsafe(no_mangle)]
/// @ffi_sig () -> void
/// メモリ統計レポート — CPU/GPU 両対応
pub extern "C" fn tl_system_mem_report() {
    if crate::device_ffi::is_cpu() {
        tl_cpu::ffi::tl_cpu_mem_stats_report();
    } else {
        eprintln!("[MEM] GPU mode: use TL_GPU_POOL_STATS for GPU statistics");
    }
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
    crate::device_ffi::runtime_gpu_sync();
}

/// System::platform() -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_platform() -> *mut StringStruct {
    let s = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let c_str = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
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

/// System::command(cmd: String) -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_system_command(cmd: *const c_char) -> *mut StringStruct {
    unsafe {
        if cmd.is_null() {
            let layout = std::alloc::Layout::new::<StringStruct>();
            let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
            (*struct_ptr).ptr = std::ptr::null_mut();
            (*struct_ptr).len = 0;
            return struct_ptr;
        }
        let cmd_str = CStr::from_ptr(cmd).to_string_lossy();
        
        // Using sh -c or cmd /C based on OS
        let output = if cfg!(target_os = "windows") {
            std::process::Command::new("cmd").args(["/C", &cmd_str]).output()
        } else {
            std::process::Command::new("sh").args(["-c", &cmd_str]).output()
        };
        
        let result_str = match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                if out.status.success() {
                    stdout.into_owned()
                } else {
                    format!("{}{}", stdout, stderr)
                }
            },
            Err(e) => format!("Execute error: {}", e),
        };
        
        let c_str = CString::new(result_str).unwrap_or_else(|_| CString::new("").unwrap());
        let ptr = c_str.into_raw();
        let len = libc::strlen(ptr) as i64;
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
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

// ========== オプティマイザ ==========

/// Adam optimizer step (GPU ネイティブ — ゼロコピーインプレース更新):
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// weight_decay > 0 の場合は AdamW: param -= lr * weight_decay * param
/// param -= lr * m_hat / (sqrt(v_hat) + eps)
/// @ffi_sig (Tensor*, Tensor*, Tensor*, Tensor*, i64, f32, f32, f32, f32, f32) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_adam_step(
    param: *mut crate::OpaqueTensor,
    grad: *mut crate::OpaqueTensor,
    m: *mut crate::OpaqueTensor,
    v: *mut crate::OpaqueTensor,
    step: i64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) {
    if param.is_null() || grad.is_null() || m.is_null() || v.is_null() {
        return;
    }

    let t = step as f32 + 1.0;
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);

    if crate::device_ffi::is_cpu() {
        // ── CPU: CpuTensor の data を直接操作 ──
        unsafe {
            let p = &mut *(param as *mut tl_cpu::CpuTensor<f32>);
            let g = &*(grad as *const tl_cpu::CpuTensor<f32>);
            let m_t = &mut *(m as *mut tl_cpu::CpuTensor<f32>);
            let v_t = &mut *(v as *mut tl_cpu::CpuTensor<f32>);

            let p_data = p.data_mut();
            let g_data = g.data();
            let m_data = m_t.data_mut();
            let v_data = v_t.data_mut();

            let n = p_data.len().min(g_data.len()).min(m_data.len()).min(v_data.len());

            // AdamW: weight decay を直接 param に適用
            if weight_decay > 0.0 {
                for i in 0..n {
                    p_data[i] -= lr * weight_decay * p_data[i];
                }
            }

            for i in 0..n {
                let gi = g_data[i].clamp(-5.0, 5.0);
                m_data[i] = beta1 * m_data[i] + (1.0 - beta1) * gi;
                v_data[i] = beta2 * v_data[i] + (1.0 - beta2) * gi * gi;
                let m_hat = m_data[i] / bc1;
                let v_hat = v_data[i] / bc2;
                p_data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            p.zero_grad();
        }
    } else {
        // ── GPU: Metal SharedMemory バッファを直接操作 ──
        #[cfg(target_os = "macos")]
        {
            // GPU コマンド完了を待機してからバッファにアクセス
            tl_metal::command_stream::sync_stream();

            unsafe {
                let p = &mut *(param as *mut tl_metal::MetalTensor);
                let g = &*(grad as *const tl_metal::MetalTensor);
                let m_t = &mut *(m as *mut tl_metal::MetalTensor);
                let v_t = &mut *(v as *mut tl_metal::MetalTensor);

                let n = p.elem_count();
                let p_ptr = p.buffer().contents() as *mut f32;
                let g_ptr = g.buffer().contents() as *const f32;
                let m_ptr = m_t.buffer().contents() as *mut f32;
                let v_ptr = v_t.buffer().contents() as *mut f32;

                // AdamW: weight decay を直接 param に適用
                if weight_decay > 0.0 {
                    for i in 0..n {
                        *p_ptr.add(i) -= lr * weight_decay * *p_ptr.add(i);
                    }
                }

                for i in 0..n {
                    let gi = (*g_ptr.add(i)).clamp(-5.0, 5.0);
                    *m_ptr.add(i) = beta1 * *m_ptr.add(i) + (1.0 - beta1) * gi;
                    *v_ptr.add(i) = beta2 * *v_ptr.add(i) + (1.0 - beta2) * gi * gi;
                    let m_hat = *m_ptr.add(i) / bc1;
                    let v_hat = *v_ptr.add(i) / bc2;
                    *p_ptr.add(i) -= lr * m_hat / (v_hat.sqrt() + eps);
                }

                p.zero_grad();
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // CUDA: CudaTensor のバッファを直接操作
            unsafe {
                let p = &mut *(param as *mut tl_cuda::tensor::CudaTensor);
                let g = &*(grad as *const tl_cuda::tensor::CudaTensor);
                let m_t = &mut *(m as *mut tl_cuda::tensor::CudaTensor);
                let v_t = &mut *(v as *mut tl_cuda::tensor::CudaTensor);

                // CUDA: to_vec でデータ取得 → インプレース更新 → from_slice で書き戻し
                let mut p_data = p.to_vec::<f32>();
                let g_data = g.to_vec::<f32>();
                let mut m_data = m_t.to_vec::<f32>();
                let mut v_data = v_t.to_vec::<f32>();

                let n = p_data.len().min(g_data.len()).min(m_data.len()).min(v_data.len());

                if weight_decay > 0.0 {
                    for i in 0..n {
                        p_data[i] -= lr * weight_decay * p_data[i];
                    }
                }

                for i in 0..n {
                    m_data[i] = beta1 * m_data[i] + (1.0 - beta1) * g_data[i];
                    v_data[i] = beta2 * v_data[i] + (1.0 - beta2) * g_data[i] * g_data[i];
                    let m_hat = m_data[i] / bc1;
                    let v_hat = v_data[i] / bc2;
                    p_data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }

                // replace_data で書き戻し
                use std::ffi::c_void;
                let shape = crate::device_ffi::read_runtime_tensor_shape(param as *mut c_void);
                let new_p = crate::device_ffi::create_runtime_tensor_f32(&p_data, &shape);
                let new_m = crate::device_ffi::create_runtime_tensor_f32(&m_data, &shape);
                let new_v = crate::device_ffi::create_runtime_tensor_f32(&v_data, &shape);
                crate::device_ffi::tl_device_tensor_replace_data(param as *mut c_void, new_p);
                crate::device_ffi::tl_device_tensor_replace_data(m as *mut c_void, new_m);
                crate::device_ffi::tl_device_tensor_replace_data(v as *mut c_void, new_v);
                crate::device_ffi::tl_device_tensor_free(new_p);
                crate::device_ffi::tl_device_tensor_free(new_m);
                crate::device_ffi::tl_device_tensor_free(new_v);

                p.zero_grad();
            }
        }
    }
}

/// SGD with momentum (GPU ネイティブ — ゼロコピーインプレース更新):
/// velocity = momentum * velocity + grad + weight_decay * param
/// param = param - lr * velocity (or nesterov variant)
/// @ffi_sig (Tensor*, Tensor*, Tensor*, f32, f32, f32, f32, bool) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_sgd_step(
    param: *mut crate::OpaqueTensor,
    grad: *mut crate::OpaqueTensor,
    velocity: *mut crate::OpaqueTensor,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
) {
    if param.is_null() || grad.is_null() || velocity.is_null() {
        return;
    }

    if crate::device_ffi::is_cpu() {
        // ── CPU: CpuTensor の data を直接操作 ──
        unsafe {
            let p = &mut *(param as *mut tl_cpu::CpuTensor<f32>);
            let g = &*(grad as *const tl_cpu::CpuTensor<f32>);
            let vel = &mut *(velocity as *mut tl_cpu::CpuTensor<f32>);

            let p_data = p.data_mut();
            let g_data = g.data();
            let v_data = vel.data_mut();

            let n = p_data.len().min(g_data.len()).min(v_data.len());

            for i in 0..n {
                let mut gi = g_data[i];
                if weight_decay > 0.0 {
                    gi += weight_decay * p_data[i];
                }
                v_data[i] = momentum * v_data[i] + (1.0 - dampening) * gi;
                if nesterov {
                    p_data[i] -= lr * (gi + momentum * v_data[i]);
                } else {
                    p_data[i] -= lr * v_data[i];
                }
            }
        }
    } else {
        // ── GPU: Metal SharedMemory バッファを直接操作 ──
        #[cfg(target_os = "macos")]
        {
            tl_metal::command_stream::sync_stream();

            unsafe {
                let p = &mut *(param as *mut tl_metal::MetalTensor);
                let g = &*(grad as *const tl_metal::MetalTensor);
                let vel = &mut *(velocity as *mut tl_metal::MetalTensor);

                let n = p.elem_count();
                let p_ptr = p.buffer().contents() as *mut f32;
                let g_ptr = g.buffer().contents() as *const f32;
                let v_ptr = vel.buffer().contents() as *mut f32;

                for i in 0..n {
                    let mut gi = *g_ptr.add(i);
                    if weight_decay > 0.0 {
                        gi += weight_decay * *p_ptr.add(i);
                    }
                    *v_ptr.add(i) = momentum * *v_ptr.add(i) + (1.0 - dampening) * gi;
                    if nesterov {
                        *p_ptr.add(i) -= lr * (gi + momentum * *v_ptr.add(i));
                    } else {
                        *p_ptr.add(i) -= lr * *v_ptr.add(i);
                    }
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // CUDA: to_vec → 計算 → replace_data
            unsafe {
                let p = &mut *(param as *mut tl_cuda::tensor::CudaTensor);
                let g = &*(grad as *const tl_cuda::tensor::CudaTensor);
                let vel = &mut *(velocity as *mut tl_cuda::tensor::CudaTensor);

                let mut p_data = p.to_vec::<f32>();
                let g_data = g.to_vec::<f32>();
                let mut v_data = vel.to_vec::<f32>();

                let n = p_data.len().min(g_data.len()).min(v_data.len());

                for i in 0..n {
                    let mut gi = g_data[i];
                    if weight_decay > 0.0 {
                        gi += weight_decay * p_data[i];
                    }
                    v_data[i] = momentum * v_data[i] + (1.0 - dampening) * gi;
                    if nesterov {
                        p_data[i] -= lr * (gi + momentum * v_data[i]);
                    } else {
                        p_data[i] -= lr * v_data[i];
                    }
                }

                use std::ffi::c_void;
                let shape = crate::device_ffi::read_runtime_tensor_shape(param as *mut c_void);
                let new_p = crate::device_ffi::create_runtime_tensor_f32(&p_data, &shape);
                let new_v = crate::device_ffi::create_runtime_tensor_f32(&v_data, &shape);
                crate::device_ffi::tl_device_tensor_replace_data(param as *mut c_void, new_p);
                crate::device_ffi::tl_device_tensor_replace_data(velocity as *mut c_void, new_v);
                crate::device_ffi::tl_device_tensor_free(new_p);
                crate::device_ffi::tl_device_tensor_free(new_v);
            }
        }
    }
}

// ========== 学習率スケジューラ ==========

/// Cosine Annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / total))
/// @ffi_sig (f32, i64, i64, f32) -> f32
#[unsafe(no_mangle)]
pub extern "C" fn tl_lr_cosine_annealing(base_lr: f32, step: i64, total_steps: i64, min_lr: f32) -> f32 {
    if total_steps <= 0 {
        return base_lr;
    }
    let progress = (step as f32) / (total_steps as f32);
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
}

/// Step LR: lr = base_lr * gamma^(step / step_size)
/// @ffi_sig (f32, i64, i64, f32) -> f32
#[unsafe(no_mangle)]
pub extern "C" fn tl_lr_step(base_lr: f32, step: i64, step_size: i64, gamma: f32) -> f32 {
    if step_size <= 0 {
        return base_lr;
    }
    base_lr * gamma.powf((step / step_size) as f32)
}

