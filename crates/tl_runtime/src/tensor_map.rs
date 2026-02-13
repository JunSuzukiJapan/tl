//! TensorMap 関連の FFI 関数
//! CPU/GPU 両方で動作するよう抽象化

use crate::string_ffi::StringStruct;
use crate::OpaqueTensor;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;


/// テンソルのデータを型非依存に保持するエントリ
pub(crate) struct TensorEntry {
    pub(crate) data_f32: Vec<f32>,
    pub(crate) shape: Vec<usize>,
    /// 0=F32, 1=F16, 2=BF16, 3=I32, 4=I64
    #[allow(dead_code)]
    pub(crate) dtype_tag: u8,
}

/// TensorMap 構造体 — CPU/GPU 両方で使用可能
pub struct OpaqueTensorMap {
    pub(crate) entries: HashMap<String, TensorEntry>,
    pub qtensors: HashMap<String, std::sync::Arc<crate::quantized::QTensor>>,
}

fn is_cpu_mode() -> bool {
    std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu")
}

/// OpaqueTensor ポインタからデータと shape を抽出 (CPU/GPU 両対応)
unsafe fn extract_tensor_data(tensor: *mut OpaqueTensor) -> Option<(Vec<f32>, Vec<usize>, u8)> {
    if tensor.is_null() {
        return None;
    }
    if is_cpu_mode() {
        unsafe {
            let cpu = &*(tensor as *mut tl_cpu::CpuTensor);
            let data = cpu.data_f32().to_vec();
            let shape = cpu.shape().to_vec();
            Some((data, shape, 0)) // F32
        }
    } else {
        unsafe {
            let metal = &*tensor;
            let data: Vec<f32> = metal.to_vec();
            let shape = tl_metal::MetalTensor::shape(metal).to_vec();
            Some((data, shape, 0)) // F32
        }
    }
}

/// TensorEntry から OpaqueTensor を作成 (CPU/GPU 両対応)
fn create_tensor_from_entry(entry: &TensorEntry) -> *mut OpaqueTensor {
    if is_cpu_mode() {
        let dtype = tl_cpu::DType::F32; // 現時点では F32 のみサポート
        let cpu = tl_cpu::CpuTensor::from_slice(&entry.data_f32, &entry.shape, dtype);
        Box::into_raw(Box::new(cpu)) as *mut OpaqueTensor
    } else {
        let metal = tl_metal::MetalTensor::from_slice(&entry.data_f32, &entry.shape, tl_metal::DType::F32);
        crate::make_metal_tensor(metal)
    }
}

/// 新しい TensorMap を作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_new() -> *mut OpaqueTensorMap {
    Box::into_raw(Box::new(OpaqueTensorMap {
        entries: HashMap::new(),
        qtensors: HashMap::new(),
    }))
}

/// TensorMap を解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_free(map: *mut OpaqueTensorMap) {
    if !map.is_null() {
        unsafe {
            let _ = Box::from_raw(map);
        }
    }
}

/// TensorMap にテンソルを挿入
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_insert(
    map: *mut OpaqueTensorMap,
    name: *mut StringStruct,
    tensor: *mut OpaqueTensor,
) {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() || tensor.is_null() {
            return;
        }
        let map_ref = &mut (*map).entries;
        let key = CStr::from_ptr((*name).ptr).to_string_lossy().into_owned();
        if let Some((data, shape, dtype_tag)) = extract_tensor_data(tensor) {
            map_ref.insert(key, TensorEntry { data_f32: data, shape, dtype_tag });
        }
    }
}

/// TensorMap からテンソルを取得
/// entries (F32) を先に検索し、なければ qtensors (量子化) をデクォンタイズして返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get(
    map: *mut OpaqueTensorMap,
    name: *mut StringStruct,
) -> *mut OpaqueTensor {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let map_ref = &(*map);
        let key = CStr::from_ptr((*name).ptr).to_string_lossy();
        
        // 1. entries (F32テンソル) を検索
        if let Some(entry) = map_ref.entries.get(key.as_ref()) {
            return create_tensor_from_entry(entry);
        }
        
        // 2. qtensors (量子化テンソル) をフォールバック検索
        if let Some(qtensor_arc) = map_ref.qtensors.get(key.as_ref()) {
            match qtensor_arc.dequantize_to_tensor() {
                Ok(tensor_ptr) => return tensor_ptr,
                Err(e) => {
                    crate::error::set_last_error(
                        format!("Failed to dequantize '{}': {}", key, e),
                        crate::error::RuntimeErrorCode::ArgumentError,
                    );
                    return std::ptr::null_mut();
                }
            }
        }
        
        crate::error::set_last_error(
            format!("Weight '{}' not found in loaded file.", key),
            crate::error::RuntimeErrorCode::ArgumentError,
        );
        std::ptr::null_mut()
    }
}

/// TensorMap を safetensors からロード
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_load(path: *mut StringStruct) -> *mut OpaqueTensorMap {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);
        
        // safetensors ファイルを読み込み
        match safetensors::SafeTensors::deserialize(&std::fs::read(&path_buf).unwrap_or_default()) {
            Ok(tensors) => {
                let mut entries = HashMap::new();
                for (name, view) in tensors.tensors() {
                    // データを f32 として読み込み
                    let data: Vec<f32> = match view.dtype() {
                        safetensors::Dtype::F32 => {
                            let bytes = view.data();
                            bytes.chunks_exact(4)
                                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                .collect()
                        }
                        safetensors::Dtype::F16 => {
                            // F16 は現在未サポート - F32 として 0 を返す
                            eprintln!("Warning: F16 tensors not yet supported, skipping {}", name);
                            continue;
                        }
                        _ => {
                            eprintln!("Warning: Unsupported dtype for tensor {}", name);
                            continue;
                        }
                    };
                    let shape: Vec<usize> = view.shape().to_vec();
                    entries.insert(name.to_string(), TensorEntry {
                        data_f32: data,
                        shape,
                        dtype_tag: 0, // F32
                    });
                }
                println!("Loaded {} tensors from {:?}", entries.len(), path_buf);
                Box::into_raw(Box::new(OpaqueTensorMap { entries, qtensors: HashMap::new() }))
            }
            Err(e) => {
                eprintln!("Failed to load safetensors: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// TensorMap をバイナリ形式で保存
/// tl_tensor_load と互換性のあるフォーマット: [rank: u64][shape: u64 * rank][data: f32 * numel]
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_save(map: *mut OpaqueTensorMap, path: *mut StringStruct) {
    unsafe {
        if map.is_null() || path.is_null() || (*path).ptr.is_null() {
            return;
        }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);
        let entries = &(*map).entries;

        // 最初のテンソルをバイナリ形式で保存（tl_tensor_load 互換）
        if let Some((_name, entry)) = entries.iter().next() {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(entry.shape.len() as u64).to_le_bytes());
            for &dim in &entry.shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            for &val in &entry.data_f32 {
                bytes.extend_from_slice(&val.to_le_bytes());
            }

            match std::fs::write(&path_buf, &bytes) {
                Ok(_) => println!("Saved {} tensors to {:?}", entries.len(), path_buf),
                Err(e) => eprintln!("Failed to save tensor: {}", e),
            }
        }
    }
}

/// Quantized テンソルを取得
/// コンパイラシグネチャ: (void_ptr, i8_ptr) -> void_ptr
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get_quantized(
    map: *mut OpaqueTensorMap,
    name: *mut StringStruct,
) -> *mut crate::quantized::QTensor {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let map_ref = &(*map);
        let key = std::ffi::CStr::from_ptr((*name).ptr).to_string_lossy();

        if let Some(qtensor_arc) = map_ref.qtensors.get(key.as_ref()) {
            let arc_clone = Arc::clone(qtensor_arc);
            return Arc::into_raw(arc_clone) as *mut crate::quantized::QTensor;
        }

        eprintln!("Warning: Quantized tensor '{}' not found", key);
        std::ptr::null_mut()
    }
}


