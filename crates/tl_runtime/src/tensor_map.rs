//! TensorMap 関連の FFI 関数

use crate::string_ffi::StringStruct;
use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;

/// TensorMap 構造体
pub struct OpaqueTensorMap {
    pub map: HashMap<String, Arc<MetalTensor>>,
}

/// 新しい TensorMap を作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_new() -> *mut OpaqueTensorMap {
    Box::into_raw(Box::new(OpaqueTensorMap {
        map: HashMap::new(),
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
        let map_ref = &mut (*map).map;
        let key = CStr::from_ptr((*name).ptr).to_string_lossy().into_owned();
        let tensor_clone = (*tensor).clone();
        map_ref.insert(key, Arc::new(tensor_clone));
    }
}

/// TensorMap からテンソルを取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get(
    map: *mut OpaqueTensorMap,
    name: *mut StringStruct,
) -> *mut OpaqueTensor {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let map_ref = &(*map).map;
        let key = CStr::from_ptr((*name).ptr).to_string_lossy();
        
        if let Some(tensor_arc) = map_ref.get(key.as_ref()) {
            let tensor_clone = tensor_arc.as_ref().clone();
            Box::into_raw(Box::new(tensor_clone))
        } else {
            crate::error::set_last_error(
                format!("Weight '{}' not found in loaded file.", key),
                crate::error::RuntimeErrorCode::ArgumentError,
            );
            std::ptr::null_mut()
        }
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
                let mut map = HashMap::new();
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
                    let tensor = MetalTensor::from_slice(&data, &shape, DType::F32);
                    map.insert(name.to_string(), Arc::new(tensor));
                }
                println!("Loaded {} tensors from {:?}", map.len(), path_buf);
                Box::into_raw(Box::new(OpaqueTensorMap { map }))
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
        let map_ref = &(*map).map;

        // 最初のテンソルをバイナリ形式で保存（tl_tensor_load 互換）
        if let Some((_name, tensor_arc)) = map_ref.iter().next() {
            let tensor = tensor_arc.as_ref();
            let data: Vec<f32> = tensor.to_vec();
            let shape = MetalTensor::shape(tensor);

            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for &dim in shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            for &val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }

            match std::fs::write(&path_buf, &bytes) {
                Ok(_) => println!("Saved {} tensors to {:?}", map_ref.len(), path_buf),
                Err(e) => eprintln!("Failed to save tensor: {}", e),
            }
        }
    }
}

/// Quantized テンソルを取得（スタブ - 未実装）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get_quantized(
    _map: i64,
    _name: *mut StringStruct,
) -> usize {
    // Quantized テンソルは未実装
    eprintln!("Warning: Quantized tensors not yet supported in Metal backend");
    0
}
