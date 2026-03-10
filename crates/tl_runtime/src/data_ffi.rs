//! Data FFI — CSV/JSON/DataLoader

use std::ffi::{c_void, CStr};
use crate::string_ffi::StringStruct;

/// CSV を 2D Tensor として読み込み (数値のみ)
/// @ffi_sig (String*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_csv_load(path: *mut StringStruct) -> *mut c_void {
    let path_str = if !path.is_null() {
        unsafe {
            if !(*path).ptr.is_null() {
                CStr::from_ptr((*path).ptr).to_string_lossy().to_string()
            } else { return std::ptr::null_mut(); }
        }
    } else { return std::ptr::null_mut(); };

    let mut reader = match csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: csv_load failed: {}", e);
            return std::ptr::null_mut();
        }
    };

    let mut rows: Vec<Vec<f32>> = Vec::new();
    for result in reader.records() {
        match result {
            Ok(record) => {
                let row: Vec<f32> = record.iter()
                    .filter_map(|field| field.trim().parse::<f32>().ok())
                    .collect();
                if !row.is_empty() {
                    rows.push(row);
                }
            }
            Err(_) => continue,
        }
    }

    if rows.is_empty() {
        return std::ptr::null_mut();
    }

    let num_rows = rows.len();
    let num_cols = rows[0].len();
    let mut data = vec![0.0f32; num_rows * num_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j < num_cols {
                data[i * num_cols + j] = val;
            }
        }
    }
    crate::device_ffi::create_runtime_tensor_f32(&data, &[num_rows, num_cols])
}

/// JSON ファイルをString として読み込み
/// @ffi_sig (String*) -> String*
#[unsafe(no_mangle)]
pub extern "C" fn tl_json_load(path: *mut StringStruct) -> *mut StringStruct {
    let path_str = if !path.is_null() {
        unsafe {
            if !(*path).ptr.is_null() {
                CStr::from_ptr((*path).ptr).to_string_lossy().to_string()
            } else { return std::ptr::null_mut(); }
        }
    } else { return std::ptr::null_mut(); };

    let content = match std::fs::read_to_string(&path_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: json_load failed: {}", e);
            return std::ptr::null_mut();
        }
    };

    // JSONとしてパース可能か検証
    if serde_json::from_str::<serde_json::Value>(&content).is_err() {
        eprintln!("Warning: json_load: file is not valid JSON");
    }

    // StringStruct として返す
    let c_str = std::ffi::CString::new(content).unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
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

// ========== DataLoader ==========

struct DataLoaderState {
    data: *mut c_void,       // テンソルポインタ
    labels: *mut c_void,     // ラベルテンソルポインタ
    batch_size: usize,
    num_samples: usize,
    current_index: usize,
    shuffle_indices: Vec<usize>,
}

/// DataLoaderを作成
/// @ffi_sig (Tensor*, Tensor*, i64, bool) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_dataloader_new(
    data: *mut c_void,
    labels: *mut c_void,
    batch_size: i64,
    shuffle: bool,
) -> i64 {
    let num_samples = crate::device_ffi::tl_device_tensor_dim(data, 0);
    let mut indices: Vec<usize> = (0..num_samples).collect();
    if shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }
    let state = Box::new(DataLoaderState {
        data,
        labels,
        batch_size: batch_size as usize,
        num_samples,
        current_index: 0,
        shuffle_indices: indices,
    });
    Box::into_raw(state) as i64
}

/// DataLoader のバッチ数を返す
/// @ffi_sig (i64) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_dataloader_len(handle: i64) -> i64 {
    if handle == 0 { return 0; }
    let state = unsafe { &*(handle as *const DataLoaderState) };
    ((state.num_samples + state.batch_size - 1) / state.batch_size) as i64
}

/// DataLoader をリセット（オプションで再シャッフル）
/// @ffi_sig (i64) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_dataloader_reset(handle: i64) {
    if handle == 0 { return; }
    let state = unsafe { &mut *(handle as *mut DataLoaderState) };
    state.current_index = 0;
    // Re-shuffle
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    state.shuffle_indices.shuffle(&mut rng);
}

/// DataLoader を解放
/// @ffi_sig (i64) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_dataloader_free(handle: i64) {
    if handle == 0 { return; }
    unsafe {
        let _ = Box::from_raw(handle as *mut DataLoaderState);
    }
}
