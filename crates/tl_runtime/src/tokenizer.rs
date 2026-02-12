//! Tokenizer 関連の FFI 関数

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

/// Tokenizer ラッパー
pub struct OpaqueTokenizer {
    pub inner: Arc<tokenizers::Tokenizer>,
}

/// 新しい Tokenizer を作成
/// codegen ABI: (path: *const c_char) -> i64 (handle)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_new(path: *const c_char) -> i64 {
    unsafe {
        if path.is_null() {
            return 0;
        }
        let path_str = CStr::from_ptr(path).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);
        
        match tokenizers::Tokenizer::from_file(&path_buf) {
            Ok(tokenizer) => {
                println!("Loaded tokenizer from {:?}", path_buf);
                let tok = Box::new(OpaqueTokenizer {
                    inner: Arc::new(tokenizer),
                });
                Box::into_raw(tok) as i64
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                0
            }
        }
    }
}

/// テキストをエンコード
/// codegen ABI: (tok_handle: i64, text: *const c_char) -> *mut OpaqueTensor
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode(
    tokenizer_handle: i64,
    text: *const c_char,
) -> *mut crate::OpaqueTensor {
    unsafe {
        if tokenizer_handle == 0 || text.is_null() {
            return std::ptr::null_mut();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let text_str = CStr::from_ptr(text).to_string_lossy();
        
        match tok.encode(text_str.as_ref(), false) {
            Ok(encoding) => {
                let ids: Vec<f32> = encoding.get_ids().iter().map(|&id| id as f32).collect();
                let shape = vec![ids.len()];
                let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
                if is_cpu {
                    tl_cpu::ffi::tl_cpu_tensor_new(ids.as_ptr(), shape.len(), shape.as_ptr()) as *mut crate::OpaqueTensor
                } else {
                    tl_metal::ffi_ops::tl_metal_new(ids.as_ptr(), shape.len(), shape.as_ptr())
                }
            }
            Err(e) => {
                eprintln!("Tokenizer encode error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// トークン ID をデコード
/// codegen ABI: (tok_handle: i64, ids: *mut OpaqueTensor) -> *const c_char
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_decode(
    tokenizer_handle: i64,
    ids: *mut crate::OpaqueTensor,
) -> *const c_char {
    unsafe {
        if tokenizer_handle == 0 || ids.is_null() {
            let empty = CString::new("").unwrap();
            return empty.into_raw();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let tensor = &*ids;
        let data: Vec<f32> = tensor.to_vec();
        let token_ids: Vec<u32> = data.iter().map(|&f| f as u32).collect();
        
        match tok.decode(&token_ids, true) {
            Ok(text) => {
                let c_str = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
                c_str.into_raw()
            }
            Err(e) => {
                eprintln!("Tokenizer decode error: {}", e);
                let empty = CString::new("").unwrap();
                empty.into_raw()
            }
        }
    }
}

