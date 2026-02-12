//! Tokenizer 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::CStr;
use std::sync::Arc;

/// Tokenizer ラッパー
pub struct OpaqueTokenizer {
    pub inner: Arc<tokenizers::Tokenizer>,
}

/// 新しい Tokenizer を作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_new(path: *mut StringStruct) -> *mut OpaqueTokenizer {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);
        
        match tokenizers::Tokenizer::from_file(&path_buf) {
            Ok(tokenizer) => {
                println!("Loaded tokenizer from {:?}", path_buf);
                Box::into_raw(Box::new(OpaqueTokenizer {
                    inner: Arc::new(tokenizer),
                }))
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// テキストをエンコード
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode(
    tokenizer: *mut OpaqueTokenizer,
    text: *mut StringStruct,
) -> *mut crate::OpaqueTensor {
    unsafe {
        if tokenizer.is_null() || text.is_null() || (*text).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let tok = &(*tokenizer).inner;
        let text_str = CStr::from_ptr((*text).ptr).to_string_lossy();
        
        match tok.encode(text_str.as_ref(), false) {
            Ok(encoding) => {
                let ids: Vec<f32> = encoding.get_ids().iter().map(|&id| id as f32).collect();
                let shape = vec![ids.len()];
                tl_metal::ffi_ops::tl_metal_new(ids.as_ptr(), shape.len(), shape.as_ptr())
            }
            Err(e) => {
                eprintln!("Tokenizer encode error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// トークン ID をデコード
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_decode(
    tokenizer: *mut OpaqueTokenizer,
    ids: *mut crate::OpaqueTensor,
) -> *mut StringStruct {
    unsafe {
        if tokenizer.is_null() || ids.is_null() {
            return std::ptr::null_mut();
        }
        let tok = &(*tokenizer).inner;
        let tensor = &*ids;
        let data: Vec<f32> = tensor.to_vec();
        let token_ids: Vec<u32> = data.iter().map(|&f| f as u32).collect();
        
        match tok.decode(&token_ids, true) {
            Ok(text) => {
                let c_str = std::ffi::CString::new(text).unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
                let ptr = c_str.into_raw();
                let len = libc::strlen(ptr) as i64;
                let layout = std::alloc::Layout::new::<StringStruct>();
                let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
                (*struct_ptr).ptr = ptr;
                (*struct_ptr).len = len;
                struct_ptr
            }
            Err(e) => {
                eprintln!("Tokenizer decode error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}
