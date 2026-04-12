//! Regex FFI bindings using ID registry
use std::collections::HashMap;
use std::ffi::CStr;

use std::sync::Mutex;
use once_cell::sync::Lazy;
use regex::Regex;
use crate::string_ffi::{make_string_struct, StringStruct};

// Global registry for compiled Regex instances
static REGEX_REGISTRY: Lazy<Mutex<HashMap<i64, Regex>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_REGEX_ID: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(1));

/// Returns a new Regex ID if compiled successfully, or -1 on error
#[unsafe(no_mangle)]
pub extern "C" fn tl_regex_new(pattern: *mut StringStruct) -> i64 {
    unsafe {
        if pattern.is_null() || (*pattern).ptr.is_null() {
            return -1;
        }
        let pattern_str = CStr::from_ptr((*pattern).ptr).to_string_lossy();
        match Regex::new(pattern_str.as_ref()) {
            Ok(re) => {
                let mut id_lock = NEXT_REGEX_ID.lock().unwrap();
                let id = *id_lock;
                *id_lock += 1;
                
                let mut registry = REGEX_REGISTRY.lock().unwrap();
                registry.insert(id, re);
                id
            },
            Err(e) => {
                eprintln!("TL Regex compilation error: {}", e);
                -1
            }
        }
    }
}

/// Checks if the regex matches the text
#[unsafe(no_mangle)]
pub extern "C" fn tl_regex_is_match(id: i64, text: *mut StringStruct) -> bool {
    if id < 0 || text.is_null() {
        return false;
    }
    unsafe {
        if (*text).ptr.is_null() {
            return false;
        }
        let registry = REGEX_REGISTRY.lock().unwrap();
        if let Some(re) = registry.get(&id) {
            let text_str = CStr::from_ptr((*text).ptr).to_string_lossy();
            re.is_match(text_str.as_ref())
        } else {
            false
        }
    }
}

/// Helper method. Replace all matching portions.
#[unsafe(no_mangle)]
pub extern "C" fn tl_regex_replace(id: i64, text: *mut StringStruct, replacement: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if text.is_null() || (*text).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let replacement_str = if replacement.is_null() || (*replacement).ptr.is_null() {
            ""
        } else {
            CStr::from_ptr((*replacement).ptr).to_str().unwrap_or("")
        };

        let registry = REGEX_REGISTRY.lock().unwrap();
        if let Some(re) = registry.get(&id) {
            let text_str = CStr::from_ptr((*text).ptr).to_string_lossy();
            let result = re.replace_all(text_str.as_ref(), replacement_str);
            make_string_struct(result.into_owned())
        } else {
            // Return original text if regex not found
            let text_str = CStr::from_ptr((*text).ptr).to_string_lossy();
            make_string_struct(text_str.into_owned())
        }
    }
}

/// Frees the compiled Regex from the registry.
/// NOTE: The name `tl_regex_release` is used instead of `tl_regex_free` to avoid 
/// collisions with the compiler's implicit/auto-generated garbage collector hooks 
/// (`tl_<type>_free`) which typically expect a raw pointer rather than an i64 ID.
#[unsafe(no_mangle)]
pub extern "C" fn tl_regex_release(id: i64) {
    if id > 0 {
        let mut registry = REGEX_REGISTRY.lock().unwrap();
        registry.remove(&id);
    }
}
