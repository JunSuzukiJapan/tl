use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Write};
use std::os::raw::c_char;
// use std::path::Path;

// --- File I/O ---

#[no_mangle]
pub extern "C" fn tl_file_open(path: *const c_char, mode: *const c_char) -> *mut File {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let mode_str = unsafe { CStr::from_ptr(mode).to_string_lossy() };

    let f = match mode_str.as_ref() {
        "r" => File::open(path_str.as_ref()),
        "w" => File::create(path_str.as_ref()),
        _ => return std::ptr::null_mut(), // TODO: proper append support or error
    };

    match f {
        Ok(file) => Box::into_raw(Box::new(file)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tl_file_read_string(file: *mut File) -> *mut c_char {
    if file.is_null() {
        return std::ptr::null_mut();
    }
    let file = unsafe { &mut *file };
    let mut content = String::new();
    // Seek to start? usually implicit for linear read
    if file.read_to_string(&mut content).is_ok() {
        match CString::new(content) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub extern "C" fn tl_file_write_string(file: *mut File, content: *const c_char) {
    if file.is_null() || content.is_null() {
        return;
    }
    let file = unsafe { &mut *file };
    let c_str = unsafe { CStr::from_ptr(content) };
    if let Ok(s) = c_str.to_str() {
        let _ = file.write_all(s.as_bytes());
    }
}

#[no_mangle]
pub extern "C" fn tl_file_close(file: *mut File) {
    if !file.is_null() {
        unsafe {
            let _ = Box::from_raw(file); // Dropping closes the file
        }
    }
}

// --- Http ---

#[no_mangle]
pub extern "C" fn tl_http_download(url: *const c_char, dest: *const c_char) -> bool {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    let dest_str = unsafe { CStr::from_ptr(dest).to_string_lossy() };

    // Use blocking client for simplicity in prototype
    if let Ok(response) = reqwest::blocking::get(url_str.as_ref()) {
        if response.status().is_success() {
            if let Ok(bytes) = response.bytes() {
                if let Ok(mut file) = File::create(dest_str.as_ref()) {
                    if file.write_all(&bytes).is_ok() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[no_mangle]
pub extern "C" fn tl_http_get(url: *const c_char) -> *mut c_char {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    if let Ok(response) = reqwest::blocking::get(url_str.as_ref()) {
        if let Ok(text) = response.text() {
            match CString::new(text) {
                Ok(c_str) => return c_str.into_raw(),
                Err(_) => return std::ptr::null_mut(),
            }
        }
    }
    std::ptr::null_mut()
}

// --- Env ---

#[no_mangle]
pub extern "C" fn tl_env_get(key: *const c_char) -> *mut c_char {
    let key_str = unsafe { CStr::from_ptr(key).to_string_lossy() };
    match std::env::var(key_str.as_ref()) {
        Ok(val) => match CString::new(val) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}
