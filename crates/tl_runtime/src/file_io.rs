//! File I/O 関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::io::{Read, Write};

/// ファイル存在確認
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_exists(path: *const c_char) -> bool {
    if path.is_null() {
        return false;
    }
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    if path_str.is_empty() {
        return false;
    }
    let path_buf = expand_path(path_str);
    path_buf.exists()
}

/// ファイル存在確認（i64 版）
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_exists_i64(path: *const c_char) -> i64 {
    if tl_file_exists(path) { 1 } else { 0 }
}

/// ファイル読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_read_file(path: *const c_char) -> *mut StringStruct {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let path_buf = expand_path(&path_str);
    
    match std::fs::read_to_string(&path_buf) {
        Ok(content) => {
            let c_str = CString::new(content.trim()).unwrap_or_else(|_| CString::new("").unwrap());
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

/// ファイル書き込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_write_file(path: *const c_char, content: *mut StringStruct) -> bool {
    unsafe {
        if path.is_null() || content.is_null() || (*content).ptr.is_null() {
            return false;
        }
        let path_str = CStr::from_ptr(path).to_string_lossy();
        let path_buf = expand_path(&path_str);
        let content_str = CStr::from_ptr((*content).ptr).to_string_lossy();
        
        match std::fs::write(&path_buf, content_str.as_bytes()) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

/// ファイルダウンロード
#[unsafe(no_mangle)]
pub extern "C" fn tl_download_file(url: *const c_char, path: *const c_char) -> i64 {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let path_buf = expand_path(&path_str);
    
    println!("Downloading from: {}", url_str);
    println!("Saving to: {:?}", path_buf);
    
    match reqwest::blocking::get(url_str.as_ref()) {
        Ok(mut response) => {
            if !response.status().is_success() {
                println!("Download failed: HTTP {}", response.status());
                return 0;
            }
            
            if let Some(parent) = path_buf.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    println!("Failed to create directories: {}", e);
                    return 0;
                }
            }
            
            match std::fs::File::create(&path_buf) {
                Ok(mut file) => {
                    if let Err(e) = std::io::copy(&mut response, &mut file) {
                        println!("Error writing to file: {}", e);
                        return 0;
                    }
                    println!("Download complete!");
                    1
                }
                Err(e) => {
                    println!("Failed to create file: {}", e);
                    0
                }
            }
        }
        Err(e) => {
            println!("Request failed: {}", e);
            0
        }
    }
}

/// HTTP GET リクエスト
#[unsafe(no_mangle)]
pub extern "C" fn tl_http_get(url: *const c_char) -> *mut StringStruct {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    
    match reqwest::blocking::get(url_str.as_ref()) {
        Ok(response) => {
            match response.text() {
                Ok(text) => {
                    let c_str = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
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
        Err(_) => std::ptr::null_mut(),
    }
}

/// HTTP ダウンロード
#[unsafe(no_mangle)]
pub extern "C" fn tl_http_download(url: *const c_char, path: *const c_char) -> i64 {
    tl_download_file(url, path)
}

/// パス展開（~ を HOME に置換）
pub fn expand_path(path: &str) -> std::path::PathBuf {
    if path.starts_with("~") {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(path.replace("~", &home));
        }
    }
    std::path::PathBuf::from(path)
}

// --- Path 関連 ---

/// パス構造体
#[repr(C)]
pub struct PathStruct {
    pub ptr: *mut c_char,
}

/// 新しいパスを作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_new(s: *const c_char) -> *mut PathStruct {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let c_str = CStr::from_ptr(s);
        let new_c_str = CString::new(c_str.to_string_lossy().into_owned()).unwrap();
        let ptr = new_c_str.into_raw();
        
        let layout = std::alloc::Layout::new::<PathStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut PathStruct;
        (*struct_ptr).ptr = ptr;
        struct_ptr
    }
}

/// パスを解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_free(p: *mut PathStruct) {
    if !p.is_null() {
        unsafe {
            if !(*p).ptr.is_null() {
                let _ = CString::from_raw((*p).ptr);
            }
            let layout = std::alloc::Layout::new::<PathStruct>();
            std::alloc::dealloc(p as *mut u8, layout);
        }
    }
}

/// パスを文字列に変換
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_to_string(p: *mut PathStruct) -> *mut StringStruct {
    unsafe {
        if p.is_null() || (*p).ptr.is_null() {
            return std::ptr::null_mut();
        }
        crate::string_ffi::tl_string_new((*p).ptr)
    }
}

/// パスを結合
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_join(a: *mut PathStruct, b: *const c_char) -> *mut PathStruct {
    unsafe {
        if a.is_null() || (*a).ptr.is_null() || b.is_null() {
            return std::ptr::null_mut();
        }
        let a_str = CStr::from_ptr((*a).ptr).to_string_lossy();
        let b_str = CStr::from_ptr(b).to_string_lossy();
        let joined = std::path::PathBuf::from(a_str.as_ref()).join(b_str.as_ref());
        let c_str = CString::new(joined.to_string_lossy().into_owned()).unwrap();
        let ptr = c_str.into_raw();
        
        let layout = std::alloc::Layout::new::<PathStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut PathStruct;
        (*struct_ptr).ptr = ptr;
        struct_ptr
    }
}

/// パスが存在するか
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_exists(p: *mut PathStruct) -> bool {
    unsafe {
        if p.is_null() || (*p).ptr.is_null() {
            return false;
        }
        let path_str = CStr::from_ptr((*p).ptr).to_string_lossy();
        expand_path(&path_str).exists()
    }
}

/// パスがファイルか
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_is_file(p: *mut PathStruct) -> bool {
    unsafe {
        if p.is_null() || (*p).ptr.is_null() {
            return false;
        }
        let path_str = CStr::from_ptr((*p).ptr).to_string_lossy();
        expand_path(&path_str).is_file()
    }
}

/// パスがディレクトリか
#[unsafe(no_mangle)]
pub extern "C" fn tl_path_is_dir(p: *mut PathStruct) -> bool {
    unsafe {
        if p.is_null() || (*p).ptr.is_null() {
            return false;
        }
        let path_str = CStr::from_ptr((*p).ptr).to_string_lossy();
        expand_path(&path_str).is_dir()
    }
}

// --- ファイルハンドル ---

/// ファイルオープン
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_open(path: *const c_char, mode: *const c_char) -> *mut std::ffi::c_void {
    unsafe {
        if path.is_null() || mode.is_null() {
            return std::ptr::null_mut();
        }
        let path_str = CStr::from_ptr(path).to_string_lossy();
        let mode_str = CStr::from_ptr(mode).to_string_lossy();
        let path_buf = expand_path(&path_str);
        
        let file = match mode_str.as_ref() {
            "r" => std::fs::File::open(&path_buf).ok(),
            "w" => std::fs::File::create(&path_buf).ok(),
            "a" => std::fs::OpenOptions::new().append(true).create(true).open(&path_buf).ok(),
            _ => None,
        };
        
        match file {
            Some(f) => Box::into_raw(Box::new(f)) as *mut std::ffi::c_void,
            None => std::ptr::null_mut(),
        }
    }
}

/// ファイルクローズ
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_close(f: *mut std::ffi::c_void) {
    if !f.is_null() {
        unsafe {
            let _ = Box::from_raw(f as *mut std::fs::File);
        }
    }
}

/// ファイルから文字列読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_read_string(f: *mut std::ffi::c_void) -> *mut StringStruct {
    if f.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let file = &mut *(f as *mut std::fs::File);
        let mut content = String::new();
        match file.read_to_string(&mut content) {
            Ok(_) => {
                let c_str = CString::new(content).unwrap_or_else(|_| CString::new("").unwrap());
                let ptr = c_str.into_raw();
                let len = libc::strlen(ptr) as i64;
                let layout = std::alloc::Layout::new::<StringStruct>();
                let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
                (*struct_ptr).ptr = ptr;
                (*struct_ptr).len = len;
                struct_ptr
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// ファイルに文字列書き込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_string(f: *mut std::ffi::c_void, s: *mut StringStruct) -> bool {
    unsafe {
        if f.is_null() || s.is_null() || (*s).ptr.is_null() {
            return false;
        }
        let file = &mut *(f as *mut std::fs::File);
        let content = CStr::from_ptr((*s).ptr).to_string_lossy();
        file.write_all(content.as_bytes()).is_ok()
    }
}

/// ファイルからバイナリ読み込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_read_binary(f: *mut std::ffi::c_void, buf: *mut u8, len: usize) -> usize {
    if f.is_null() || buf.is_null() {
        return 0;
    }
    unsafe {
        let file = &mut *(f as *mut std::fs::File);
        let slice = std::slice::from_raw_parts_mut(buf, len);
        file.read(slice).unwrap_or(0)
    }
}

/// ファイルにバイナリ書き込み
#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_binary(f: *mut std::ffi::c_void, buf: *const u8, len: usize) -> bool {
    if f.is_null() || buf.is_null() {
        return false;
    }
    unsafe {
        let file = &mut *(f as *mut std::fs::File);
        let slice = std::slice::from_raw_parts(buf, len);
        file.write_all(slice).is_ok()
    }
}

/// 環境変数取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_env_get(name: *const c_char) -> *mut StringStruct {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let name_str = unsafe { CStr::from_ptr(name).to_string_lossy() };
    match std::env::var(name_str.as_ref()) {
        Ok(value) => {
            let c_str = CString::new(value).unwrap_or_else(|_| CString::new("").unwrap());
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

/// 環境変数設定
#[unsafe(no_mangle)]
pub extern "C" fn tl_env_set(name: *const c_char, value: *const c_char) {
    if name.is_null() || value.is_null() {
        return;
    }
    let name_str = unsafe { CStr::from_ptr(name).to_string_lossy() };
    let value_str = unsafe { CStr::from_ptr(value).to_string_lossy() };
    unsafe { std::env::set_var(name_str.as_ref(), value_str.as_ref()); }
}
