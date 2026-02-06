//! String 関連の FFI 関数

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// String 構造体 (C ABI 互換)
#[repr(C)]
pub struct StringStruct {
    pub ptr: *mut c_char,
    pub len: i64,
}

/// 新しい StringStruct を作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_new(s: *const c_char) -> *mut StringStruct {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let c_str = CStr::from_ptr(s);
        let s_slice = c_str.to_string_lossy().into_owned();
        let new_c_str = CString::new(s_slice).unwrap();
        let ptr = new_c_str.into_raw();
        let len = libc::strlen(ptr) as i64;
        
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
    }
}

/// StringStruct の長さを取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_len(s: *mut StringStruct) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).len }
}

/// 2つの StringStruct を連結
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_concat(a: *mut StringStruct, b: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if a.is_null() || b.is_null() {
            return std::ptr::null_mut();
        }
        let a_str = if (*a).ptr.is_null() { String::new() } else { CStr::from_ptr((*a).ptr).to_string_lossy().into_owned() };
        let b_str = if (*b).ptr.is_null() { String::new() } else { CStr::from_ptr((*b).ptr).to_string_lossy().into_owned() };
        let result = format!("{}{}", a_str, b_str);
        let c_str = CString::new(result).unwrap();
        let ptr = c_str.into_raw();
        let len = libc::strlen(ptr) as i64;
        
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
    }
}

/// 文字列比較
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_eq(a: *mut StringStruct, b: *mut StringStruct) -> bool {
    unsafe {
        if a.is_null() || b.is_null() {
            return a.is_null() && b.is_null();
        }
        if (*a).ptr.is_null() || (*b).ptr.is_null() {
            return (*a).ptr.is_null() && (*b).ptr.is_null();
        }
        let a_str = CStr::from_ptr((*a).ptr);
        let b_str = CStr::from_ptr((*b).ptr);
        a_str == b_str
    }
}

/// 部分文字列を含むか
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_contains(s: *mut StringStruct, sub: *mut StringStruct) -> bool {
    unsafe {
        if s.is_null() || sub.is_null() || (*s).ptr.is_null() || (*sub).ptr.is_null() {
            return false;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let sub_str = CStr::from_ptr((*sub).ptr).to_string_lossy();
        s_str.contains(sub_str.as_ref())
    }
}

/// 文字列から整数へ変換
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_i64(s: *mut StringStruct) -> i64 {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return 0;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        s_str.trim().parse::<i64>().unwrap_or(0)
    }
}

/// 整数から文字列へ変換
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_int(i: i64) -> *mut StringStruct {
    let s = format!("{}", i);
    let c_str = CString::new(s).unwrap();
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

/// 文字から文字列へ変換
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_char(c: i64) -> *mut StringStruct {
    let ch = char::from_u32(c as u32).unwrap_or('?');
    let s = ch.to_string();
    let c_str = CString::new(s).unwrap();
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

/// 文字列のi番目の文字を取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_char_at(s: *mut StringStruct, i: i64) -> i64 {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() || i < 0 {
            return 0;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        s_str.chars().nth(i as usize).map(|c| c as i64).unwrap_or(0)
    }
}

/// 文字列のハッシュを取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_hash_string(s: *mut StringStruct) -> i64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return 0;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let mut hasher = DefaultHasher::new();
        s_str.hash(&mut hasher);
        hasher.finish() as i64
    }
}
