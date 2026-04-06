//! String 関連の FFI 関数

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// String 構造体 (C ABI 互換)
#[repr(C)]
pub struct StringStruct {
    pub ptr: *mut c_char,
    pub len: i64,
}

/// @ffi_sig (i8*) -> String*
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

/// @ffi_sig (String*) -> void
/// StringStruct を解放
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_free(s: *mut StringStruct) {
    if !s.is_null() {
        unsafe {
            if !(*s).ptr.is_null() {
                let _ = CString::from_raw((*s).ptr);
            }
            let layout = std::alloc::Layout::new::<StringStruct>();
            std::alloc::dealloc(s as *mut u8, layout);
        }
    }
}

/// @ffi_sig (String*) -> i64
/// StringStruct の長さを取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_len(s: *mut StringStruct) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).len }
}

/// @ffi_sig (String*, String*) -> String*
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

/// @ffi_sig (String*) -> String*
/// StringStruct を deep clone
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_clone(s: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if s.is_null() {
            return std::ptr::null_mut();
        }
        
        let s_str = if (*s).ptr.is_null() { 
            String::new() 
        } else { 
            CStr::from_ptr((*s).ptr).to_string_lossy().into_owned() 
        };
        
        let c_str = CString::new(s_str).unwrap();
        let ptr = c_str.into_raw();
        let len = libc::strlen(ptr) as i64;
        
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
    }
}

/// @ffi_sig (String*, String*) -> bool
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

/// @ffi_sig (String*, String*) -> bool
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

/// @ffi_sig (String*) -> i64
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

/// @ffi_sig (i64) -> String*
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

/// tl_string_from_f64: Convert f64 to string
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_f64(f: f64) -> *mut StringStruct {
    unsafe {
        make_string_struct(format!("{}", f))
    }
}

/// tl_string_from_bool: Convert bool to string
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_bool(b: bool) -> *mut StringStruct {
    unsafe {
        make_string_struct(if b { "true".to_string() } else { "false".to_string() })
    }
}

/// @ffi_sig (i64) -> String*
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

/// @ffi_sig (String*, i64) -> i64
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

/// @ffi_sig (String*) -> i64
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

/// Helper: Rust String → *mut StringStruct
pub unsafe fn make_string_struct(s: String) -> *mut StringStruct {
    unsafe {
        let c_str = CString::new(s).unwrap();
        let ptr = c_str.into_raw();
        let len = libc::strlen(ptr) as i64;
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
    }
}

/// String.trim() -> String
/// 前後の空白を除去
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_trim(s: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        make_string_struct(s_str.trim().to_string())
    }
}

/// String.starts_with(prefix: String) -> bool
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_starts_with(s: *mut StringStruct, prefix: *mut StringStruct) -> bool {
    unsafe {
        if s.is_null() || prefix.is_null() || (*s).ptr.is_null() || (*prefix).ptr.is_null() {
            return false;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let prefix_str = CStr::from_ptr((*prefix).ptr).to_string_lossy();
        s_str.starts_with(prefix_str.as_ref())
    }
}

/// String.ends_with(suffix: String) -> bool
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_ends_with(s: *mut StringStruct, suffix: *mut StringStruct) -> bool {
    unsafe {
        if s.is_null() || suffix.is_null() || (*s).ptr.is_null() || (*suffix).ptr.is_null() {
            return false;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let suffix_str = CStr::from_ptr((*suffix).ptr).to_string_lossy();
        s_str.ends_with(suffix_str.as_ref())
    }
}

/// String.replace(from: String, to: String) -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_replace(
    s: *mut StringStruct,
    from: *mut StringStruct,
    to: *mut StringStruct,
) -> *mut StringStruct {
    unsafe {
        if s.is_null() || from.is_null() || to.is_null()
            || (*s).ptr.is_null() || (*from).ptr.is_null() || (*to).ptr.is_null()
        {
            return std::ptr::null_mut();
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let from_str = CStr::from_ptr((*from).ptr).to_string_lossy();
        let to_str = CStr::from_ptr((*to).ptr).to_string_lossy();
        make_string_struct(s_str.replace(from_str.as_ref(), to_str.as_ref()))
    }
}

/// String.substring(start: i64, len: i64) -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_substring(s: *mut StringStruct, start: i64, len: i64) -> *mut StringStruct {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() || start < 0 || len < 0 {
            return make_string_struct(String::new());
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let result: String = s_str
            .chars()
            .skip(start as usize)
            .take(len as usize)
            .collect();
        make_string_struct(result)
    }
}

/// String.is_empty() -> bool
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_is_empty(s: *mut StringStruct) -> bool {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return true;
        }
        (*s).len == 0
    }
}

/// String.to_uppercase() -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_uppercase(s: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        make_string_struct(s_str.to_uppercase())
    }
}

/// String.to_lowercase() -> String
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_lowercase(s: *mut StringStruct) -> *mut StringStruct {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        make_string_struct(s_str.to_lowercase())
    }
}

/// String.index_of(needle: String) -> i64 (-1 if not found)
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_index_of(s: *mut StringStruct, needle: *mut StringStruct) -> i64 {
    unsafe {
        if s.is_null() || needle.is_null() || (*s).ptr.is_null() || (*needle).ptr.is_null() {
            return -1;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let needle_str = CStr::from_ptr((*needle).ptr).to_string_lossy();
        match s_str.find(needle_str.as_ref()) {
            Some(pos) => pos as i64,
            None => -1,
        }
    }
}

/// String.split(sep: String) -> *mut VecStruct (Vec<String>)
/// Returns a heap-allocated VecStruct { ptr, cap, len } where ptr points to an array of *mut StringStruct
#[repr(C)]
pub struct VecStruct {
    pub ptr: *mut u8,
    pub cap: i64,
    pub len: i64,
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_split(s: *mut StringStruct, sep: *mut StringStruct) -> *mut VecStruct {
    unsafe {
        let layout = std::alloc::Layout::new::<VecStruct>();
        let vec_ptr = std::alloc::alloc(layout) as *mut VecStruct;

        if s.is_null() || sep.is_null() || (*s).ptr.is_null() || (*sep).ptr.is_null() {
            (*vec_ptr).ptr = std::ptr::null_mut();
            (*vec_ptr).cap = 0;
            (*vec_ptr).len = 0;
            return vec_ptr;
        }

        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let sep_str = CStr::from_ptr((*sep).ptr).to_string_lossy();
        let parts: Vec<&str> = s_str.split(sep_str.as_ref()).collect();
        let count = parts.len() as i64;

        // Allocate array of *mut StringStruct (each pointer is 8 bytes)
        let ptr_size = std::mem::size_of::<*mut StringStruct>();
        let array_layout = std::alloc::Layout::from_size_align(ptr_size * parts.len(), 8).unwrap();
        let array_ptr = std::alloc::alloc(array_layout) as *mut *mut StringStruct;

        for (i, part) in parts.iter().enumerate() {
            let str_ptr = make_string_struct(part.to_string());
            *array_ptr.add(i) = str_ptr;
        }

        (*vec_ptr).ptr = array_ptr as *mut u8;
        (*vec_ptr).cap = count;
        (*vec_ptr).len = count;
        vec_ptr
    }
}

/// assert(cond: bool, msg: String) — aborts if cond is false
#[unsafe(no_mangle)]
pub extern "C" fn tl_assert(cond: bool, msg: *mut StringStruct) {
    if !cond {
        unsafe {
            if !msg.is_null() && !(*msg).ptr.is_null() {
                let msg_str = CStr::from_ptr((*msg).ptr).to_string_lossy();
                eprintln!("Assertion failed: {}", msg_str);
            } else {
                eprintln!("Assertion failed");
            }
        }
        std::process::exit(1);
    }
}

// ========== グローバルユーティリティ関数 ==========

/// random() -> f64 — [0.0, 1.0) の一様乱数
#[unsafe(no_mangle)]
pub extern "C" fn tl_random() -> f64 {
    use rand::Rng;
    rand::thread_rng().r#gen::<f64>()
}

/// random_int(min: i64, max: i64) -> i64 — [min, max] の一様整数乱数
#[unsafe(no_mangle)]
pub extern "C" fn tl_random_int(min: i64, max: i64) -> i64 {
    use rand::Rng;
    if min >= max {
        return min;
    }
    rand::thread_rng().gen_range(min..=max)
}

/// min(a: i64, b: i64) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_min_i64(a: i64, b: i64) -> i64 {
    if a < b { a } else { b }
}

/// max(a: i64, b: i64) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_max_i64(a: i64, b: i64) -> i64 {
    if a > b { a } else { b }
}

/// min(a: f64, b: f64) -> f64
#[unsafe(no_mangle)]
pub extern "C" fn tl_min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// max(a: f64, b: f64) -> f64
#[unsafe(no_mangle)]
pub extern "C" fn tl_max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// String.to_f64() -> f64 — 文字列を f64 にパース。失敗時は 0.0
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_f64(s: *mut StringStruct) -> f64 {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() {
            return 0.0;
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        s_str.parse::<f64>().unwrap_or(0.0)
    }
}

/// String.repeat(n: i64) -> String — 文字列を n 回繰り返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_repeat(s: *mut StringStruct, n: i64) -> *mut StringStruct {
    unsafe {
        if s.is_null() || (*s).ptr.is_null() || n <= 0 {
            return make_string_struct(String::new());
        }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let repeated = s_str.repeat(n as usize);
        make_string_struct(repeated)
    }
}

/// String.chars() -> Vec<i64> — 各文字の Unicode コードポイントを Vec に格納
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_chars(s: *mut StringStruct) -> *mut VecStruct {
    unsafe {
        let layout = std::alloc::Layout::new::<VecStruct>();
        let vec_ptr = std::alloc::alloc(layout) as *mut VecStruct;

        if s.is_null() || (*s).ptr.is_null() {
            (*vec_ptr).ptr = std::ptr::null_mut();
            (*vec_ptr).cap = 0;
            (*vec_ptr).len = 0;
            return vec_ptr;
        }

        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        let chars: Vec<i64> = s_str.chars().map(|c| c as i64).collect();
        let count = chars.len() as i64;

        // i64 の配列を割り当て
        let elem_size = std::mem::size_of::<i64>();
        let array_layout = std::alloc::Layout::from_size_align(
            elem_size * chars.len().max(1), 8
        ).unwrap();
        let array_ptr = std::alloc::alloc(array_layout) as *mut i64;

        for (i, &cp) in chars.iter().enumerate() {
            *array_ptr.add(i) = cp;
        }

        (*vec_ptr).ptr = array_ptr as *mut u8;
        (*vec_ptr).cap = count;
        (*vec_ptr).len = count;
        vec_ptr
    }
}

/// String.from_chars(chars: Vec<i64>) -> String
/// Vec<i64> の各要素を Unicode コードポイントとして1つの文字列を構築
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_chars(vec: *mut VecStruct) -> *mut StringStruct {
    unsafe {
        if vec.is_null() || (*vec).ptr.is_null() || (*vec).len <= 0 {
            return make_string_struct(String::new());
        }
        
        let len = (*vec).len as usize;
        let ptr = (*vec).ptr as *const i64;
        let mut s = String::with_capacity(len);
        
        for i in 0..len {
            let cp = *ptr.add(i);
            if let Some(ch) = char::from_u32(cp as u32) {
                s.push(ch);
            } else {
                s.push('?');
            }
        }
        make_string_struct(s)
    }
}
