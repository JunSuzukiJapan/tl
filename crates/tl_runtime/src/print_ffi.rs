//! Print 関連の FFI 関数

use crate::string_ffi::StringStruct;
use crate::OpaqueTensor;
use std::ffi::CStr;
use std::io::Write;

/// 文字列を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_string(s: *mut StringStruct) {
    unsafe {
        if !s.is_null() && !(*s).ptr.is_null() {
            let c_str = CStr::from_ptr((*s).ptr);
            print!("{}", c_str.to_string_lossy());
            let _ = std::io::stdout().flush();
        }
    }
}

/// 文字列を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_string(s: *mut StringStruct) {
    unsafe {
        if !s.is_null() && !(*s).ptr.is_null() {
            let c_str = CStr::from_ptr((*s).ptr);
            println!("{}", c_str.to_string_lossy());
        }
    }
}

/// tl_string_print は tl_print_string のエイリアス
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_print(s: *mut StringStruct) {
    tl_print_string(s);
}

/// 整数を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_i64(i: i64) {
    println!("{}", i);
}

/// 浮動小数点数を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_f64(f: f64) {
    println!("{}", f);
}

/// ブール値を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_bool(b: bool) {
    println!("{}", b);
}

/// 整数を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_i64(i: i64) {
    print!("{}", i);
    let _ = std::io::stdout().flush();
}

/// 浮動小数点数を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_f64(f: f64) {
    print!("{}", f);
    let _ = std::io::stdout().flush();
}

/// ブール値を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_bool(b: bool) {
    print!("{}", b);
    let _ = std::io::stdout().flush();
}

/// ポインタを出力
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_ptr(p: *mut std::ffi::c_void) {
    println!("{:p}", p);
}

/// テンソルを出力
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_display(t: *mut OpaqueTensor) {
    if t.is_null() {
        print!("Tensor[null]");
        return;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tensor = unsafe { &*t };
        let data: Vec<f32> = tensor.to_vec();
        if data.len() <= 10 {
            print!("{:?}", data);
        } else {
            let preview: Vec<f32> = data.iter().take(5).cloned().collect();
            print!("{:?}...({} elements)", preview, data.len());
        }
    }));
    if result.is_err() {
        print!("Tensor[invalid]");
    }
    let _ = std::io::stdout().flush();
}

// ========== i32/f32 Variants ==========

/// 文字を出力（改行付き）- i32 として受け取り char として表示
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_char(c: i32) {
    if let Some(ch) = char::from_u32(c as u32) {
        println!("{}", ch);
    } else {
        println!("?");
    }
}

/// 文字を出力（改行なし）- i32 として受け取り char として表示
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_char(c: i32) {
    if let Some(ch) = char::from_u32(c as u32) {
        print!("{}", ch);
    } else {
        print!("?");
    }
    let _ = std::io::stdout().flush();
}

/// i32 を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_i32(i: i32) {
    println!("{}", i);
}

/// f32 を出力（改行付き）
#[unsafe(no_mangle)]
pub extern "C" fn tl_print_f32(f: f32) {
    println!("{}", f);
}

/// i32 を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_i32(i: i32) {
    print!("{}", i);
    let _ = std::io::stdout().flush();
}

/// f32 を出力（改行なし）
#[unsafe(no_mangle)]
pub extern "C" fn tl_display_f32(f: f32) {
    print!("{}", f);
    let _ = std::io::stdout().flush();
}

// ========== Tensor Print Variants ==========

/// テンソルを出力（1引数版）- 名前とテンソル
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_1(t: *mut OpaqueTensor) {
    if t.is_null() {
        println!("Tensor: [null]");
        return;
    }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    let data: Vec<f32> = tensor.to_vec();
    println!("Tensor shape={:?} data={:?}", shape, &data[..data.len().min(20)]);
}

/// テンソルを出力（2引数版）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_2(name: *mut StringStruct, t: *mut OpaqueTensor) {
    unsafe {
        let name_str = if name.is_null() || (*name).ptr.is_null() {
            "Tensor".to_string()
        } else {
            CStr::from_ptr((*name).ptr).to_string_lossy().into_owned()
        };
        if t.is_null() {
            println!("{}: [null]", name_str);
            return;
        }
        let tensor = &*t;
        let shape = tensor.shape();
        let data: Vec<f32> = tensor.to_vec();
        println!("{} shape={:?} data={:?}", name_str, shape, &data[..data.len().min(20)]);
    }
}

/// テンソルを出力（3引数版）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_3(name: *mut StringStruct, t: *mut OpaqueTensor, max_items: i64) {
    unsafe {
        let name_str = if name.is_null() || (*name).ptr.is_null() {
            "Tensor".to_string()
        } else {
            CStr::from_ptr((*name).ptr).to_string_lossy().into_owned()
        };
        if t.is_null() {
            println!("{}: [null]", name_str);
            return;
        }
        let tensor = &*t;
        let shape = tensor.shape();
        let data: Vec<f32> = tensor.to_vec();
        let max = max_items.max(1) as usize;
        println!("{} shape={:?} data={:?}", name_str, shape, &data[..data.len().min(max)]);
    }
}
