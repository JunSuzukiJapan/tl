//! コマンドライン引数管理モジュール
//! TLプログラムからコマンドライン引数にアクセスするためのランタイム関数を提供。

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::{LazyLock, RwLock};

/// グローバル引数ストレージ
static ARGS: LazyLock<RwLock<Vec<String>>> = LazyLock::new(|| RwLock::new(Vec::new()));

/// TLプログラム実行前にコマンドライン引数を初期化
/// main.rsから呼び出される
pub fn init_args(args: Vec<String>) {
    if let Ok(mut stored) = ARGS.write() {
        *stored = args;
    }
}

/// 引数の数を返す
#[unsafe(no_mangle)]
/// @ffi_sig () -> i64
pub extern "C" fn tl_args_count() -> i64 {
    ARGS.read().map(|a| a.len() as i64).unwrap_or(0)
}

/// 指定インデックスの引数を返す
/// インデックスが範囲外の場合はNULLを返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_args_get(index: i64) -> *mut c_char {
    if index < 0 {
        return std::ptr::null_mut();
    }

    let idx = index as usize;
    if let Ok(args) = ARGS.read() {
        if idx < args.len() {
            if let Ok(c_str) = CString::new(args[idx].clone()) {
                return c_str.into_raw();
            }
        }
    }
    std::ptr::null_mut()
}
