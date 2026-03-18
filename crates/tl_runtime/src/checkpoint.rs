//! Gradient Checkpoint モジュール
//!
//! Param::checkpoint(obj.method, input) の実装。
//! 現在はパススルー実装（メソッドを直接呼び出す）。
//! 将来的にはメモリ最適化（forward 時に中間値を破棄し、backward 時に再計算）を追加予定。

use std::ffi::c_void;

/// チェックポイント関数の型定義
/// TL のメソッドは (sret_dest: *mut, self_ptr: *mut, input: *mut) の形式で呼ばれ、
/// sret パターンで結果を返す。
/// ただし、Tensor を返す場合は sret を使わず *mut を直接返す。
type TlMethodFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;

/// tl_checkpoint(ctx: *mut, func: *mut, input: *mut) -> *mut
///
/// コンパイラが生成するコード:
/// - ctx: 構造体 (self) のポインタ
/// - func: メソッドの関数ポインタ (fn(self_ptr, input) -> tensor_ptr)
/// - input: 入力テンソルのポインタ
///
/// パススルー実装: func(ctx, input) を直接呼び出して結果を返す。
#[unsafe(no_mangle)]
pub extern "C" fn tl_checkpoint(
    ctx: *mut c_void,
    func: *mut c_void,
    input: *mut c_void,
) -> *mut c_void {
    if func.is_null() {
        eprintln!("[checkpoint] Error: null function pointer");
        return input; // Fallback: return input unchanged
    }
    if ctx.is_null() {
        eprintln!("[checkpoint] Error: null context pointer");
        return input;
    }

    // Cast func to the method function pointer type and call it
    let method: TlMethodFn = unsafe { std::mem::transmute(func) };
    let result = unsafe { method(ctx, input) };

    result
}
