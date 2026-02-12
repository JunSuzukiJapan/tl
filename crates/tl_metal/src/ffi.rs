//! Metal FFI functions
//!
//! tl_runtime から呼び出される Metal バックエンドの FFI エントリポイント。
//! CPU バックエンド (tl_cpu) と同じシグネチャを持ち、JIT リンク時に切り替えられる。

use crate::tensor::MetalTensor;

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;

// OpaqueTensor は MetalTensor のエイリアスとして扱う
type OpaqueTensor = MetalTensor;

/// テンソルを Arc で包んでポインタを返す（V5.0 メモリ管理）
fn make_tensor(t: MetalTensor) -> *mut OpaqueTensor {
    let arc = Arc::new(UnsafeCell::new(t));
    Arc::into_raw(arc) as *mut OpaqueTensor
}

#[no_mangle]
pub extern "C" fn tl_metal_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.clone())
}

#[no_mangle]
pub extern "C" fn tl_metal_release(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    // Arc::from_raw で復元し drop。RC-1、RC=0 で MetalTensor が Drop される。
    unsafe { let _ = Arc::from_raw(t as *const UnsafeCell<MetalTensor>); }
}

#[no_mangle]
pub extern "C" fn tl_metal_numel(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    tensor.shape().iter().product::<usize>() as i64
}

#[no_mangle]
pub extern "C" fn tl_metal_data(t: *mut OpaqueTensor) -> *mut c_void {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    
    // WARNING: MetalTensor は GPU 上にあるため、データポインタを直接返すことはできない。
    // ここではデバッグ/互換性のため、GPU -> CPU コピーを行い、そのリークしたポインタを返す。
    // これはメモリリークを引き起こすため、頻繁に呼ぶべきではない。
    // 本来は tl_metal_to_cpu 等を使うべき。
    
    let vec: Vec<f32> = tensor.to_vec(); // GPU -> CPU sync copy, explicit f32
    let mut boxed_slice = vec.into_boxed_slice();
    let ptr = boxed_slice.as_mut_ptr();
    std::mem::forget(boxed_slice); // Leak memory to keep pointer valid
    ptr as *mut c_void
}
