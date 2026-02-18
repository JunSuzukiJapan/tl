//! Metal FFI functions
//!
//! tl_runtime から呼び出される Metal バックエンドの FFI エントリポイント。
//! CPU バックエンド (tl_cpu) と同じシグネチャを持ち、JIT リンク時に切り替えられる。

use crate::tensor::MetalTensor;

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;

// OpaqueTensor は MetalTensor のエイリアスとして扱う
pub type OpaqueTensor = MetalTensor;

/// テンソルを Arc で包んでポインタを返す（V5.0 メモリ管理）
// fn make_tensor(t: MetalTensor) -> *mut OpaqueTensor {
//     let arc = Arc::new(UnsafeCell::new(t));
//     Arc::into_raw(arc) as *mut OpaqueTensor
// }

#[no_mangle]
pub extern "C" fn tl_metal_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<MetalTensor>);
        let cloned = arc.clone();
        let _ = Arc::into_raw(arc); // Keep original alive
        Arc::into_raw(cloned) as *mut OpaqueTensor
    }
}

#[no_mangle]
pub extern "C" fn tl_metal_shallow_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<MetalTensor>);
        let cloned = arc.clone();
        let _ = Arc::into_raw(arc); // Keep original alive
        Arc::into_raw(cloned) as *mut OpaqueTensor
    }
}

#[no_mangle]
pub extern "C" fn tl_metal_release(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    // Arc RC-1: release_if_live は直接 Arc::from_raw → drop
    crate::ffi_ops::release_if_live(t);
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
    
    // GPU -> CPU コピーを行い、スレッドローカルキャッシュに保持。
    // 前回のバッファは自動解放されるため、mem::forget によるリークを回避。
    thread_local! {
        static LAST_DATA_BUFFER: std::cell::RefCell<Option<Box<[f32]>>> = 
            std::cell::RefCell::new(None);
    }
    
    let vec: Vec<f32> = tensor.to_vec();
    let boxed_slice = vec.into_boxed_slice();
    let ptr = boxed_slice.as_ptr() as *mut c_void;
    LAST_DATA_BUFFER.with(|cell| {
        *cell.borrow_mut() = Some(boxed_slice);
    });
    ptr
}
