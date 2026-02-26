//! CUDA FFI functions
//!
//! tl_runtime から呼び出される CUDA バックエンドの FFI エントリポイント。
//! Metal バックエンドの ffi.rs と同じ V5.0 Arc ベースメモリ管理。

use crate::tensor::CudaTensor;

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;

// OpaqueTensor は CudaTensor のエイリアスとして扱う
pub type OpaqueTensor = CudaTensor;

#[no_mangle]
pub extern "C" fn tl_cuda_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<CudaTensor>);
        let cloned = arc.clone();
        let _ = Arc::into_raw(arc); // Keep original alive
        Arc::into_raw(cloned) as *mut OpaqueTensor
    }
}

#[no_mangle]
pub extern "C" fn tl_cuda_shallow_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<CudaTensor>);
        let cloned = arc.clone();
        let _ = Arc::into_raw(arc); // Keep original alive
        Arc::into_raw(cloned) as *mut OpaqueTensor
    }
}

#[no_mangle]
pub extern "C" fn tl_cuda_release(t: *mut OpaqueTensor) {
    if t.is_null() {
        return;
    }
    crate::ffi_ops::release_if_live(t);
}

#[no_mangle]
pub extern "C" fn tl_cuda_numel(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    tensor.shape().iter().product::<usize>() as i64
}

#[no_mangle]
pub extern "C" fn tl_cuda_data(t: *mut OpaqueTensor) -> *mut c_void {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };

    // GPU -> CPU コピーを行い、スレッドローカルキャッシュに保持。
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
