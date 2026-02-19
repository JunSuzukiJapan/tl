//! CUDA FFI functions
//!
//! tl_runtime から呼び出される CUDA バックエンドの FFI エントリポイント。

use crate::tensor::CudaTensor;
use std::ffi::c_void;

// OpaqueTensor は CudaTensor のエイリアスとして扱う
pub type OpaqueTensor = CudaTensor;

#[no_mangle]
pub extern "C" fn tl_cuda_clone(_t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unimplemented!("tl_cuda_clone")
}

#[no_mangle]
pub extern "C" fn tl_cuda_shallow_clone(_t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unimplemented!("tl_cuda_shallow_clone")
}

#[no_mangle]
pub extern "C" fn tl_cuda_release(_t: *mut OpaqueTensor) {
    unimplemented!("tl_cuda_release")
}

#[no_mangle]
pub extern "C" fn tl_cuda_numel(_t: *mut OpaqueTensor) -> i64 {
    unimplemented!("tl_cuda_numel")
}

#[no_mangle]
pub extern "C" fn tl_cuda_data(_t: *mut OpaqueTensor) -> *mut c_void {
    unimplemented!("tl_cuda_data")
}
