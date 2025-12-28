pub mod device;

use crate::runtime::device::get_device;
use candle_core::Tensor;
use std::ffi::c_float;
use std::slice;

// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.
pub struct OpaqueTensor(Tensor);

#[no_mangle]
pub extern "C" fn tl_tensor_new(
    data: *const c_float,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };
    let num_elements: usize = shape_slice.iter().product();

    let data_slice = unsafe { slice::from_raw_parts(data, num_elements) };

    let device = get_device();
    // Create tensor from slice
    let tensor = Tensor::from_slice(data_slice, shape_slice, &device).unwrap();

    Box::into_raw(Box::new(OpaqueTensor(tensor)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = (t_a + t_b).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = (t_a * t_b).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_print(t: *mut OpaqueTensor) {
    unsafe {
        let tensor = &(*t).0;
        println!("{}", tensor);
    }
}

// Memory management
#[no_mangle]
pub extern "C" fn tl_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        unsafe {
            let _ = Box::from_raw(t);
        }
    }
}
