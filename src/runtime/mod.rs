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

#[no_mangle]
pub extern "C" fn tl_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.neg().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_len(t: *mut OpaqueTensor) -> i64 {
    unsafe {
        let tensor = &(*t).0;
        // Return size of the first dimension, or total elements?
        // For 1D loop, dims[0] is appropriate.
        if tensor.rank() > 0 {
            tensor.dims()[0] as i64
        } else {
            1 // Scalar has 'length' 1? or 0?
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_get(t: *mut OpaqueTensor, idx: i64) -> c_float {
    unsafe {
        let tensor = &(*t).0;
        // Naive implementation: assume 1D or flat index
        // To get scalar, we can reshape to 1D and get.
        // For now, assume tensor is 1D or we want flat index.
        let val: f32 = tensor
            .flatten_all()
            .unwrap()
            .get(idx as usize)
            .unwrap()
            .to_scalar()
            .unwrap();
        val
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_slice(t: *mut OpaqueTensor, start: i64, len: i64) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Slice along first dimension
        let result = tensor.narrow(0, start as usize, len as usize).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_print_i64(v: i64) {
    println!("{}", v);
}

#[no_mangle]
pub extern "C" fn tl_print_f32(v: c_float) {
    println!("{}", v);
}

#[no_mangle]
pub extern "C" fn tl_tensor_dim(t: *mut OpaqueTensor, dim_idx: usize) -> i64 {
    unsafe {
        let tensor = &(*t).0;
        if dim_idx >= tensor.rank() {
            0
        } else {
            tensor.dims()[dim_idx] as i64
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_get_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
) -> c_float {
    unsafe {
        let tensor = &(*t).0;
        let idxs = slice::from_raw_parts(indices, rank);
        let idxs_usize: Vec<usize> = idxs.iter().map(|&x| x as usize).collect();

        // Calculate flat index based on dimensions
        let dims = tensor.dims();
        let mut flat_idx = 0;
        let mut stride = 1;
        for i in (0..rank).rev() {
            flat_idx += idxs_usize[i] * stride;
            stride *= dims[i];
        }

        let val = tensor
            .flatten_all()
            .unwrap()
            .get(flat_idx)
            .unwrap()
            .to_scalar()
            .unwrap();
        val
    }
}
