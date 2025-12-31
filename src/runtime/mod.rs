pub mod device;
pub mod registry;

use crate::runtime::device::get_device;
use candle_core::Tensor;
use candle_nn; // Import candle_nn
use std::ffi::c_float;
use std::slice;

// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.
// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.
pub struct OpaqueTensor(Tensor);

// Thread-local storage for the latest gradients computed by backward()
thread_local! {
    static LATEST_GRADS: std::cell::RefCell<Option<candle_core::backprop::GradStore>> = std::cell::RefCell::new(None);
}

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
    // Workaround: Create on CPU then move to device to ensure data upload works reliably
    let t_cpu = Tensor::from_slice(data_slice, shape_slice, &candle_core::Device::Cpu).unwrap();
    let tensor = if device.is_metal() || device.is_cuda() {
        t_cpu.to_device(&device).unwrap()
    } else {
        t_cpu
    };

    Box::into_raw(Box::new(OpaqueTensor(tensor)))
}

// Function to create a tensor that requires gradients (similar to Var, but returning Tensor)
// Actually Var is a wrapper around Tensor. For backprop to work, we usually use Var or
// create a tensor and then watch it? Candle's graph is dynamic.
// `Tensor::new` creates a leaf.
// To support `requires_grad`, we effectively need the tensor to be a variable in the graph.
// In Candle, `Var` is for trainable weights.
// Let's explicitly add a helper for creating vars.

#[no_mangle]
pub extern "C" fn tl_tensor_randn(
    rank: usize,
    shape: *const usize,
    requires_grad: bool,
) -> *mut OpaqueTensor {
    // Basic randn implementation
    let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };
    let device = get_device();

    // Create random tensor
    let t = Tensor::randn(0.0f32, 1.0f32, shape_slice, &device).unwrap();

    if requires_grad {
        // In Candle, to track gradients for a leaf node, we use `Var`.
        // But `Var` is a distinct type. `Var` derefs to `Tensor`.
        // If we want backprop from this node, we should probably turn it into a Var,
        // then take its tensor.
        // BUT: Simply taking `var.as_tensor()` gives a tensor that tracks the id.
        // The issue is keeping the `Var` alive.
        // For simplicity in this FFI, let's cheat:
        // We just return the tensor. If Candle requires `Var` struct to be kept alive, this leaks semantics.
        // Checking Candle source: `Var` holds `Tensor` and shares an ID.
        // `Tensor` itself doesn't naturally "require grad" unless it's an operation result or a Var.
        // Actually, `Var` is what you want for weights.
        // Let's create a Var, extract the Tensor, and rely on the fact that
        // the Tensor copies the storage and ID.
        // WAIT: If we drop the Var, does the ID persist in the graph?
        // Candle's `Var` is basically `Arc<Mutex<SimpleVar>>`.
        // If we drop it, the "variable-ness" might be lost if nothing else holds it.
        // However, for `backward` to work, the graph is built forward.
        // If we use the tensor in ops, the ops record the dependencies.
        // So just returning `var.as_tensor().clone()` should start the graph.
        let var = candle_core::Var::from_tensor(&t).unwrap();
        let t_var = var.as_tensor().clone();
        // Leak the Var to ensure it stays alive for the graph
        std::mem::forget(var);
        // We leak the Var? Use it just to start the trace?
        // Actually `var.as_tensor()` returns a tensor linked to the Var's ID.
        // If we perform ops on `t_var`, they get tracked.
        Box::into_raw(Box::new(OpaqueTensor(t_var)))
    } else {
        Box::into_raw(Box::new(OpaqueTensor(t)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_add(t_b)
            .unwrap_or_else(|_| t_a.add(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_sub(t_b)
            .unwrap_or_else(|_| t_a.sub(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_mul(t_b)
            .unwrap_or_else(|_| t_a.mul(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // dim is i64, convert to usize or generic dim
        // Support negative indexing if possible? Candle supports D::Minus1 etc but usually via specific Enums or usize.
        // For now assume positive usize or handle -1 for last?
        // Candle's softmax takes usize usually.
        // Let's coerce to usize for now. User needs to pass positive.
        let d = dim as usize;
        let result = candle_nn::ops::softmax(tensor, d).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_cross_entropy(
    logits: *mut OpaqueTensor,
    targets: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let l = &(*logits).0;
        let t = &(*targets).0;

        // Expect targets to be F32 (0.0, 1.0, 2.0...) -> Cast to U32 for indices
        let t_u32 = t.to_dtype(candle_core::DType::U32).unwrap();

        // Log Softmax on last dim (-1)
        // candle_nn::ops::log_softmax takes (tensor, dim)
        let log_sm = candle_nn::ops::log_softmax(l, candle_core::D::Minus1).unwrap();

        // NLL
        let loss = candle_nn::loss::nll(&log_sm, &t_u32).unwrap();

        Box::into_raw(Box::new(OpaqueTensor(loss)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_detach(t: *mut OpaqueTensor, req_grad: bool) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let detached = tensor.detach();
        if req_grad {
            let var = candle_core::Var::from_tensor(&detached).unwrap();
            let t_ref = var.as_tensor().clone();
            std::mem::forget(var);
            Box::into_raw(Box::new(OpaqueTensor(t_ref)))
        } else {
            Box::into_raw(Box::new(OpaqueTensor(detached)))
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_enable_grad(t: *mut OpaqueTensor) {
    unsafe {
        let tensor = &(*t).0;
        // detached copy
        let detached = tensor.detach();
        // Create var
        let var = candle_core::Var::from_tensor(&detached).unwrap();
        let t_var = var.as_tensor().clone();
        std::mem::forget(var);

        // Update pointer
        (*t).0 = t_var;
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
pub extern "C" fn tl_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        unsafe {
            let _ = Box::from_raw(t);
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_clone(t: *const OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let cloned = tensor.clone();
        Box::into_raw(Box::new(OpaqueTensor(cloned)))
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
pub extern "C" fn tl_print_string(s: *const std::os::raw::c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        if let Ok(c_str) = std::ffi::CStr::from_ptr(s).to_str() {
            println!("{}", c_str);
        }
    }
}

pub fn force_link() {
    let _ = tl_print_string as *const ();
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

#[no_mangle]
pub extern "C" fn tl_tensor_transpose(
    t: *mut OpaqueTensor,
    dim0: usize,
    dim1: usize,
) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.transpose(dim0, dim1).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_backward(t: *mut OpaqueTensor) {
    unsafe {
        let tensor = &(*t).0;
        // Perform backpropagation
        if let Ok(grads) = tensor.backward() {
            // Store gradients in thread-local storage, replacing any previous ones
            LATEST_GRADS.with(|g| {
                *g.borrow_mut() = Some(grads);
            });
        } else {
            // Error handling or fallback?
            eprintln!(
                "Runtime Error: backward() failed. Ensure the tensor behaves like a scalar loss."
            );
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Retrieve gradient for this tensor from LATEST_GRADS
        let grad = LATEST_GRADS.with(|g| {
            let borrow = g.borrow();
            if let Some(store) = &*borrow {
                store.get(tensor).cloned()
            } else {
                None
            }
        });

        if let Some(g) = grad {
            Box::into_raw(Box::new(OpaqueTensor(g)))
        } else {
            // Return null pointer if no gradient found
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sum_all().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sub_assign(ref_t: *mut OpaqueTensor, val_t: *mut OpaqueTensor) {
    // In-place subtraction: ref_t -= val_t
    // Note: Candle Tensors are immutable, but Vars are mutable.
    // However, we are using OpaqueTensor which wraps Tensor.
    // If we want to support SGD update `w -= lr * g`, we ideally need `Var`.
    // But for now, since our JIT pointers are mutable pointers to OpaqueTensor,
    // we can update the content of OpaqueTensor to point to a new Tensor.
    // This effectively effectively mutates the variable from the perspective of the JIT.
    unsafe {
        let t_dst = &(*ref_t).0;
        let t_src = &(*val_t).0;

        // Compute new value: dst - src
        // Using broadcast_sub to be safe
        let result = t_dst
            .broadcast_sub(t_src)
            .unwrap_or_else(|_| t_dst.sub(t_src).unwrap());

        // Update the OpaqueTensor to hold the new tensor
        (*ref_t).0 = result;
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_mul_assign(ref_t: *mut OpaqueTensor, val_t: *mut OpaqueTensor) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let t_src = &(*val_t).0;
        let result = t_dst
            .broadcast_mul(t_src)
            .unwrap_or_else(|_| t_dst.mul(t_src).unwrap());
        (*ref_t).0 = result;
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_div_assign(ref_t: *mut OpaqueTensor, val_t: *mut OpaqueTensor) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let t_src = &(*val_t).0;
        let result = t_dst
            .broadcast_div(t_src)
            .unwrap_or_else(|_| t_dst.div(t_src).unwrap());
        (*ref_t).0 = result;
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_add_assign(ref_t: *mut OpaqueTensor, val_t: *mut OpaqueTensor) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let t_src = &(*val_t).0;
        let result = t_dst
            .broadcast_add(t_src)
            .unwrap_or_else(|_| t_dst.add(t_src).unwrap());
        (*ref_t).0 = result;
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_reshape(
    t: *mut OpaqueTensor,
    shape_tensor: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let shape_t = &(*shape_tensor).0;
        // Convert shape tensor to Vec<usize>
        // Assuming shape tensor fits in memory and is 1D.
        // We need to get data back to CPU if on GPU.
        let shape_vec: Vec<f32> = shape_t.flatten_all().unwrap().to_vec1().unwrap();
        let new_shape: Vec<usize> = shape_vec.iter().map(|&x| x as usize).collect();

        let result = tensor.reshape(new_shape).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_div(t_b)
            .unwrap_or_else(|_| t_a.div(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.exp().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_pow(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        // Broadcasting pow
        let result = t_a
            .broadcast_pow(t_b)
            .unwrap_or_else(|_| t_a.pow(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.log().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sqrt().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_matmul(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a.matmul(t_b).unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result)))
    }
}
