pub mod device;
pub mod registry;
pub mod stdlib; // Add this

use crate::runtime::device::get_device;
use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn; // Import candle_nn
use std::cell::RefCell;
use std::ffi::c_float;
use std::slice;
use std::sync::{Arc, Mutex};

// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.

// Thread-local storage for the latest gradients computed by backward()
thread_local! {
    static LATEST_GRADS: RefCell<Option<candle_core::backprop::GradStore>> = RefCell::new(None);
}

// Global VarMap for tracking all trainable parameters
lazy_static::lazy_static! {
    static ref GLOBAL_VAR_MAP: Mutex<candle_nn::VarMap> = Mutex::new(candle_nn::VarMap::new());
}

/// OpaqueTensor wraps a Candle Tensor and optionally a Var for gradient tracking
/// The Var is stored as Arc to allow sharing across clones while keeping the same variable alive
/// The name is used to register the Var in the global VarMap
#[repr(C)]
pub struct OpaqueTensor(Tensor, Option<Arc<candle_core::Var>>, Option<String>);

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

    Box::into_raw(Box::new(OpaqueTensor(tensor, None, None)))
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
        // Create a Var and store it in Arc to keep it alive
        // The Var maintains the gradient tracking information
        let var = candle_core::Var::from_tensor(&t).unwrap();
        let t_var = var.as_tensor().clone();

        // Store Var in Arc so it can be shared across clones
        let var_arc = Arc::new(var);
        Box::into_raw(Box::new(OpaqueTensor(t_var, Some(var_arc), None)))
    } else {
        Box::into_raw(Box::new(OpaqueTensor(t, None, None)))
    }
}

// VarBuilder-based parameter management (following Candle's official pattern)
// This allows proper gradient tracking for parameters stored in struct fields

/// Create or retrieve a parameter from the global VarMap
/// This is the key to proper gradient tracking - all trainable parameters
/// must be created through this function to ensure they are registered in VarMap
#[no_mangle]
pub extern "C" fn tl_varbuilder_get(
    name: *const std::os::raw::c_char,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    use std::ffi::CStr;

    let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap().to_string() };
    let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };

    let varmap = GLOBAL_VAR_MAP.lock().unwrap();
    let device = get_device();

    // VarMap.data() returns &Mutex<HashMap<String, Var>>
    let data = varmap.data();
    let mut data_guard = data.lock().unwrap();

    // Check if this parameter already exists
    if !data_guard.contains_key(&name_str) {
        // Create new parameter with random initialization
        let tensor = Tensor::randn(0.0f32, 1.0f32, shape_slice, &device).unwrap();
        let var = candle_core::Var::from_tensor(&tensor).unwrap();
        data_guard.insert(name_str.clone(), var);
    }

    // Get the Var from VarMap
    let var = data_guard.get(&name_str).unwrap().clone();
    drop(data_guard); // Release lock
    drop(varmap); // Release varmap lock

    let tensor = var.as_tensor().clone();

    // Store Var reference in Arc so it persists across clones
    let var_arc = Arc::new(var);

    Box::into_raw(Box::new(OpaqueTensor(
        tensor,
        Some(var_arc),
        Some(name_str),
    )))
}

/// Update all parameters in VarMap using SGD
#[no_mangle]
pub extern "C" fn tl_update_all_params(learning_rate: f32) {
    let varmap = GLOBAL_VAR_MAP.lock().unwrap();
    let data = varmap.data();
    let data_guard = data.lock().unwrap();

    LATEST_GRADS.with(|g| {
        if let Some(grads) = &*g.borrow() {
            for (name, var) in data_guard.iter() {
                if let Some(grad) = grads.get(var.as_tensor()) {
                    // SGD update: param = param - lr * grad
                    let updated = (var.as_tensor() - (grad * learning_rate as f64).unwrap())
                        .unwrap()
                        .detach();

                    // Update the Var with new value
                    var.set(&updated).unwrap();

                    println!("Updated param '{}': shape {:?}", name, updated.shape());
                }
            }
        }
    });
}

/// Get gradient for a specific parameter by name
#[no_mangle]
pub extern "C" fn tl_varbuilder_grad(name: *const std::os::raw::c_char) -> *mut OpaqueTensor {
    use std::ffi::CStr;

    let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap() };

    let varmap = GLOBAL_VAR_MAP.lock().unwrap();
    let data = varmap.data();
    let data_guard = data.lock().unwrap();

    if let Some(var) = data_guard.get(name_str) {
        let grad = LATEST_GRADS.with(|g| {
            g.borrow()
                .as_ref()
                .and_then(|store| store.get(var.as_tensor()).cloned())
        });

        if let Some(g) = grad {
            return Box::into_raw(Box::new(OpaqueTensor(g, None, None)));
        }
    }

    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_add(t_b)
            .unwrap_or_else(|_| t_a.add(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        if a.is_null() || b.is_null() {
            panic!(
                "null pointer dereference occurred in tl_tensor_mul: a={:p} b={:p}",
                a, b
            );
        }
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_mul(t_b)
            .unwrap_or_else(|_| t_a.mul(t_b).unwrap());
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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

        Box::into_raw(Box::new(OpaqueTensor(loss, None, None)))
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
            Box::into_raw(Box::new(OpaqueTensor(t_ref, None, None)))
        } else {
            Box::into_raw(Box::new(OpaqueTensor(detached, None, None)))
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
        let var_ref = &(*t).1;
        let cloned = tensor.clone();

        // Clone the Arc<Var> if it exists to maintain gradient tracking
        let cloned_var = var_ref.as_ref().map(|v| Arc::clone(v));

        Box::into_raw(Box::new(OpaqueTensor(cloned, cloned_var, None)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.neg().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        let var_ref = &(*t).1;

        use std::io::Write;
        println!(
            "DEBUG: grad() for tensor {:p} shape={:?}",
            t,
            tensor.shape()
        );
        std::io::stdout().flush().unwrap();

        // If this tensor has an associated Var, get the gradient from it directly
        // TODO: Candle's Var doesn't expose a direct grad() method
        // Need to investigate VarMap-based approach or alternative API
        if let Some(_var_arc) = var_ref {
            println!("DEBUG: Tensor has Var, but cannot access grad() - API limitation");
            std::io::stdout().flush().unwrap();

            // Candle's Var API doesn't provide direct gradient access
            // Gradients must be retrieved through GradStore after backward()
            // This is a known limitation requiring further investigation
        }

        // Fallback: Retrieve gradient from LATEST_GRADS (for non-Var tensors or intermediate results)
        let grad = LATEST_GRADS.with(|g| {
            let borrow = g.borrow();
            if let Some(store) = &*borrow {
                store.get(tensor).cloned()
            } else {
                println!("DEBUG: No grads in LATEST_GRADS");
                None
            }
        });

        if let Some(g) = grad {
            println!("DEBUG: Grad FOUND from GradStore for tensor {:p}", t);
            Box::into_raw(Box::new(OpaqueTensor(g, None, None)))
        } else {
            println!("DEBUG: Grad NOT found for tensor {:p} - returning NULL", t);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sum_all().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_reshape_dims(
    t: *mut OpaqueTensor,
    dims: *const i64,
    num_dims: i64,
) -> *mut OpaqueTensor {
    unsafe {
        // use std::io::Write;
        // println!("DEBUG: reshape_dims t={:p} dims={:p} num={}", t, dims, num_dims);
        // std::io::stdout().flush().unwrap();
        if t.is_null() || dims.is_null() {
            panic!("Null pointer passed to reshape_dims");
        }
        let tensor = &(*t).0;
        let dims_slice = std::slice::from_raw_parts(dims, num_dims as usize);
        // println!("DEBUG: dims_slice={:?}", dims_slice);
        // std::io::stdout().flush().unwrap();
        let new_shape: Vec<usize> = dims_slice.iter().map(|&x| x as usize).collect();
        // println!("DEBUG: new_shape={:?}", new_shape);
        // std::io::stdout().flush().unwrap();
        match tensor.reshape(new_shape) {
            Ok(result) => Box::into_raw(Box::new(OpaqueTensor(result, None, None))),
            Err(e) => {
                println!("Error reshaping tensor: {}", e);
                std::process::abort();
            }
        }
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.exp().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.log().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}

// --- Transformer Support ---

#[no_mangle]
pub extern "C" fn tl_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.sin().unwrap();
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.cos().unwrap();
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.relu().unwrap();
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.gelu().unwrap();
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

// Lower triangular mask (diagonal=0 includes the diagonal)
#[no_mangle]
pub extern "C" fn tl_tensor_tril(t: *mut OpaqueTensor, diagonal: i32) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let dims = t.dims();
    let len = dims.len();
    if len < 2 {
        // Scalar or 1D, return as is? Or convention?
        // Usually tril is for 2D+.
        return Box::into_raw(Box::new(OpaqueTensor(t.clone(), None, None)));
    }
    let h = dims[len - 2];
    let w = dims[len - 1];
    let dev = t.device();

    // Create mask: col <= row + diagonal
    let r = Tensor::arange(0u32, h as u32, dev)
        .unwrap()
        .unsqueeze(1)
        .unwrap();
    let c = Tensor::arange(0u32, w as u32, dev)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Convert to I64 for safe arithmetic
    let r = r.to_dtype(candle_core::DType::I64).unwrap();
    let c = c.to_dtype(candle_core::DType::I64).unwrap();
    let diag = Tensor::new(diagonal as i64, dev).unwrap();

    let boundary = r.broadcast_add(&diag).unwrap();

    // Explicit broadcast for comparison [1, W] vs [H, 1] -> [H, W]
    let h_dim = h as usize;
    let w_dim = w as usize;
    let target_shape = vec![h_dim, w_dim];

    let c_b = c.broadcast_as(&target_shape[..]).unwrap();
    let b_b = boundary.broadcast_as(&target_shape[..]).unwrap();
    let mask = c_b.le(&b_b).unwrap();

    let mask = mask.to_dtype(t.dtype()).unwrap();
    let res = t.broadcast_mul(&mask).unwrap();
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_sum_dim(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    // Candle sum: if keep_dim is true, it retains the dim.
    let res = if keep_dim {
        t.sum_keepdim(dim).unwrap()
    } else {
        t.sum(dim).unwrap()
    };
    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

// Embedding: indices [B, S], weights [V, D] -> [B, S, D]
#[no_mangle]
pub extern "C" fn tl_tensor_embedding(
    indices: *mut OpaqueTensor,
    weights: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    let indices = unsafe { &(*indices).0 };
    let weights = unsafe { &(*weights).0 };

    // Cast indices to U32 (required/safer for index_select)
    // Input indices might be F32 if coming from literals.
    let indices_u32 = indices.to_dtype(candle_core::DType::U32).unwrap();

    // 1. Flatten indices to 1D
    let flat_indices = indices_u32.flatten_all().unwrap();
    // 2. Index Select on dim 0 of weights (gather)
    let gathered = weights.index_select(&flat_indices, 0).unwrap();

    // 3. Reshape result to [indices.shape, weights.dim(-1)]
    let mut new_shape = indices.dims().to_vec();
    new_shape.push(weights.dim(candle_core::D::Minus1).unwrap());

    let res = gathered.reshape(new_shape).unwrap();

    Box::into_raw(Box::new(OpaqueTensor(res, None, None)))
}

#[no_mangle]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sqrt().unwrap();
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
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
        // Use broadcast_matmul to support [B, S, D] x [D, O] -> [B, S, O]
        let result = t_a.broadcast_matmul(t_b).unwrap_or_else(|_| {
            // Fallback to strict matmul if broadcast fails or if not needed?
            // Actually broadcast_matmul typically covers strict matmul too.
            // But let's unwrap and panic with message if fails.
            t_a.matmul(t_b).unwrap()
        });
        Box::into_raw(Box::new(OpaqueTensor(result, None, None)))
    }
}
