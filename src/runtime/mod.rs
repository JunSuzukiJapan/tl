pub mod arena;
pub mod device;
pub mod memory_manager; // Arena allocator for tensor memory optimization

pub mod checkpoint;
pub mod registry;
pub mod stdlib;
pub mod tensor_pool;

use crate::runtime::device::get_device;
use candle_core::Tensor;
// Import candle_nn
use std::cell::RefCell;
use std::ffi::{c_float, c_void};
use std::slice;
use std::sync::{Arc, Mutex};

// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.

// Thread-local storage for the latest gradients computed by backward()
thread_local! {
    static LATEST_GRADS: RefCell<Option<candle_core::backprop::GradStore>> = const { RefCell::new(None) };
}

// Global VarMap for tracking all trainable parameters
lazy_static::lazy_static! {
    static ref GLOBAL_VAR_MAP: Mutex<candle_nn::VarMap> = Mutex::new(candle_nn::VarMap::new());
    static ref GLOBAL_PARAM_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
}

/// OpaqueTensor wraps a Candle Tensor and optionally a Var for gradient tracking
/// The Var is stored as Arc to allow sharing across clones while keeping the same variable alive
/// The name is used to register the Var in the global// Opaque wrapper for Candle Tensor
#[repr(C)]
pub struct OpaqueTensor(
    pub Tensor,
    pub Option<Arc<candle_core::Var>>,
    pub Option<String>,
);

// Drop implementation removed as memory is managed by memory_manager::free explicitly.
// OpaqueTensor pointers are raw pointers in C-ABI and should be freed via tl_mem_exit_scope.

// Helper to create and register an OpaqueTensor from a Candle Tensor
// NOTE: Do NOT register here - caller is responsible for lifetime management.
// Registering here causes tensors to be freed when the function scope exits,
// even if the tensor is returned and stored in a struct.
// even if the tensor is returned and stored in a struct.
pub(crate) fn make_tensor(t: Tensor) -> *mut OpaqueTensor {
    let boxed = Box::new(OpaqueTensor(t, None, None));
    let ptr = Box::into_raw(boxed);
    memory_manager::register_tensor_global(ptr);
    println!("DEBUG: Alloc Tensor {:p} (make_tensor)", ptr);
    ptr
}

// Helper to create and register an OpaqueTensor from a Candle Var
// Helper to create and register an OpaqueTensor from a Candle Var
pub(crate) fn make_var(v: candle_core::Var) -> *mut OpaqueTensor {
    let t_ref = v.as_tensor().clone();
    let var_arc = Arc::new(v);

    let boxed = Box::new(OpaqueTensor(t_ref, Some(var_arc), None));
    let ptr = Box::into_raw(boxed);

    // NOTE: Caller is responsible for lifetime management.
    memory_manager::register_tensor_global(ptr);
    println!("DEBUG: Alloc Tensor {:p} (make_var)", ptr);
    ptr
}

// --- File I/O ---

#[no_mangle] // Existing functions continue...
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

    let res = make_tensor(tensor);
    // println!("DEBUG: Alloc Tensor {:p}", res); // Handled in make_tensor
    res
}

// tl_tensor_argmax(t: *mut, dim: i64, keep_dim: bool) -> *mut
#[no_mangle]
pub extern "C" fn tl_tensor_argmax(
    t: *mut OpaqueTensor,
    dim: i64,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    unsafe {
        let ten = &(*t).0;
        match ten.argmax_keepdim(dim as usize) {
            Ok(res) => {
                let final_res = if keep_dim {
                    res
                } else {
                    res.squeeze(dim as usize).unwrap_or(res)
                };
                // Ensure result is F32 to match system expectation
                let final_f32 = final_res
                    .to_dtype(candle_core::DType::F32)
                    .unwrap_or(final_res);
                make_tensor(final_f32)
            }
            Err(e) => {
                eprintln!("tl_tensor_argmax error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

// tl_tensor_item_i64(t: *mut) -> i64
// Extract single scalar value from a 0-D or 1-element tensor
#[no_mangle]
pub extern "C" fn tl_tensor_item_i64(t: *mut OpaqueTensor) -> i64 {
    unsafe {
        let ten = &(*t).0;
        // Try to get as scalar i64 directly
        match ten.to_scalar::<i64>() {
            Ok(v) => v,
            Err(_) => {
                // Try u32 (argmax returns U32)
                match ten.to_scalar::<u32>() {
                    Ok(v) => v as i64,
                    Err(_) => {
                        // Try f32 and cast
                        match ten.to_scalar::<f32>() {
                            Ok(v) => v as i64,
                            Err(e) => {
                                // Convert to 1D vec and take first?
                                let dims = ten.dims();
                                let elem_count: usize = dims.iter().product();
                                if elem_count == 1 {
                                    // Try u32 vec first (for argmax results)
                                    if let Ok(v) = ten.flatten_all().unwrap().to_vec1::<u32>() {
                                        return v[0] as i64;
                                    }
                                    // Fallback to f32
                                    let v = ten.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                    v[0] as i64
                                } else {
                                    eprintln!("tl_tensor_item_i64 error: Tensor has {} elements, expected 1. Error: {}", elem_count, e);
                                    0
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
// Function to create a tensor that requires gradients (similar to Var, but returning Tensor)
// Actually Var is a wrapper around Tensor. For backprop to work, we usually use Var or
// create a tensor and then watch it? Candle's graph is dynamic.
// `Tensor::new` creates a leaf.
// To support `requires_grad`, we effectively need the tensor to be a variable in the graph.
// In Candle, `Var` is for trainable weights.
// Let's explicitly add a helper for creating vars.

#[no_mangle]
pub extern "C" fn tl_tensor_randn_debug(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_data: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, rank).to_vec() };

    let device = get_device();

    // Create random tensor: mean=0.0, std=1.0
    // Correct signature: randn(mean, std, shape, device)
    let t = Tensor::randn(0.0f32, 1.0f32, &shape_data[..], &device).unwrap();

    let ptr = if req_grad {
        let var = candle_core::Var::from_tensor(&t).unwrap();
        make_var(var)
    } else {
        make_tensor(t)
    };

    ptr
}

#[no_mangle]
pub extern "C" fn tl_tensor_zeros(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_data: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, rank).to_vec() };

    let device = get_device();

    // Create zero tensor
    let t = Tensor::zeros(&shape_data[..], candle_core::DType::F32, &device).unwrap();

    let ptr = if req_grad {
        let var = candle_core::Var::from_tensor(&t).unwrap();
        make_var(var)
    } else {
        make_tensor(t)
    };

    ptr
}

/// Create a 1D Tensor from an i64 array (for reshape shape arguments)

#[no_mangle]
pub extern "C" fn tl_tensor_from_i64_array(data: *const i64, len: usize) -> *mut OpaqueTensor {
    let data_slice = unsafe { slice::from_raw_parts(data, len) };
    let device = get_device();

    // Convert i64 to f32 for Tensor (Candle tensors are typically f32)
    let data_f32: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();

    let tensor = Tensor::from_slice(&data_f32, &[len], &device).unwrap();
    make_tensor(tensor)
}

// VarBuilder-based parameter management (following Candle's official pattern)
// This allows proper gradient tracking for parameters stored in struct fields

/// Create or retrieve a parameter from the global VarMap
/// This is the key to proper gradient tracking - all trainable parameters
/// must be created through this function to ensure they are registered in VarMap
#[no_mangle]
pub extern "C" fn tl_varbuilder_get_from_tensor(
    name_ptr: *const std::os::raw::c_char,
    shape_tensor: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    use std::ffi::CStr;

    let name_str = unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() };

    // Extract shape from tensor
    let shape_data = unsafe {
        let t = &(*shape_tensor).0;
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>()
    };
    let shape_slice = &shape_data[..];

    tl_varbuilder_get_common(name_str, shape_slice)
}

fn tl_varbuilder_get_common(name_str: String, shape_slice: &[usize]) -> *mut OpaqueTensor {
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
    drop(data_guard);
    drop(varmap);

    let tensor = var.as_tensor().clone();
    let var_arc = Arc::new(var);
    let boxed = Box::new(OpaqueTensor(tensor, Some(var_arc), Some(name_str)));

    let ptr = if arena::tl_arena_is_active() {
        let size = std::mem::size_of::<OpaqueTensor>() as i64;
        let arena_ptr = arena::tl_arena_malloc(size) as *mut OpaqueTensor;
        if !arena_ptr.is_null() {
            unsafe {
                std::ptr::write(arena_ptr, *boxed);
            }
            arena_ptr
        } else {
            Box::into_raw(boxed)
        }
    } else {
        Box::into_raw(boxed)
    };

    memory_manager::register_tensor_global(ptr);
    ptr
}

#[no_mangle]
pub extern "C" fn tl_varbuilder_get(
    name: *const std::os::raw::c_char,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    use std::ffi::CStr;

    let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap().to_string() };
    let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };

    tl_varbuilder_get_common(name_str, shape_slice)
}

/// Update all parameters in VarMap using SGD
#[no_mangle]
pub extern "C" fn tl_update_all_params(learning_rate: f32) {
    let varmap = GLOBAL_VAR_MAP.lock().unwrap();
    let data = varmap.data();
    let data_guard = data.lock().unwrap();

    LATEST_GRADS.with(|g| {
        if let Some(grads) = &*g.borrow() {
            for (_name, var) in data_guard.iter() {
                if let Some(grad) = grads.get(var.as_tensor()) {
                    // SGD update: param = param - lr * grad
                    let updated = (var.as_tensor() - (grad * learning_rate as f64).unwrap())
                        .unwrap()
                        .detach();

                    // Update the Var with new value
                    var.set(&updated).unwrap();

                    // println!("Updated param '{}': shape {:?}", name, updated.shape());
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
            return make_tensor(g);
        }
    }

    std::ptr::null_mut()
}

/// Get current process memory usage in MB
/// Returns RSS (Resident Set Size) memory in megabytes
#[no_mangle]
pub extern "C" fn tl_get_memory_mb() -> i64 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let pid = std::process::id();
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output();

        if let Ok(output) = output {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(kb) = s.trim().parse::<i64>() {
                    return kb / 1024; // Convert KB to MB
                }
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Fallback: return -1 on unsupported platforms
    }

    -1
}

#[no_mangle]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = t_a
            .broadcast_add(t_b)
            .unwrap_or_else(|_| t_a.add(t_b).unwrap());
        make_tensor(result)
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
        make_tensor(result)
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
        make_tensor(result)
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
        make_tensor(result)
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

        make_tensor(loss)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_detach(t: *mut OpaqueTensor, req_grad: bool) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let detached = tensor.detach();
        if req_grad {
            let var = candle_core::Var::from_tensor(&detached).unwrap();
            make_var(var)
        } else {
            make_tensor(detached)
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
pub extern "C" fn tl_tensor_print(t: *const OpaqueTensor) {
    if t.is_null() {
        println!("Tensor(NULL)");
        return;
    }
    unsafe {
        let tensor = &(*t).0;
        println!("{}", tensor);
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_print_2(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

#[no_mangle]
pub extern "C" fn tl_tensor_print_1(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

#[no_mangle]
pub extern "C" fn tl_tensor_print_3(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

/// Internal function to free tensor resources without unregistering
/// Used by MemoryManager to avoid deadlock
pub(crate) fn free_tensor_resources(t: *mut OpaqueTensor) {
    if !t.is_null() {
        unsafe {
            if arena::tl_arena_contains(t as *mut std::ffi::c_void) {
                // Arena-allocated tensors MUST be dropped to release Candle resources (GPU memory, etc)
                // The memory for OpaqueTensor itself is reclaimed by arena reset, but the inner content needs Drop.
                std::ptr::drop_in_place(t);
            } else {
                // Heap allocated, safe to free via Box (calls drop implicitly)
                let _ = Box::from_raw(t);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        memory_manager::tl_tensor_release(t);
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_clone(t: *const OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let var_ref = &(*t).1;
        let cloned = tensor.clone();

        // Clone the Arc<Var> if it exists to maintain gradient tracking
        let cloned_var = var_ref.as_ref().map(Arc::clone);
        let boxed = Box::new(OpaqueTensor(cloned, cloned_var, None));

        // NOTE: Do NOT use arena allocator here.
        // Arena allocations are reset on scope exit, causing memory reuse.
        // Cloned tensors must persist beyond the clone function's scope.
        let ptr = Box::into_raw(boxed);

        // NOTE: Do NOT register cloned tensor here.
        // The cloned tensor is returned to the caller (e.g., Embedding constructor).
        // The caller should manage its lifetime and register it if needed.
        // Registering here causes immediate freeing when the clone function's scope exits.
        // memory_manager::register_tensor_global(ptr);

        ptr
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.neg().unwrap();
        make_tensor(result)
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
pub extern "C" fn tl_vec_u8_read_i32_be(ptr: *mut Vec<u8>, idx: i64) -> i64 {
    unsafe {
        let vec = &*ptr;
        let idx = idx as usize;
        if idx + 4 > vec.len() {
            return 0;
        }
        let sub = &vec[idx..idx + 4];
        let val = u32::from_be_bytes(sub.try_into().unwrap());
        val as i64
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_from_vec_u8(
    ptr: *mut Vec<u8>,
    offset: i64,
    shape_ptr: *const i64,
    rank: usize,
) -> *mut OpaqueTensor {
    unsafe {
        let vec = &*ptr;
        let offset = offset as usize;
        let shape_slice = std::slice::from_raw_parts(shape_ptr, rank);
        let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

        let total_elements: usize = shape.iter().product();

        if offset + total_elements > vec.len() {
            eprintln!(
                "tl_tensor_from_vec_u8: Not enough elements. Needed {}, have {} (offset {})",
                total_elements,
                vec.len() - offset,
                offset
            );
            panic!("tl_tensor_from_vec_u8: Out of bounds");
        }

        let sub_vec = &vec[offset..offset + total_elements];
        let data_f32: Vec<f32> = sub_vec.iter().map(|&b| b as f32 / 255.0).collect();

        let tensor =
            candle_core::Tensor::from_vec(data_f32, shape, &candle_core::Device::Cpu).unwrap();

        let tensor_with_grad = OpaqueTensor(tensor, None, None);

        Box::into_raw(Box::new(tensor_with_grad))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_from_u8_labels(
    ptr: *mut Vec<u8>,
    offset: i64,
    count: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let vec = &*ptr;
        let offset = offset as usize;
        let count = count as usize;

        if offset + count > vec.len() {
            panic!("tl_tensor_from_u8_labels: Out of bounds");
        }

        let sub_vec = &vec[offset..offset + count];
        let data_i64: Vec<i64> = sub_vec.iter().map(|&b| b as i64).collect();
        let shape = vec![count];

        let tensor =
            candle_core::Tensor::from_vec(data_i64, shape, &candle_core::Device::Cpu).unwrap();

        let tensor_with_grad = OpaqueTensor(tensor, None, None);

        Box::into_raw(Box::new(tensor_with_grad))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_get(t: *mut OpaqueTensor, idx: i64) -> c_float {
    unsafe {
        if t.is_null() {
            println!("FATAL: tl_tensor_get received NULL pointer");
            return 0.0;
        }
        let tensor = &(*t).0;
        let i = idx as usize;

        // Generic scalar extraction
        let scalar_val = tensor.flatten_all().unwrap().get(i).unwrap();
        match scalar_val.dtype() {
            candle_core::DType::F32 => scalar_val.to_scalar::<f32>().unwrap(),
            candle_core::DType::I64 => scalar_val.to_scalar::<i64>().unwrap() as f32,
            candle_core::DType::U8 => scalar_val.to_scalar::<u8>().unwrap() as f32,
            dt => panic!("tl_tensor_get: Unsupported dtype {:?}", dt),
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_item(t: *mut OpaqueTensor) -> c_float {
    tl_tensor_get(t, 0)
}

#[no_mangle]
pub extern "C" fn tl_tensor_set_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
    val: c_float,
) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let idxs = slice::from_raw_parts(indices, rank);
        let idxs_usize: Vec<usize> = idxs.iter().map(|&x| x as usize).collect();

        // Calculate flat index
        let dims = tensor.dims();
        let mut flat_idx = 0;
        let mut stride = 1;
        // Strides are usually computed from right to left assuming contiguous C order
        for i in (0..rank).rev() {
            flat_idx += idxs_usize[i] * stride;
            stride *= dims[i];
        }

        // Slow path: converting to Vec, updating, and recreating Tensor
        // Efficient enough for small tensors in examples.
        let mut data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        if flat_idx < data.len() {
            data[flat_idx] = val;
        } else {
            eprintln!(
                "Index out of bounds in tensor_set: {} >= {}",
                flat_idx,
                data.len()
            );
        }

        let new_tensor = Tensor::from_vec(data, tensor.shape(), tensor.device()).unwrap();
        make_tensor(new_tensor)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_slice(t: *mut OpaqueTensor, start: i64, len: i64) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Slice along first dimension
        let result = tensor.narrow(0, start as usize, len as usize).unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_print_i64(v: i64) {
    println!("{}", v);
}

#[no_mangle]
pub extern "C" fn tl_print_ptr(_ptr: *const std::ffi::c_void) {
    // Debug function - no output in production
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
    let _ = tl_clear_grads as *const ();
    let _ = tl_tensor_randn_debug as *const ();
    let _ = tl_tensor_set_f32_md as *const ();
    let _ = tl_tensor_free as *const ();
    let _ = memory_manager::tl_mem_unregister as *const ();
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
        // Assuming row-major layout (C-style)
        // flat_idx = i0 * stride0 + i1 * stride1 + ...
        // stride_last = 1
        // stride_k = stride_{k+1} * dim_{k+1}
        let mut stride = 1;
        for i in (0..rank).rev() {
            flat_idx += idxs_usize[i] * stride;
            stride *= dims[i];
        }

        tensor
            .flatten_all()
            .unwrap()
            .get(flat_idx)
            .unwrap()
            .to_scalar()
            .unwrap()
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
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_backward(t: *mut OpaqueTensor) {
    unsafe {
        if t.is_null() {
            eprintln!("Runtime Error: tl_tensor_backward received NULL pointer");

            return;
        }

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
pub extern "C" fn tl_clear_grads() {
    LATEST_GRADS.with(|g| {
        *g.borrow_mut() = None;
    });
}
#[no_mangle]
pub extern "C" fn tl_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        if t.is_null() {
            return std::ptr::null_mut();
        }
        let tensor = &(*t).0;

        // If this tensor has an associated Var, get the gradient from it directly
        // (Known limitation: Candle Var doesn't expose grad directly easily here)

        // Retrieve gradient from LATEST_GRADS
        let grad = LATEST_GRADS.with(|g| {
            let borrow = g.borrow();
            if let Some(grads) = borrow.as_ref() {
                grads.get(tensor).cloned()
            } else {
                None
            }
        });

        if let Some(g) = grad {
            return make_tensor(g);
        }

        // FALLBACK: If no gradient found, return a zero tensor of the same shape
        // This prevents NULL pointer crashes in training loops
        if let Ok(zeros) = tensor.zeros_like() {
            return make_tensor(zeros);
        }

        std::ptr::null_mut()
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sum_all().unwrap();
        make_tensor(result)
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
        // IMPORTANT: We must re-create a Var from the detached result to ensure
        // that gradients are tracked for the NEXT iteration.
        // If we just use result.detach(), it returns a tensor with no history and usually no requires_grad.
        let detached = result.detach();
        let var = candle_core::Var::from_tensor(&detached).unwrap();
        let t_var = var.as_tensor().clone();
        let var_arc = Arc::new(var);

        (*ref_t).0 = t_var;
        (*ref_t).1 = Some(var_arc);
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
        make_tensor(result)
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
            Ok(result) => make_tensor(result),
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
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.exp().unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_pow(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        if a.is_null() || b.is_null() {
            println!(
                "FATAL: tl_tensor_pow received NULL pointer: a={:p}, b={:p}",
                a, b
            );
            return std::ptr::null_mut();
        }
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        // Broadcasting pow
        let result = t_a
            .broadcast_pow(t_b)
            .unwrap_or_else(|_| t_a.pow(t_b).unwrap());
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.log().unwrap();
        make_tensor(result)
    }
}

// --- Transformer Support ---

#[no_mangle]
pub extern "C" fn tl_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.sin().unwrap();
    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.cos().unwrap();
    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.relu().unwrap();
    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.gelu().unwrap();
    make_tensor(res)
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
        return make_tensor(t.clone());
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
    let h_dim = h;
    let w_dim = w;
    let target_shape = [h_dim, w_dim];

    let c_b = c.broadcast_as(&target_shape[..]).unwrap();
    let b_b = boundary.broadcast_as(&target_shape[..]).unwrap();
    let mask = c_b.le(&b_b).unwrap();

    let mask = mask.to_dtype(t.dtype()).unwrap();
    let res = t.broadcast_mul(&mask).unwrap();
    make_tensor(res)
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
    make_tensor(res)
}

// Embedding: indices [B, S], weights [V, D] -> [B, S, D]
#[no_mangle]
pub extern "C" fn tl_tensor_embedding(
    indices: *mut OpaqueTensor,
    weights: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    let indices_tensor = unsafe { &(*indices).0 };
    let weights_tensor = unsafe { &(*weights).0 };
    let w_dims = weights_tensor.dims();
    let _i_dims = indices_tensor.dims();

    if w_dims.len() != 2 {
        // Handle error or specific case for non-2D weights
        // For now, let's just panic as it's an unexpected shape for embedding weights
        panic!("Embedding weights must be 2-dimensional, got {:?}", w_dims);
    }
    // Input indices might be F32 if coming from literals.
    let indices_u32 = indices_tensor.to_dtype(candle_core::DType::U32).unwrap();

    // 1. Flatten indices to 1D
    let flat_indices = indices_u32.flatten_all().unwrap();
    // 2. Select embeddings
    // index_select operates on dimension 0
    let out = weights_tensor.index_select(&flat_indices, 0).unwrap();

    // 3. Reshape output
    let mut out_shape = indices_tensor.dims().to_vec();
    out_shape.push(weights_tensor.dim(candle_core::D::Minus1).unwrap());

    let res = out.reshape(out_shape).unwrap();

    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sqrt().unwrap();
        make_tensor(result)
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
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Candleのcontiguous()を使用してメモリレイアウトを連続化
        let result = tensor.contiguous().unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_save(t: *mut OpaqueTensor, path: *const std::os::raw::c_char) {
    use std::ffi::CStr;
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        let tensor = &(*t).0;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("tensor".to_string(), tensor.clone());
        candle_core::safetensors::save(&tensors, path_str).unwrap();
        // println!("Saved tensor to {}", path_str);
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_load(path: *const std::os::raw::c_char) -> *mut OpaqueTensor {
    use std::ffi::CStr;
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        let device = get_device();
        let tensors = candle_core::safetensors::load(path_str, &device).unwrap();
        let tensor = tensors
            .get("tensor")
            .expect("Failed to find 'tensor' key in file")
            .clone();
        make_tensor(tensor)
    }
}

// --- Tensor Map (State Dict) Support ---

#[repr(C)]
pub struct OpaqueTensorMap(pub std::collections::HashMap<String, Tensor>);

#[no_mangle]
pub extern "C" fn tl_tensor_map_new() -> *mut OpaqueTensorMap {
    let map = OpaqueTensorMap(std::collections::HashMap::new());
    Box::into_raw(Box::new(map))
}

#[no_mangle]
pub extern "C" fn tl_tensor_map_insert(
    map: *mut OpaqueTensorMap,
    name: *const std::os::raw::c_char,
    tensor: *mut OpaqueTensor,
) {
    unsafe {
        if map.is_null() || name.is_null() || tensor.is_null() {
            return;
        }

        let map_ref = &mut (*map).0;
        let c_str = std::ffi::CStr::from_ptr(name);
        let key = c_str.to_string_lossy().into_owned();

        let t_ref = &(*tensor).0;
        map_ref.insert(key, t_ref.clone());
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_map_save(map: *mut OpaqueTensorMap, path: *const std::os::raw::c_char) {
    unsafe {
        let map_ref = &(*map).0;
        let p_str = std::ffi::CStr::from_ptr(path).to_str().unwrap();
        candle_core::safetensors::save(map_ref, p_str).unwrap();
        // println!("Saved model to {}", p_str);
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_map_load(path: *const std::os::raw::c_char) -> *mut OpaqueTensorMap {
    unsafe {
        let p_str = std::ffi::CStr::from_ptr(path).to_str().unwrap();
        let device = get_device();
        let map =
            candle_core::safetensors::load(p_str, &device).expect("Failed to load model file");
        let opaque = OpaqueTensorMap(map);
        Box::into_raw(Box::new(opaque))
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_map_get(
    map: *mut OpaqueTensorMap,
    name: *const std::os::raw::c_char,
) -> *mut OpaqueTensor {
    unsafe {
        let map_ref = &(*map).0;
        let c_str = std::ffi::CStr::from_ptr(name);
        let key = c_str.to_string_lossy();

        if let Some(t) = map_ref.get(key.as_ref()) {
            make_tensor(t.clone())
        } else {
            // Panic or Return Null? Panic for now standard behavior
            panic!("Weight '{}' not found in loaded file.", key);
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_map_free(map: *mut OpaqueTensorMap) {
    unsafe {
        let _ = Box::from_raw(map); // Drop
    }
}
#[no_mangle]
pub extern "C" fn tl_save_all_params(path: *const std::os::raw::c_char) {
    use std::ffi::CStr;
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        let varmap = GLOBAL_VAR_MAP.lock().unwrap();
        let data = varmap.data();
        let data_guard = data.lock().unwrap();

        // Convert VarMap to HashMap<String, Tensor> for saving
        let mut tensors = std::collections::HashMap::new();
        for (key, var) in data_guard.iter() {
            tensors.insert(key.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path_str).unwrap();
        println!("Saved {} parameters to {}", tensors.len(), path_str);
    }
}

#[no_mangle]
pub extern "C" fn tl_load_all_params(path: *const std::os::raw::c_char) {
    use std::ffi::CStr;
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        let device = get_device();

        match candle_core::safetensors::load(path_str, &device) {
            Ok(tensors) => {
                let varmap = GLOBAL_VAR_MAP.lock().unwrap();
                let data = varmap.data();
                let data_guard = data.lock().unwrap();

                let mut loaded_count = 0;
                for (key, tensor) in tensors.iter() {
                    if let Some(var) = data_guard.get(key) {
                        if let Err(e) = var.set(tensor) {
                            eprintln!("Warning: Failed to set param {}: {}", key, e);
                        } else {
                            loaded_count += 1;
                        }
                    } else {
                        // Parameter not found in current model (maybe model structure changed or unused weight)
                        // eprintln!("Warning: Param {} found in file but not in current model", key);
                    }
                }
                println!("Loaded {} parameters from {}", loaded_count, path_str);
            }
            Err(e) => {
                eprintln!("Error loading parameters from {}: {}", path_str, e);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_add_parameter(name: *const std::os::raw::c_char, t: *mut OpaqueTensor) {
    use std::ffi::CStr;
    unsafe {
        println!(
            "tl_add_parameter called for {} with ptr {:p}",
            CStr::from_ptr(name).to_string_lossy(),
            t
        );
        if t.is_null() {
            println!("ERROR: tl_add_parameter got NULL ptr");
            return;
        }
        let name_str = CStr::from_ptr(name).to_str().unwrap().to_string();
        let tensor_wrapper = &*t;

        // Check if OpaqueTensor already has a Var associated
        let var = if let Some(ref v) = tensor_wrapper.1 {
            v.as_ref().clone()
        } else {
            // If not, create a new Var from the inner tensor
            candle_core::Var::from_tensor(&tensor_wrapper.0).unwrap()
        };

        let varmap = GLOBAL_VAR_MAP.lock().unwrap();
        let data = varmap.data();
        let mut data_guard = data.lock().unwrap();

        data_guard.insert(name_str.clone(), var);
    }
}

#[no_mangle]
pub extern "C" fn tl_register_parameter(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use std::sync::atomic::Ordering;
    unsafe {
        let tensor_wrapper = &mut *t;
        // Ensure we have a Var
        let var_arc = if let Some(ref v) = tensor_wrapper.1 {
            v.clone()
        } else {
            // Create Var if not present
            let v = candle_core::Var::from_tensor(&tensor_wrapper.0).unwrap();
            let arc = Arc::new(v);
            tensor_wrapper.1 = Some(arc.clone());
            tensor_wrapper.0 = arc.as_tensor().clone();
            arc
        };

        // Generate Name
        let id = GLOBAL_PARAM_COUNTER.fetch_add(1, Ordering::SeqCst);
        let name = format!("param_{}", id);

        // Register
        let varmap = GLOBAL_VAR_MAP.lock().unwrap();
        let data = varmap.data();
        let mut data_guard = data.lock().unwrap();
        data_guard.insert(name, var_arc.as_ref().clone());

        // CRITICAL: Acquire reference to keep OpaqueTensor alive (owned by Global Registry concepts)
        memory_manager::tl_tensor_acquire(t);

        t // Return same pointer
    }
}
/// Allocate a temporary buffer, using the Arena if it's active and the size is "large"
#[no_mangle]
pub extern "C" fn tl_alloc_tmp(size: i64) -> *mut c_void {
    if size <= 0 {
        return std::ptr::null_mut();
    }

    // Threshold for using arena for temporary buffers: 256 bytes (e.g. 64 f32 elements)
    if size >= 256 && arena::tl_arena_is_active() {
        let ptr = arena::tl_arena_malloc(size);
        if !ptr.is_null() {
            return ptr;
        }
    }

    // Fallback to heap
    unsafe { libc::malloc(size as usize) }
}

/// Free a temporary buffer allocated with tl_alloc_tmp
#[no_mangle]
pub extern "C" fn tl_free_tmp(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    // Check if it belongs to the arena
    if arena::tl_arena_contains(ptr) {
        // Arena handles its own deallocation (nothing to do here)
        return;
    }

    // Heap allocated, must free
    unsafe {
        libc::free(ptr);
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.to_dtype(candle_core::DType::F32).unwrap();
    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_tensor_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.to_dtype(candle_core::DType::I64).unwrap();
    make_tensor(res)
}

#[no_mangle]
pub extern "C" fn tl_vec_void_len(ptr: *mut std::ffi::c_void) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let vec = &*(ptr as *mut Vec<*mut std::ffi::c_void>);
        vec.len()
    }
}

#[no_mangle]
pub extern "C" fn tl_vec_void_get(ptr: *mut std::ffi::c_void, idx: usize) -> *mut std::ffi::c_void {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let vec = &*(ptr as *mut Vec<*mut std::ffi::c_void>);
        if idx < vec.len() {
            vec[idx]
        } else {
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_vec_void_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        unsafe {
            // Reconstruct Vec from raw pointer to drop the container
            // This frees the Vec struct logic (cap/len/buffer pointer)
            let _ = Box::from_raw(ptr as *mut Vec<*mut std::ffi::c_void>);
        }
    }
}

// --- Vec<u8> support for binary data ---

#[no_mangle]
pub extern "C" fn tl_vec_u8_new() -> *mut Vec<u8> {
    Box::into_raw(Box::new(Vec::<u8>::new()))
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_with_capacity(cap: usize) -> *mut Vec<u8> {
    Box::into_raw(Box::new(Vec::<u8>::with_capacity(cap)))
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_len(ptr: *mut Vec<u8>) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).len() }
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_get(ptr: *mut Vec<u8>, idx: usize) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let vec = &*ptr;
        if idx < vec.len() {
            vec[idx]
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_set(ptr: *mut Vec<u8>, idx: usize, val: u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let vec = &mut *ptr;
        if idx < vec.len() {
            vec[idx] = val;
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_push(ptr: *mut Vec<u8>, val: u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        (*ptr).push(val);
    }
}

#[no_mangle]
pub extern "C" fn tl_vec_u8_free(ptr: *mut Vec<u8>) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// String helper
#[no_mangle]
pub extern "C" fn tl_string_new(s: *const std::os::raw::c_char) -> *mut std::os::raw::c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let len = libc::strlen(s);
        let dest = libc::malloc(len + 1) as *mut std::os::raw::c_char;
        if !dest.is_null() {
            libc::strcpy(dest, s);
        }
        dest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::memory_manager;

    #[test]
    fn test_tensor_creation() {
        // Need to enter scope to avoid warning/crash logic
        memory_manager::tl_mem_enter_scope();

        let shape: Vec<usize> = vec![2, 2];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let ptr = tl_tensor_new(data.as_ptr(), 2, shape.as_ptr());

        assert!(!ptr.is_null());

        memory_manager::tl_mem_exit_scope();
    }
}
