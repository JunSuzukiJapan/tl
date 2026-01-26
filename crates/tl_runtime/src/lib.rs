pub mod arena;
pub mod args;
pub mod device;
pub mod error;
pub mod memory_manager; // Arena allocator for tensor memory optimization
pub use memory_manager::{
    tl_get_pool_count, tl_get_refcount_count, tl_get_scope_depth, tl_mem_function_enter,
    tl_mem_function_exit, tl_mem_get_buffer,
};

pub mod checkpoint;
pub mod context;
pub mod cuda_ext;
pub mod knowledge_base;
pub mod llm;
pub mod logic;
pub mod registry;
pub mod stdlib;
pub mod tensor_pool;

use crate::device::get_device;
use candle_core::Tensor;
// Import candle_nn
use std::cell::RefCell;
use std::ffi::{c_float, c_void};
use std::io::Write;
use std::slice;
use std::sync::{Arc, Mutex, OnceLock};

// Opaque struct to represent a Tensor in C-ABI (LLVM IR)
// In reality, this will be a raw pointer to a Heap-allocated Tensor.
// For FFI safety, we use a wrapper.

// Thread-local storage for the latest gradients computed by backward()
thread_local! {
    static LATEST_GRADS: RefCell<Option<candle_core::backprop::GradStore>> = const { RefCell::new(None) };
}

pub(crate) fn mem_log_enabled() -> bool {
    // Check every time to allow runtime enablement via set_var
    std::env::var("TL_MEM_LOG")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false)
}

fn mem_trace_enabled() -> bool {
    static MEM_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
    *MEM_TRACE_ENABLED.get_or_init(|| {
        std::env::var("TL_MEM_TRACE")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

#[derive(Copy, Clone)]
pub(crate) enum FreeOutcome {
    ArenaDrop,
    Pooled,
    Freed,
}

fn dtype_size_bytes(dtype: candle_core::DType) -> usize {
    match dtype {
        candle_core::DType::F32 => 4,
        candle_core::DType::F64 => 8,
        candle_core::DType::I64 => 8,
        candle_core::DType::U32 => 4,
        candle_core::DType::U8 => 1,
        candle_core::DType::F16 => 2,
        candle_core::DType::BF16 => 2,
    }
}

#[track_caller]
fn record_tensor_alloc(ctx: &str, ptr: *mut OpaqueTensor, t: &Tensor, pooled: bool) {
    if !mem_log_enabled() {
        return;
    }
    let elem_count = t.elem_count();
    let dtype = t.dtype();
    let bytes = elem_count.saturating_mul(dtype_size_bytes(dtype));
    let shape = t.dims().iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
    let device = match t.device() {
        candle_core::Device::Cpu => "cpu",
        candle_core::Device::Cuda(_) => "cuda",
        candle_core::Device::Metal(_) => "metal",
    };
    let loc = std::panic::Location::caller();
    let meta = memory_manager::AllocationMeta {
        ctx: ctx.to_string(),
        bytes,
        dtype: format!("{:?}", dtype),
        elems: elem_count,
        shape: shape.clone(),
        device: device.to_string(),
        loc_file: loc.file().to_string(),
        loc_line: loc.line(),
        pooled,
    };
    memory_manager::register_tensor_meta_global(ptr, meta);
    if !pooled {
        eprintln!(
            "[TL_MEM] alloc_non_pool ctx={} bytes={} dtype={:?} elems={} shape=[{}] device={} loc={}:{}",
            ctx,
            bytes,
            dtype,
            elem_count,
            shape,
            device,
            loc.file(),
            loc.line()
        );
    }
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

// Helper to report runtime errors from JIT code
// Replaced by tl_handle_runtime_error, kept for compatibility if needed.
#[unsafe(no_mangle)]
pub extern "C" fn tl_report_runtime_error(_msg: *const std::os::raw::c_char) {
    // Deprecated
}

fn handle_runtime_error_internal(
    code: u32,
    msg: String,
    file: Option<String>,
    line: u32,
    col: u32,
) {
    let error_code = unsafe { std::mem::transmute(code) };
    crate::error::LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(crate::error::LastError {
            code: error_code,
            message: msg,
            file: file.unwrap_or("unknown".to_string()),
            line,
            col,
        });
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_handle_runtime_error(
    code: u32,
    msg: *const std::os::raw::c_char,
    file: *const std::os::raw::c_char,
    line: u32,
    col: u32,
) {
    let message = if !msg.is_null() {
        unsafe { std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned() }
    } else {
        "Unknown Error".to_string()
    };

    let filename = if !file.is_null() {
        unsafe {
            std::ffi::CStr::from_ptr(file)
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown".to_string()
    };

    handle_runtime_error_internal(code, message, Some(filename), line, col);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_amend_error_loc(file: *const std::os::raw::c_char, line: u32, col: u32) {
    let filename = if !file.is_null() {
        unsafe {
            std::ffi::CStr::from_ptr(file)
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown".to_string()
    };

    crate::error::LAST_ERROR.with(|e| {
        if let Some(err) = e.borrow_mut().as_mut() {
            err.file = filename;
            err.line = line;
            err.col = col;
        }
    });
}

fn return_ptr_or_null(
    res: std::thread::Result<Result<*mut OpaqueTensor, crate::error::RuntimeError>>,
) -> *mut OpaqueTensor {
    match res {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            handle_runtime_error_internal(e.code() as u32, e.to_string(), None, 0, 0);
            std::ptr::null_mut()
        }
        Err(_) => {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::InternalError as u32,
                "Panic caught in runtime".to_string(),
                None,
                0,
                0,
            );
            std::ptr::null_mut()
        }
    }
}


// Helper to create and register an OpaqueTensor from a Candle Tensor
// NOTE: Do NOT register here - caller is responsible for lifetime management.
// Registering here causes tensors to be freed when the function scope exits,
// even if the tensor is returned and stored in a struct.
// even if the tensor is returned and stored in a struct.
#[track_caller]
pub(crate) fn make_tensor(t: Tensor) -> *mut OpaqueTensor {
    let num_elements = t.elem_count();
    let dtype_id = dtype_to_id(t.dtype());
    let device_id = device_to_id(t.device());

    // Try to acquire from pool first
    let pooled_ptr = if let Ok(mut pool) = memory_manager::TENSOR_POOL.lock() {
        pool.acquire(num_elements, dtype_id, device_id)
    } else {
        None
    };

    if let Some(ptr) = pooled_ptr {
        unsafe {
            // Reuse the OpaqueTensor, replace inner Tensor without dropping old (already dropped) content
            // (*ptr).0 = t; // This would drop the old content!
            std::ptr::write(ptr, OpaqueTensor(t, None, None));
        }
        memory_manager::register_tensor_global(ptr);
        record_tensor_alloc("make_tensor", ptr, unsafe { &(*ptr).0 }, true);
        return ptr;
    }

    // No pooled tensor available, allocate new
    let boxed = Box::new(OpaqueTensor(t, None, None));
    let ptr = Box::into_raw(boxed);
    memory_manager::register_tensor_global(ptr);
    record_tensor_alloc("make_tensor", ptr, unsafe { &(*ptr).0 }, false);
    ptr
}

/// Convert DType to u8 for pool key
fn dtype_to_id(dtype: candle_core::DType) -> u8 {
    match dtype {
        candle_core::DType::F32 => 0,
        candle_core::DType::F64 => 1,
        candle_core::DType::I64 => 2,
        candle_core::DType::U32 => 3,
        candle_core::DType::U8 => 4,
        candle_core::DType::F16 => 5,
        candle_core::DType::BF16 => 6,
    }
}

/// Convert Device to u8 for pool key
fn device_to_id(device: &candle_core::Device) -> u8 {
    match device {
        candle_core::Device::Cpu => 0,
        candle_core::Device::Cuda(_) => 1,
        candle_core::Device::Metal(_) => 2,
    }
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
    record_tensor_alloc("make_var", ptr, unsafe { &(*ptr).0 }, false);
    ptr
}

pub(crate) fn expand_tilde(path: &str) -> String {
    if path.starts_with("~") {
        if let Some(home) = std::env::var("HOME").ok() {
            if path == "~" {
                return home;
            }
            if path.starts_with("~/") {
                return format!("{}{}", home, &path[1..]);
            }
        }
    }
    path.to_string()
}

// --- File I/O ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new(
    data: *const f32,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;

    let res = std::panic::catch_unwind(|| {
        let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };
        let num_elements: usize = shape_slice.iter().product();
        let data_slice = unsafe { slice::from_raw_parts(data, num_elements) };

        let device = get_device();
        // Create on CPU first for stability
        let t_cpu = Tensor::from_slice(data_slice, shape_slice, &candle_core::Device::Cpu)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;

        let tensor = if device.is_metal() || device.is_cuda() {
            t_cpu
                .to_device(&device)
                .map_err(|e| RuntimeError::DeviceError(e.to_string()))?
        } else {
            t_cpu
        };

        Ok(make_tensor(tensor))
    });

    return_ptr_or_null(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new_i64(
    data: *const i64,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;

    let res = std::panic::catch_unwind(|| {
        let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };

        let num_elements: usize = shape_slice.iter().product();
        let data_slice = unsafe { slice::from_raw_parts(data, num_elements) };

        let device = get_device();
        // Create on CPU first for stability
        let t_cpu = Tensor::from_slice(data_slice, shape_slice, &candle_core::Device::Cpu)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;

        let tensor = if device.is_metal() || device.is_cuda() {
            t_cpu
                .to_device(&device)
                .map_err(|e| RuntimeError::DeviceError(e.to_string()))?
        } else {
            t_cpu
        };

        Ok(make_tensor(tensor))
    });

    return_ptr_or_null(res)
}

// tl_tensor_argmax(t: *mut, dim: i64, keep_dim: bool) -> *mut
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmax(
    t: *mut OpaqueTensor,
    dim: i64,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if t.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Tensor is null".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let ten = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let res = ten
                .argmax_keepdim(dim as usize)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            let final_res = if keep_dim {
                res
            } else {
                res.squeeze(dim as usize).unwrap_or(res)
            };
            // Ensure result is I64 for index usage in tl
            let final_i64 = final_res
                .to_dtype(candle_core::DType::I64)
                .unwrap_or(final_res);
            Ok(make_tensor(final_i64))
        }));

        return_ptr_or_null(res)
    }
}

// tl_tensor_item_i64(t: *mut) -> i64
// Extract single scalar value from a 0-D or 1-element tensor
#[unsafe(no_mangle)]
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
                                    // Try i64 vec
                                    if let Ok(v) = ten.flatten_all().unwrap().to_vec1::<i64>() {
                                        return v[0];
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_randn_debug(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;

    if shape.is_null() {
        handle_runtime_error_internal(
            crate::error::RuntimeErrorCode::NullPointerError as u32,
            "Shape is null".to_string(),
            None,
            0,
            0,
        );
        return std::ptr::null_mut();
    }

    let res = std::panic::catch_unwind(|| {
        let shape_data: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, rank).to_vec() };

        let device = get_device();

        // Create random tensor: mean=0.0, std=1.0
        let t = Tensor::randn(0.0f32, 1.0f32, &shape_data[..], &device)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;

        let ptr = if req_grad {
            let var = candle_core::Var::from_tensor(&t).map_err(|e| {
                RuntimeError::AllocationError(format!("Var creation failed: {}", e))
            })?;
            make_var(var)
        } else {
            make_tensor(t)
        };
        Ok(ptr)
    });

    return_ptr_or_null(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_zeros(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;

    // Safety check
    if shape.is_null() {
        handle_runtime_error_internal(
            crate::error::RuntimeErrorCode::NullPointerError as u32,
            "Shape is null".to_string(),
            None,
            0,
            0,
        );
        return std::ptr::null_mut();
    }

    let res = std::panic::catch_unwind(|| {
        let shape_data: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, rank).to_vec() };

        let device = get_device();

        // Create zero tensor
        let t = Tensor::zeros(&shape_data[..], candle_core::DType::F32, &device)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;

        let ptr = if req_grad {
            // Var creation might fail if tensor is not contiguous or something, though zeros usually is
            let var = candle_core::Var::from_tensor(&t).map_err(|e| {
                RuntimeError::AllocationError(format!("Var creation failed: {}", e))
            })?;
            make_var(var)
        } else {
            make_tensor(t)
        };

        Ok(ptr)
    });

    return_ptr_or_null(res)
}

/// Create a 1D Tensor from an i64 array (for reshape shape arguments)

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_i64_array(data: *const i64, len: usize) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;

    let res = std::panic::catch_unwind(|| {
        let data_slice = unsafe { slice::from_raw_parts(data, len) };
        let device = get_device();

        // Use I64 directly
        let data_vec: Vec<i64> = data_slice.to_vec();

        let tensor = Tensor::from_slice(&data_vec, &[len], &device)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;
        Ok(make_tensor(tensor))
    });

    return_ptr_or_null(res)
}

// VarBuilder-based parameter management (following Candle's official pattern)
// This allows proper gradient tracking for parameters stored in struct fields

/// Create or retrieve a parameter from the global VarMap
/// This is the key to proper gradient tracking - all trainable parameters
/// must be created through this function to ensure they are registered in VarMap
#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get_from_tensor(
    name_ptr: *const std::os::raw::c_char,
    shape_tensor: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    use std::ffi::CStr;

    unsafe {
        if name_ptr.is_null() || shape_tensor.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null ptr".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let name_str = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();

            // Extract shape from tensor
            let t = &(*shape_tensor).0;
            let shape_data = t
                .flatten_all()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>();

            let shape_slice = &shape_data[..];

            tl_varbuilder_get_common(name_str, shape_slice)
        }));

        return_ptr_or_null(res)
    }
}

fn tl_varbuilder_get_common(
    name_str: String,
    shape_slice: &[usize],
) -> Result<*mut OpaqueTensor, crate::error::RuntimeError> {
    use crate::error::RuntimeError;
    let varmap = GLOBAL_VAR_MAP
        .lock()
        .map_err(|_| RuntimeError::InternalError("Mutex poisoned".to_string()))?;
    let device = get_device();

    // VarMap.data() returns &Mutex<HashMap<String, Var>>
    let data = varmap.data();
    let mut data_guard = data
        .lock()
        .map_err(|_| RuntimeError::InternalError("Mutex poisoned".to_string()))?;

    // Check if this parameter already exists
    if !data_guard.contains_key(&name_str) {
        // Create new parameter with random initialization
        let tensor = Tensor::randn(0.0f32, 1.0f32, shape_slice, &device)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;
        let var = candle_core::Var::from_tensor(&tensor)
            .map_err(|e| RuntimeError::AllocationError(e.to_string()))?;
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
    record_tensor_alloc("varbuilder_get", ptr, unsafe { &(*ptr).0 }, false);
    Ok(ptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_get(
    name: *const std::os::raw::c_char,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    // use crate::error::RuntimeError;
    use std::ffi::CStr;

    let res = std::panic::catch_unwind(|| {
        let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap().to_string() };
        let shape_slice = unsafe { slice::from_raw_parts(shape, rank) };
        tl_varbuilder_get_common(name_str, shape_slice)
    });
    return_ptr_or_null(res)
}

/// Update all parameters in VarMap using SGD
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub extern "C" fn tl_varbuilder_grad(name: *const std::os::raw::c_char) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    use std::ffi::CStr;

    let res = std::panic::catch_unwind(|| {
        let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap() };

        let varmap = GLOBAL_VAR_MAP
            .lock()
            .map_err(|_| RuntimeError::InternalError("Mutex poisoned".to_string()))?;
        let data = varmap.data();
        let data_guard = data
            .lock()
            .map_err(|_| RuntimeError::InternalError("Mutex poisoned".to_string()))?;

        if let Some(var) = data_guard.get(name_str) {
            let grad = LATEST_GRADS.with(|g| {
                g.borrow()
                    .as_ref()
                    .and_then(|store| store.get(var.as_tensor()).cloned())
            });

            if let Some(g) = grad {
                return Ok(make_tensor(g));
            }
        }
        // Return null if not found
        Ok(std::ptr::null_mut())
    });

    return_ptr_or_null(res)
}

/// Get current process memory usage in MB
/// Returns RSS (Resident Set Size) memory in megabytes
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_mb() -> i64 {
    #[cfg(target_os = "macos")]
    {
        #[allow(non_camel_case_types)]
        use libc::{c_int, c_uint, c_void};

        #[allow(non_camel_case_types)]
        type kern_return_t = c_int;
        #[allow(non_camel_case_types)]
        type mach_port_t = c_uint;
        #[allow(non_camel_case_types)]
        type mach_msg_type_number_t = c_uint;
        #[allow(non_camel_case_types)]
        type natural_t = c_uint;
        #[allow(non_camel_case_types)]
        type integer_t = c_int;
        #[allow(non_camel_case_types)]
        type policy_t = c_int;
        #[allow(non_camel_case_types)]
        type mach_vm_size_t = u64;

        #[repr(C)]
        struct time_value_t {
            seconds: integer_t,
            microseconds: integer_t,
        }

        #[repr(C)]
        struct mach_task_basic_info {
            virtual_size: mach_vm_size_t,
            resident_size: mach_vm_size_t,
            resident_size_max: mach_vm_size_t,
            user_time: time_value_t,
            system_time: time_value_t,
            policy: policy_t,
            suspend_count: integer_t,
        }

        unsafe extern "C" {
            fn mach_task_self() -> mach_port_t;
            fn task_info(
                target_task: mach_port_t,
                flavor: c_int,
                task_info_out: *mut c_void,
                task_info_out_cnt: *mut mach_msg_type_number_t,
            ) -> kern_return_t;
        }

        const KERN_SUCCESS: kern_return_t = 0;
        const MACH_TASK_BASIC_INFO: c_int = 20;

        let mut info = mach_task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            resident_size_max: 0,
            user_time: time_value_t {
                seconds: 0,
                microseconds: 0,
            },
            system_time: time_value_t {
                seconds: 0,
                microseconds: 0,
            },
            policy: 0,
            suspend_count: 0,
        };

        let mut count = (std::mem::size_of::<mach_task_basic_info>() / std::mem::size_of::<natural_t>()) as mach_msg_type_number_t;
        let kr = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut c_void,
                &mut count as *mut mach_msg_type_number_t,
            )
        };
        if kr == KERN_SUCCESS {
            return (info.resident_size / (1024 * 1024)) as i64;
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Fallback: return -1 on unsupported platforms
    }

    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_bytes() -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_mb() -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_metal_pool_count() -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_metal_sync() {
    if let candle_core::Device::Metal(metal) = get_device() {
        let _ = metal.wait_until_completed();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_log_alloc(
    ptr: *const std::ffi::c_void,
    size: i64,
    file: *const std::os::raw::c_char,
    line: i32,
) {
    if !mem_log_enabled() {
        return;
    }
    // Filter: Only print if file is NULL (Non-Tensor allocations like Vec, String)
    if !file.is_null() {
        return;
    }

    let file_str = "non_tensor"; 
    eprintln!("[ALLOC] File: {}, Line: {}, Size: {}, Ptr: {:p}", file_str, line, size, ptr);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_log_free(
    ptr: *const std::ffi::c_void,
    file: *const std::os::raw::c_char,
    line: i32,
) {
    if !mem_log_enabled() {
        return;
    }
    // Filter: Only print if file is NULL
    if !file.is_null() {
        return;
    }

    let file_str = "non_tensor";
    eprintln!("[FREE] File: {}, Line: {}, Ptr: {:p}", file_str, line, ptr);
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_trace_mem(
    file: *const std::os::raw::c_char,
    line: u32,
    col: u32,
    tag: *const std::os::raw::c_char,
) {
    if !mem_trace_enabled() {
        return;
    }
    let file_str = if !file.is_null() {
        unsafe { std::ffi::CStr::from_ptr(file).to_string_lossy().into_owned() }
    } else {
        "unknown".to_string()
    };
    let tag_str = if !tag.is_null() {
        unsafe { std::ffi::CStr::from_ptr(tag).to_string_lossy().into_owned() }
    } else {
        "stmt".to_string()
    };
    let rss_mb = tl_get_memory_mb();
    let pool_count = tl_get_pool_count();
    let refcount_count = tl_get_refcount_count();
    let scope_depth = tl_get_scope_depth();
    eprintln!(
        "[TL_MEM_TRACE] {}:{}:{} tag={} rss_mb={} pool_count={} refcount_count={} scope_depth={}",
        file_str,
        line,
        col,
        tag_str,
        rss_mb,
        pool_count,
        refcount_count,
        scope_depth
    );
}


// Comparison Ops returning Mask (Tensor of 0.0/1.0)
macro_rules! impl_cmp_op {
    ($name:ident, $op:ident) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
            use crate::error::RuntimeError;
            unsafe {
                let t_a = &(*a).0;
                let t_b = &(*b).0;
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Candle binary ops support broadcasting
                    let result = t_a
                        .$op(t_b)
                        .map_err(|e| RuntimeError::InternalError(e.to_string()))?
                        .to_dtype(candle_core::DType::F32) // Convert bool/u8 to f32
                        .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
                    Ok(make_tensor(result))
                }));
                return_ptr_or_null(res)
            }
        }
    };
}

impl_cmp_op!(tl_tensor_eq, eq);
impl_cmp_op!(tl_tensor_neq, ne);
impl_cmp_op!(tl_tensor_gt, gt);
impl_cmp_op!(tl_tensor_ge, ge);
impl_cmp_op!(tl_tensor_lt, lt);
impl_cmp_op!(tl_tensor_le, le);

// Remainder (Mod)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rem(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
             let div = (t_a / t_b).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let floor = div.floor().map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let mul = (floor * t_b).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let res = (t_a - mul).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             Ok(make_tensor(res))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {

    use crate::error::RuntimeError;
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .broadcast_add(t_b)
                .or_else(|_| t_a.add(t_b))
                .map_err(|e| RuntimeError::ShapeMismatch(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .broadcast_sub(t_b)
                .or_else(|_| t_a.sub(t_b))
                .map_err(|e| RuntimeError::ShapeMismatch(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .broadcast_mul(t_b)
                .or_else(|_| t_a.mul(t_b))
                .map_err(|e| RuntimeError::ShapeMismatch(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_softmax(tensor: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t = &(*tensor).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let d = dim as usize;
            let result = candle_nn::ops::softmax(t, d)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cross_entropy(
    logits: *mut OpaqueTensor,
    targets: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if logits.is_null() || targets.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null tensor".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let l = &(*logits).0;
        let t = &(*targets).0;

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Expect targets to be F32 (0.0, 1.0, 2.0...) -> Cast to U32 for indices
            let t_u32 = t
                .to_dtype(candle_core::DType::U32)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            // Log Softmax on last dim (-1)
            let log_sm = candle_nn::ops::log_softmax(l, candle_core::D::Minus1)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            // NLL
            let loss = candle_nn::loss::nll(&log_sm, &t_u32)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            Ok(make_tensor(loss))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_detach(t: *mut OpaqueTensor, req_grad: bool) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if t.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null tensor".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let tensor = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let detached = tensor.detach();
            let ptr = if req_grad {
                let var = candle_core::Var::from_tensor(&detached)
                    .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
                make_var(var)
            } else {
                make_tensor(detached)
            };
            Ok(ptr)
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_enable_grad(t: *mut OpaqueTensor) {
    unsafe {
        let tensor_wrapper = &mut *t;
        if tensor_wrapper.1.is_none() {
            // Determine if we can create a Var.
            // Var::from_tensor requires contiguous layout usually, which wrapper.0 likely is.
            let v = candle_core::Var::from_tensor(&tensor_wrapper.0).unwrap();
            let arc = Arc::new(v);
            tensor_wrapper.1 = Some(arc.clone());
            // Update the inner tensor to match the Var's tensor
            tensor_wrapper.0 = arc.as_tensor().clone();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print(t: *const OpaqueTensor) {
    if t.is_null() {
        println!("Tensor(NULL)");
        return;
    }
    unsafe {
        let tensor = &(*t).0;
        // Logic Integration: Check if tensor is i64 and contains entities
        if tensor.dtype() == candle_core::DType::I64 {
            // Import helper from knowledge_base (assumed linked)
            unsafe extern "C" {
                fn tl_kb_get_entity_name(id: i64) -> *const std::os::raw::c_char;
            }

            // Custom formatting
            let dims = tensor.dims();
            if dims.len() == 2 && dims[1] == 1 {
                // Vector-like
                print!("[");
                let vec: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
                for (i, &val) in vec.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }

                    let name_ptr = tl_kb_get_entity_name(val);
                    if !name_ptr.is_null() {
                        let c_str = std::ffi::CStr::from_ptr(name_ptr);
                        print!("{}", c_str.to_string_lossy());
                    } else {
                        print!("{}", val);
                    }
                }
                println!("]");
                return;
            }
        }
        println!("{}", tensor);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_display(t: *const OpaqueTensor) {
    use std::io::Write;
    if t.is_null() {
        print!("Tensor(NULL)");
    } else {
        let tensor = unsafe { &(*t).0 };
        // Logic Integration: Check if tensor is i64 and contains entities
        if tensor.dtype() == candle_core::DType::I64 {
            unsafe extern "C" {
                fn tl_kb_get_entity_name(id: i64) -> *const std::os::raw::c_char;
            }

            let dims = tensor.dims();
            if dims.len() == 2 && dims[1] == 1 {
                print!("[");
                let vec: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
                for (i, &val) in vec.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }

                    let name_ptr = unsafe { tl_kb_get_entity_name(val) };
                    if !name_ptr.is_null() {
                        let c_str = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
                        print!("{}", c_str.to_string_lossy());
                    } else {
                        print!("{}", val);
                    }
                }
                print!("]");
                let _ = std::io::stdout().flush();
                return;
            }
        }
        print!("{}", tensor);
    }
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_device_id(t: *const OpaqueTensor) -> i64 {
    if t.is_null() {
        return -1;
    }
    unsafe {
        let tensor = &(*t).0;
        device_to_id(tensor.device()) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_2(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_1(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print_3(t: *const OpaqueTensor) {
    tl_tensor_print(t);
}

/// Internal function to free tensor resources without unregistering
/// Used by MemoryManager to avoid deadlock
pub(crate) fn free_tensor_resources(t: *mut OpaqueTensor) -> FreeOutcome {
    if t.is_null() {
        return FreeOutcome::Freed;
    }
    unsafe {
        // Skip arena-allocated tensors (they are NOT poolable)
        if arena::tl_arena_contains(t as *mut std::ffi::c_void) {
            // Arena-allocated tensors MUST be dropped to release Candle resources (GPU memory, etc)
            // The memory for OpaqueTensor itself is reclaimed by arena reset, but the inner content needs Drop.
            // println!("DEBUG: Drop arena tensor {:p}", t);
            std::ptr::drop_in_place(t);
            return FreeOutcome::ArenaDrop;
        }

        // Try to release to pool (heap-allocated tensors only)
        let tensor = &(*t).0;
        let num_elements = tensor.elem_count();
        let dtype_id = dtype_to_id(tensor.dtype());
        let device_id = device_to_id(tensor.device());

        if let Ok(mut pool) = memory_manager::TENSOR_POOL.lock() {
            match pool.release(t, num_elements, dtype_id, device_id) {
                memory_manager::PoolOutcome::Pooled => return FreeOutcome::Pooled,
                memory_manager::PoolOutcome::Duplicate => return FreeOutcome::Pooled,
                memory_manager::PoolOutcome::Full => { /* Fallthrough to free */ }
            }
        }

        // Pool is full or lock failed, actually free
        // println!("DEBUG: Actually free tensor {:p}", t);
        let _ = Box::from_raw(t);
        FreeOutcome::Freed
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        memory_manager::tl_tensor_release(t);
    }
}

/// Finalize a tensor (drop content) without freeing the struct memory
/// Used for Slot-backed tensors where the container is managed by the slot/stack
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_finalize(t: *mut OpaqueTensor) {
    if !t.is_null() {
         unsafe {
             // Drop the content (Tensor, Var, etc)
             std::ptr::drop_in_place(t);
         }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_clone(t: *const OpaqueTensor) -> *mut OpaqueTensor {
    // use crate::error::RuntimeError;
    unsafe {
        if t.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null tensor".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let tensor = &(*t).0;
        let var_ref = &(*t).1;

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let cloned = tensor.clone();

            // Clone the Arc<Var> if it exists to maintain gradient tracking
            let cloned_var = var_ref.as_ref().map(Arc::clone);
            let boxed = Box::new(OpaqueTensor(cloned, cloned_var, None));

            let ptr = Box::into_raw(boxed);
            memory_manager::register_tensor_global(ptr);
            record_tensor_alloc("tensor_clone", ptr, &(*ptr).0, false);

            Ok(ptr)
        }));

        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.neg().unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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
            crate::error::set_last_error("tl_tensor_from_vec_u8: Out of bounds", crate::error::RuntimeErrorCode::IndexOutOfBounds);
            return std::ptr::null_mut();
        }

        let sub_vec = &vec[offset..offset + total_elements];
        let data_f32: Vec<f32> = sub_vec.iter().map(|&b| b as f32 / 255.0).collect();

        let device = get_device();
        let t_cpu =
            candle_core::Tensor::from_vec(data_f32, shape, &candle_core::Device::Cpu).unwrap();
        let tensor = if device.is_metal() || device.is_cuda() {
            t_cpu.to_device(&device).unwrap()
        } else {
            t_cpu
        };

        let tensor_with_grad = OpaqueTensor(tensor, None, None);

        Box::into_raw(Box::new(tensor_with_grad))
    }
}

#[unsafe(no_mangle)]
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
            crate::error::set_last_error("tl_tensor_from_u8_labels: Out of bounds", crate::error::RuntimeErrorCode::IndexOutOfBounds);
            return std::ptr::null_mut();
        }

        let sub_vec = &vec[offset..offset + count];
        let data_i64: Vec<i64> = sub_vec.iter().map(|&b| b as i64).collect();
        let shape = vec![count];

        let device = get_device();
        let t_cpu =
            candle_core::Tensor::from_vec(data_i64, shape, &candle_core::Device::Cpu).unwrap();
        let tensor = if device.is_metal() || device.is_cuda() {
            t_cpu.to_device(&device).unwrap()
        } else {
            t_cpu
        };

        let tensor_with_grad = OpaqueTensor(tensor, None, None);

        Box::into_raw(Box::new(tensor_with_grad))
    }
}

#[unsafe(no_mangle)]
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
            dt => {
                crate::error::set_last_error(format!("tl_tensor_get: Unsupported dtype {:?}", dt), crate::error::RuntimeErrorCode::TypeMismatch);
                panic!("tl_tensor_get: Unsupported dtype {:?}", dt);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_item(t: *mut OpaqueTensor) -> c_float {
    tl_tensor_get(t, 0)
}

#[unsafe(no_mangle)]
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
        // Mutate the existing OpaqueTensor in-place
        (*t).0 = new_tensor;
        // Return original pointer
        t
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_slice(t: *mut OpaqueTensor, start: i64, len: i64) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Slice along first dimension
        let result = tensor.narrow(0, start as usize, len as usize).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_i64(v: i64) {
    println!("{}", v);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_i64(v: i64) {
    print!("{}", v);
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_i32(v: i32) {
    println!("{}", v);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_i32(v: i32) {
    print!("{}", v);
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_ptr(_ptr: *const std::ffi::c_void) {
    // Debug function - no output in production
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_f32(v: c_float) {
    println!("{}", v);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_f32(v: c_float) {
    print!("{}", v);
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_f64(v: f64) {
    println!("{}", v);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_f64(v: f64) {
    print!("{}", v);
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_bool(v: bool) {
    if v {
        println!("true");
    } else {
        println!("false");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_bool(v: bool) {
    if v {
        print!("true");
    } else {
        print!("false");
    }
    let _ = std::io::stdout().flush();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_abs(v: c_float) -> c_float {
    v.abs()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_acos(v: c_float) -> c_float {
    v.acos()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_acosh(v: c_float) -> c_float {
    v.acosh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_asin(v: c_float) -> c_float {
    v.asin()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_asinh(v: c_float) -> c_float {
    v.asinh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atan(v: c_float) -> c_float {
    v.atan()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atan2(v: c_float, other: c_float) -> c_float {
    v.atan2(other)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atanh(v: c_float) -> c_float {
    v.atanh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_cbrt(v: c_float) -> c_float {
    v.cbrt()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_ceil(v: c_float) -> c_float {
    v.ceil()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_copysign(v: c_float, sign: c_float) -> c_float {
    v.copysign(sign)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_cos(v: c_float) -> c_float {
    v.cos()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_cosh(v: c_float) -> c_float {
    v.cosh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp(v: c_float) -> c_float {
    v.exp()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp2(v: c_float) -> c_float {
    v.exp2()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp_m1(v: c_float) -> c_float {
    v.exp_m1()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_floor(v: c_float) -> c_float {
    v.floor()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_fract(v: c_float) -> c_float {
    v.fract()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_hypot(v: c_float, other: c_float) -> c_float {
    v.hypot(other)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_ln(v: c_float) -> c_float {
    v.ln()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_ln_1p(v: c_float) -> c_float {
    v.ln_1p()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log(v: c_float, base: c_float) -> c_float {
    v.log(base)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log10(v: c_float) -> c_float {
    v.log10()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log2(v: c_float) -> c_float {
    v.log2()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_powf(v: c_float, exp: c_float) -> c_float {
    v.powf(exp)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_powi(v: c_float, exp: i64) -> c_float {
    v.powi(exp as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_recip(v: c_float) -> c_float {
    v.recip()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_round(v: c_float) -> c_float {
    v.round()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_signum(v: c_float) -> c_float {
    v.signum()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sin(v: c_float) -> c_float {
    v.sin()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sinh(v: c_float) -> c_float {
    v.sinh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sqrt(v: c_float) -> c_float {
    v.sqrt()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_tan(v: c_float) -> c_float {
    v.tan()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_tanh(v: c_float) -> c_float {
    v.tanh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_to_degrees(v: c_float) -> c_float {
    v.to_degrees()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_to_radians(v: c_float) -> c_float {
    v.to_radians()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_trunc(v: c_float) -> c_float {
    v.trunc()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_abs(v: f64) -> f64 {
    v.abs()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_acos(v: f64) -> f64 {
    v.acos()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_acosh(v: f64) -> f64 {
    v.acosh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_asin(v: f64) -> f64 {
    v.asin()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_asinh(v: f64) -> f64 {
    v.asinh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_atan(v: f64) -> f64 {
    v.atan()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_atan2(v: f64, other: f64) -> f64 {
    v.atan2(other)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_atanh(v: f64) -> f64 {
    v.atanh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cbrt(v: f64) -> f64 {
    v.cbrt()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ceil(v: f64) -> f64 {
    v.ceil()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_copysign(v: f64, sign: f64) -> f64 {
    v.copysign(sign)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cos(v: f64) -> f64 {
    v.cos()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cosh(v: f64) -> f64 {
    v.cosh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_exp(v: f64) -> f64 {
    v.exp()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_exp2(v: f64) -> f64 {
    v.exp2()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_exp_m1(v: f64) -> f64 {
    v.exp_m1()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_floor(v: f64) -> f64 {
    v.floor()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_fract(v: f64) -> f64 {
    v.fract()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_hypot(v: f64, other: f64) -> f64 {
    v.hypot(other)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ln(v: f64) -> f64 {
    v.ln()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ln_1p(v: f64) -> f64 {
    v.ln_1p()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_log(v: f64, base: f64) -> f64 {
    v.log(base)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_log10(v: f64) -> f64 {
    v.log10()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_log2(v: f64) -> f64 {
    v.log2()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_powf(v: f64, exp: f64) -> f64 {
    v.powf(exp)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_powi(v: f64, exp: i64) -> f64 {
    v.powi(exp as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_recip(v: f64) -> f64 {
    v.recip()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_round(v: f64) -> f64 {
    v.round()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_signum(v: f64) -> f64 {
    v.signum()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_sin(v: f64) -> f64 {
    v.sin()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_sinh(v: f64) -> f64 {
    v.sinh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_sqrt(v: f64) -> f64 {
    v.sqrt()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_tan(v: f64) -> f64 {
    v.tan()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_tanh(v: f64) -> f64 {
    v.tanh()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_to_degrees(v: f64) -> f64 {
    v.to_degrees()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_to_radians(v: f64) -> f64 {
    v.to_radians()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_trunc(v: f64) -> f64 {
    v.trunc()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_abs(v: i64) -> i64 {
    v.abs()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_signum(v: i64) -> i64 {
    v.signum()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_pow(v: i64, exp: i64) -> i64 {
    v.pow(exp as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_div_euclid(v: i64, rhs: i64) -> i64 {
    v.div_euclid(rhs)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_rem_euclid(v: i64, rhs: i64) -> i64 {
    v.rem_euclid(rhs)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_is_positive(v: i64) -> bool {
    v.is_positive()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_is_negative(v: i64) -> bool {
    v.is_negative()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_abs(v: i32) -> i32 {
    v.abs()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_signum(v: i32) -> i32 {
    v.signum()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_pow(v: i32, exp: i32) -> i32 {
    v.pow(exp as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_div_euclid(v: i32, rhs: i32) -> i32 {
    v.div_euclid(rhs)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_rem_euclid(v: i32, rhs: i32) -> i32 {
    v.rem_euclid(rhs)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_is_positive(v: i32) -> bool {
    v.is_positive()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_is_negative(v: i32) -> bool {
    v.is_negative()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_print_string(s: *const std::os::raw::c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        if let Ok(c_str) = std::ffi::CStr::from_ptr(s).to_str() {
            println!("{}", c_str);
            let _ = std::io::stdout().flush();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_display_string(s: *const std::os::raw::c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        if let Ok(c_str) = std::ffi::CStr::from_ptr(s).to_str() {
            print!("{}", c_str);
            let _ = std::io::stdout().flush();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_set_device(device_ptr: *const std::ffi::c_void) {
    if device_ptr.is_null() {
        return;
    }
    // Read tag (first i32)
    let tag = unsafe { *(device_ptr as *const i32) };

    // Map tag to device string
    // Enum Device { Auto=0, Cpu=1, Metal=2, Cuda=3 }
    let device_str = match tag {
        0 => "auto",
        1 => "cpu",
        2 => "metal",
        3 => "cuda",
        _ => {
            eprintln!("Runtime Warning: Unknown Device enum tag: {}", tag);
            return;
        }
    };

    let _ = crate::device::DEVICE_MANAGER
        .lock()
        .unwrap()
        .set_device(device_str);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_device(
    tensor: *mut OpaqueTensor,
    device_name: *const i8,
) -> *mut OpaqueTensor {
    if tensor.is_null() || device_name.is_null() {
        return tensor;
    } // Or panic?

    let c_str = unsafe { std::ffi::CStr::from_ptr(device_name) };
    let device_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return tensor,
    };

    let target_device = match device_str {
        "cpu" => candle_core::Device::Cpu,
        "cuda" => match candle_core::Device::new_cuda(0) {
            Ok(device) => device,
            Err(e) => {
                eprintln!("Failed to initialize CUDA device: {}", e);
                return std::ptr::null_mut();
            }
        },
        "metal" => match candle_core::Device::new_metal(0) {
            Ok(device) => device,
            Err(e) => {
                eprintln!("Failed to initialize Metal device: {}", e);
                return std::ptr::null_mut();
            }
        },
        _ => return tensor,
    };

    let t = unsafe { &(*tensor).0 };
    match t.to_device(&target_device) {
        Ok(new_t) => make_tensor(new_t),
        Err(_e) => {
            // eprintln!("Failed to move tensor to device: {}", e); // Removed debug print
            std::ptr::null_mut() // Return null on error, consistent with other error paths
        }
    }
}

pub fn force_link() {
    std::hint::black_box(tl_set_device as *const ());
    std::hint::black_box(tl_tensor_to_device as *const ());
    std::hint::black_box(tl_print_string as *const ());
    std::hint::black_box(tl_tensor_device_id as *const ());
    std::hint::black_box(tl_clear_grads as *const ());
    std::hint::black_box(tl_tensor_randn_debug as *const ());
    std::hint::black_box(tl_tensor_set_f32_md as *const ());
    std::hint::black_box(tl_tensor_free as *const ());
    std::hint::black_box(tl_tensor_pow_scalar as *const ());
    std::hint::black_box(memory_manager::tl_mem_unregister as *const ());
    std::hint::black_box(memory_manager::tl_mem_function_exit as *const ());
    std::hint::black_box(memory_manager::tl_mem_get_buffer as *const ());
    std::hint::black_box(crate::arena::tl_arena_init as *const ());
    std::hint::black_box(crate::arena::tl_arena_is_active as *const ());
    std::hint::black_box(crate::arena::tl_arena_alloc as *const ());
    std::hint::black_box(crate::arena::tl_arena_reset as *const ());
    std::hint::black_box(crate::arena::tl_arena_free as *const ());
    std::hint::black_box(crate::arena::tl_arena_get_offset as *const ());
    std::hint::black_box(crate::arena::tl_arena_get_capacity as *const ());
    std::hint::black_box(tl_tensor_reshape_dims as *const ());
    std::hint::black_box(crate::stdlib::tl_prompt as *const ());
    std::hint::black_box(crate::stdlib::tl_string_contains as *const ());
    std::hint::black_box(crate::stdlib::tl_string_concat as *const ());
    std::hint::black_box(crate::stdlib::tl_string_from_int as *const ());
    std::hint::black_box(crate::llm::tl_tensor_sample as *const ());
    std::hint::black_box(crate::args::tl_args_count as *const ());
    std::hint::black_box(crate::args::tl_args_get as *const ());
    std::hint::black_box(crate::stdlib::tl_string_char_at as *const ());
    std::hint::black_box(crate::stdlib::tl_string_len as *const ());
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_i64_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
) -> i64 {
    unsafe {
        let tensor = &(*t).0;
        let idxs = slice::from_raw_parts(indices, rank);
        let idxs_usize: Vec<usize> = idxs.iter().map(|&x| x as usize).collect();

        let dims = tensor.dims();
        let mut flat_idx = 0;
        let mut stride = 1;
        for i in (0..rank).rev() {
            flat_idx += idxs_usize[i] * stride;
            stride *= dims[i];
        }

        let scalar = tensor.flatten_all().unwrap().get(flat_idx).unwrap();
        match scalar.dtype() {
            candle_core::DType::I64 => scalar.to_scalar::<i64>().unwrap(),
            candle_core::DType::U32 => scalar.to_scalar::<u32>().unwrap() as i64,
            candle_core::DType::U8 => scalar.to_scalar::<u8>().unwrap() as i64,
            candle_core::DType::F32 => scalar.to_scalar::<f32>().unwrap() as i64,
            candle_core::DType::F64 => scalar.to_scalar::<f64>().unwrap() as i64,
            dt => {
                crate::error::set_last_error(format!("tl_tensor_get_i64_md: Unsupported dtype {:?}", dt), crate::error::RuntimeErrorCode::TypeMismatch);
                panic!("tl_tensor_get_i64_md: Unsupported dtype {:?}", dt);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_transpose(
    t: *mut OpaqueTensor,
    dim0: usize,
    dim1: usize,
) -> *mut OpaqueTensor {
    if t.is_null() {
        // println!("ERROR: tl_tensor_transpose received NULL pointer"); // Removed debug print
        std::process::abort();
    }
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.transpose(dim0, dim1).unwrap();
        let ptr = make_tensor(result);
        ptr
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_clear_grads() {
    LATEST_GRADS.with(|g| {
        *g.borrow_mut() = None;
    });
}
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let tensor = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = tensor
                .sum_all()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

// Scalar variants for compound assignment operators
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mul_assign_scalar_f32(ref_t: *mut OpaqueTensor, scalar: f32) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let result = (t_dst * scalar as f64).unwrap();
        (*ref_t).0 = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div_assign_scalar_f32(ref_t: *mut OpaqueTensor, scalar: f32) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let result = (t_dst / scalar as f64).unwrap();
        (*ref_t).0 = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_add_assign_scalar_f32(ref_t: *mut OpaqueTensor, scalar: f32) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let result = (t_dst + scalar as f64).unwrap();
        (*ref_t).0 = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub_assign_scalar_f32(ref_t: *mut OpaqueTensor, scalar: f32) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let result = (t_dst - scalar as f64).unwrap();
        (*ref_t).0 = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mod_assign(ref_t: *mut OpaqueTensor, val_t: *mut OpaqueTensor) {
    unsafe {
        let t_dst = &(*ref_t).0;
        let t_src = &(*val_t).0;
        // Compute modulo: a % b = a - b * floor(a/b)
        let div_result = t_dst.broadcast_div(t_src).unwrap();
        let floor_result = div_result.floor().unwrap();
        let mul_back = floor_result.broadcast_mul(t_src).unwrap();
        let result = t_dst.broadcast_sub(&mul_back).unwrap();
        (*ref_t).0 = result;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mod_assign_scalar_f32(ref_t: *mut OpaqueTensor, scalar: f32) {
    use crate::error::RuntimeError;
    unsafe {
        let t_dst = &(*ref_t).0;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
             // Ensure scalar matches tensor dtype to avoid mismatch errors
             let dtype = t_dst.dtype();
             let scalar_tensor = Tensor::new(scalar, t_dst.device())
                 .and_then(|t| t.to_dtype(dtype))
                 .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

             // a % b = a - b * floor(a/b)
             // Use broadcasting with scalar tensor
             let div_result = t_dst.broadcast_div(&scalar_tensor).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let floor_result = div_result.floor().map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let mul_back = floor_result.broadcast_mul(&scalar_tensor).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             let result = t_dst.broadcast_sub(&mul_back).map_err(|e| RuntimeError::InternalError(e.to_string()))?;
             
             (*ref_t).0 = result;
             Ok::<(), RuntimeError>(())
        })).map_err(|_| {
             eprintln!("Panic in tl_tensor_mod_assign_scalar_f32");
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_new(
    t: *mut OpaqueTensor,
    shape_tensor: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        // println!("DEBUG: Inside tl_tensor_reshape_new");
        let tensor = &(*t).0;
        let shape_t = &(*shape_tensor).0;

        // Support both I64 and F32 shape tensors
        let new_shape: Vec<usize> = match shape_t.dtype() {
            candle_core::DType::I64 => {
                let shape_vec: Vec<i64> = shape_t.flatten_all().unwrap().to_vec1().unwrap();
                shape_vec.iter().map(|&x| x as usize).collect()
            }
            candle_core::DType::F32 => {
                let shape_vec: Vec<f32> = shape_t.flatten_all().unwrap().to_vec1().unwrap();
                shape_vec.iter().map(|&x| x as usize).collect()
            }
            dt => {
                eprintln!("Error: Reshape shape tensor must be numeric, got {:?}", dt);
                std::process::abort();
            }
        };

        // println!("DEBUG: Calculated new shape: {:?}", new_shape);

        match tensor.reshape(new_shape) {
            Ok(result) => make_tensor(result),
            Err(e) => {
                println!("Error reshaping tensor (in tl_tensor_reshape_new): {}", e);
                std::process::abort();
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_dims(
    t: *mut OpaqueTensor,
    dims_ptr: *const i64,
    num_dims: i64,
) -> *mut OpaqueTensor {
    unsafe {
        // println!("DEBUG: Inside tl_tensor_reshape_dims");
        if t.is_null() || dims_ptr.is_null() {
            crate::error::set_last_error(format!("Null pointer passed to reshape_dims: t={:?}, dims={:?}", t, dims_ptr), crate::error::RuntimeErrorCode::NullPointerError);
            return std::ptr::null_mut();
        }
        let tensor = &(*t).0;
        let dims_slice = std::slice::from_raw_parts(dims_ptr, num_dims as usize);
        let new_shape: Vec<usize> = dims_slice.iter().map(|&x| x as usize).collect();

        match tensor.reshape(new_shape) {
            Ok(result) => make_tensor(result),
            Err(e) => {
                println!("Error reshaping tensor (in tl_tensor_reshape_dims): {}", e);
                std::process::abort();
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .broadcast_div(t_b)
                .or_else(|_| t_a.div(t_b))
                .map_err(|e| RuntimeError::ShapeMismatch(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let tensor = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = tensor
                .exp()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_pow(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if a.is_null() || b.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null tensor".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .broadcast_pow(t_b)
                .or_else(|_| t_a.pow(t_b))
                .map_err(|e| RuntimeError::ShapeMismatch(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

/// Scalar exponent version of pow - more common use case
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_pow_scalar(a: *mut OpaqueTensor, exp: c_float) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if a.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null tensor".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }
        let t_a = &(*a).0;
        let exp_f64 = exp as f64;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = t_a
                .powf(exp_f64)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let tensor = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = tensor
                .log()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

// --- Transformer Support ---

// Narrow/Slice: Extract a slice along a dimension
// (t, dim, start, length) -> t[..., start:start+length, ...]
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_narrow(
    t: *mut OpaqueTensor,
    dim: i64,
    start: i64,
    length: i64,
) -> *mut OpaqueTensor {
    let tensor = unsafe { &(*t).0 };
    let result = tensor
        .narrow(dim as usize, start as usize, length as usize)
        .unwrap();
    make_tensor(result)
}

// Repeat interleave: Repeat elements along a dimension
// Used for GQA to expand K/V heads to match Q heads
// (t, repeats, dim) -> expanded tensor
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_repeat_interleave(
    t: *mut OpaqueTensor,
    repeats: i64,
    dim: i64,
) -> *mut OpaqueTensor {
    let tensor = unsafe { &(*t).0 };
    let dim = dim as usize;
    let repeats = repeats as usize;

    // Manual repeat interleave implementation:
    // For each position along dim, repeat the slice 'repeats' times
    let dims = tensor.dims();
    let dim_size = dims[dim];

    let mut slices = Vec::new();
    for i in 0..dim_size {
        let slice = tensor.narrow(dim, i, 1).unwrap();
        for _ in 0..repeats {
            slices.push(slice.clone());
        }
    }

    // Concatenate all slices
    let slice_refs: Vec<&Tensor> = slices.iter().collect();
    let result = Tensor::cat(&slice_refs, dim).unwrap();
    make_tensor(result)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.sin().unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_scale(t: *mut OpaqueTensor, scale: f32) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = (tensor * (scale as f64)).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.cos().unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.relu().unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.gelu().unwrap();
    make_tensor(res)
}

// Lower triangular mask (diagonal=0 includes the diagonal)
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum_dim(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        let t_ref = &(*t).0;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = if keep_dim {
                t_ref
                    .sum_keepdim(dim)
                    .map_err(|e| RuntimeError::InternalError(e.to_string()))?
            } else {
                t_ref
                    .sum(dim)
                    .map_err(|e| RuntimeError::InternalError(e.to_string()))?
            };
            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

// Embedding: indices [B, S], weights [V, D] -> [B, S, D]
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_embedding(
    indices: *mut OpaqueTensor,
    weights: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    let indices_tensor = unsafe { &(*indices).0 };
    let weights_tensor = unsafe { &(*weights).0 };
    let w_dims = weights_tensor.dims();

    if w_dims.len() != 2 {
        crate::error::set_last_error(format!("Embedding weights must be 2-dimensional, got {:?}", w_dims), crate::error::RuntimeErrorCode::ShapeMismatch);
        return std::ptr::null_mut();
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = tensor.sqrt().unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_matmul(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    use crate::error::RuntimeError;
    unsafe {
        if a.is_null() || b.is_null() {
            handle_runtime_error_internal(
                crate::error::RuntimeErrorCode::NullPointerError as u32,
                "Null pointer".to_string(),
                None,
                0,
                0,
            );
            return std::ptr::null_mut();
        }

        let t_a = &(*a).0;
        let t_b = &(*b).0;

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Make tensors contiguous before matmul to avoid striding issues
            let t_a_contig = t_a
                .contiguous()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
            let t_b_contig = t_b
                .contiguous()
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            // Use broadcast_matmul to support [B, S, D] x [D, O] -> [B, S, O]
            let result = t_a_contig
                .broadcast_matmul(&t_b_contig)
                .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

            Ok(make_tensor(result))
        }));
        return_ptr_or_null(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // Candleのcontiguous()を使用してメモリレイアウトを連続化
        let result = tensor.contiguous().unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

use candle_nn::Module; // For forward()

pub enum LoadedTensor {
    Standard(Tensor),
    Quantized(Arc<candle_core::quantized::QTensor>),
}

#[repr(C)]
pub struct OpaqueTensorMap(pub std::collections::HashMap<String, LoadedTensor>);

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_new() -> *mut OpaqueTensorMap {
    let map = OpaqueTensorMap(std::collections::HashMap::new());
    Box::into_raw(Box::new(map))
}

#[unsafe(no_mangle)]
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
        map_ref.insert(key, LoadedTensor::Standard(t_ref.clone()));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_save(map: *mut OpaqueTensorMap, path: *const std::os::raw::c_char) {
    unsafe {
        let map_ref = &(*map).0;
        let mut save_map = std::collections::HashMap::new();
        for (k, v) in map_ref.iter() {
            if let LoadedTensor::Standard(t) = v {
                save_map.insert(k.clone(), t.clone());
            }
        }
        let p_str = std::ffi::CStr::from_ptr(path).to_str().unwrap();
        candle_core::safetensors::save(&save_map, p_str).unwrap();
        // println!("Saved model to {}", p_str);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_load(path: *const std::os::raw::c_char) -> *mut OpaqueTensorMap {
    unsafe {
        let p_str = std::ffi::CStr::from_ptr(path).to_str().unwrap();
        let device = get_device();
        let map =
            candle_core::safetensors::load(p_str, &device).expect("Failed to load model file");

        let mut loaded_map = std::collections::HashMap::new();
        for (k, v) in map {
            loaded_map.insert(k, LoadedTensor::Standard(v));
        }

        let opaque = OpaqueTensorMap(loaded_map);
        Box::into_raw(Box::new(opaque))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new_causal_mask(dim: i64) -> *mut OpaqueTensor {
    // Return [dim, dim] matrix.
    // Tril (inc diag) = 0.0
    // Upper = -1e9
    let d = dim as usize;
    let device = get_device();
    // Manual tril logic:
    // row >= col
    let idx = Tensor::arange(0u32, d as u32, &device).unwrap();
    let row_idx = idx.reshape((d, 1)).unwrap().broadcast_as((d, d)).unwrap();
    let col_idx = idx.reshape((1, d)).unwrap().broadcast_as((d, d)).unwrap();
    let mask_u8 = col_idx.le(&row_idx).unwrap(); // 1 where col <= row (tril)

    // We want 0 in lower (where mask_u8 is 1), and -inf in upper (where mask_u8 is 0).
    let neg_inf = Tensor::new(-1e9f32, &device)
        .unwrap()
        .broadcast_as((d, d))
        .unwrap();
    let zeros = Tensor::zeros((d, d), candle_core::DType::F32, &device).unwrap();

    // where mask_u8 == 1 ? zeros : neg_inf
    let mask = mask_u8.where_cond(&zeros, &neg_inf).unwrap();
    make_tensor(mask)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_i64(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let t_a = &(*a).0;
        let t_b = &(*b).0;
        let result = Tensor::cat(&[t_a, t_b], dim as usize).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get(
    map: *mut OpaqueTensorMap,
    name: *const std::os::raw::c_char,
) -> *mut OpaqueTensor {
    unsafe {
        let map_ref = &(*map).0;
        let c_str = std::ffi::CStr::from_ptr(name);
        let key = c_str.to_string_lossy();

        if let Some(loaded) = map_ref.get(key.as_ref()) {
            match loaded {
                LoadedTensor::Standard(t) => {
                    let device = get_device();
                    let t_on_device = if device_to_id(t.device()) == device_to_id(&device) {
                        t.clone()
                    } else {
                        t.to_device(&device).unwrap()
                    };
                    // Update not needed for standard? Or do we need to cache device move?
                    // Keeping behavior similar to before: if move happened, maybe update?
                    // But original code: map_mut.insert(key, t_on_device.clone())
                    // Let's duplicate that logic.
                    let map_mut = &mut (*map).0;
                    map_mut.insert(key.to_string(), LoadedTensor::Standard(t_on_device.clone()));
                    make_tensor(t_on_device)
                }
                LoadedTensor::Quantized(qt) => {
                    // Ephemeral dequantize for F32 usage (legacy/compat support)
                    let device = get_device();
                    let t = qt.dequantize(&device).unwrap();
                    make_tensor(t)
                }
            }
        } else {
            crate::error::set_last_error(format!("Weight '{}' not found in loaded file.", key), crate::error::RuntimeErrorCode::ArgumentError);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_free(map: *mut OpaqueTensorMap) {
    unsafe {
        let _ = Box::from_raw(map); // Drop
    }
}
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.to_dtype(candle_core::DType::F32).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.to_dtype(candle_core::DType::I64).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_shape(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let t = &(*t).0;
        let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
        let device = get_device();
        let t_cpu =
            Tensor::from_vec(dims.clone(), (dims.len(),), &candle_core::Device::Cpu).unwrap();
        let shape_tensor = if device.is_metal() || device.is_cuda() {
            t_cpu.to_device(&device).unwrap()
        } else {
            t_cpu
        };
        make_tensor(shape_tensor)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_void_len(ptr: *mut std::ffi::c_void) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let vec = &*(ptr as *mut Vec<*mut std::ffi::c_void>);
        vec.len()
    }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_void_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        tl_log_free(
            ptr,
            std::ptr::null(),
            0
        );
        unsafe {
            // Reconstruct Vec from raw pointer to drop the container
            // This frees the Vec struct logic (cap/len/buffer pointer)
            let _ = Box::from_raw(ptr as *mut Vec<*mut std::ffi::c_void>);
        }
    }
}

// --- Vec<u8> support for binary data ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_new() -> *mut Vec<u8> {
    let ptr = Box::into_raw(Box::new(Vec::<u8>::new()));
    tl_log_alloc(
        ptr as *const c_void,
        0, // size 0 initially
        std::ptr::null(), // unknown file
        0
    );
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_with_capacity(cap: usize) -> *mut Vec<u8> {
    let ptr = Box::into_raw(Box::new(Vec::<u8>::with_capacity(cap)));
    tl_log_alloc(
        ptr as *const c_void,
        cap as i64, 
        std::ptr::null(),
        0
    );
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_len(ptr: *mut Vec<u8>) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).len() }
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_push(ptr: *mut Vec<u8>, val: u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        (*ptr).push(val);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_u8_free(ptr: *mut Vec<u8>) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// String helper
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_new(s: *const std::os::raw::c_char) -> *mut std::os::raw::c_char {
    // println!("tl_string_new via mod.rs: {:p}", s);
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

// --- QTensor Support ---

pub struct OpaqueQTensor(pub Arc<candle_core::quantized::QTensor>);

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_map_get_quantized(
    map: i64,
    name: *const std::os::raw::c_char,
) -> usize {
    unsafe {
        use std::ffi::CStr;
        let map_ptr = map as *mut OpaqueTensorMap;
        if map_ptr.is_null() {
            crate::error::set_last_error("Map pointer is null", crate::error::RuntimeErrorCode::NullPointerError);
            return 0;
        }
        let map_ref = &(*map_ptr).0;
        let c_str = CStr::from_ptr(name);
        let key_str = c_str.to_str().unwrap();

        if let Some(loaded) = map_ref.get(key_str) {
            match loaded {
                LoadedTensor::Quantized(qt) => {
                    let arc = qt.clone();
                    Box::into_raw(Box::new(OpaqueQTensor(arc))) as usize
                }
                LoadedTensor::Standard(_) => {
                    crate::error::set_last_error(format!("Requested quantized tensor '{}', but found standard tensor.", key_str), crate::error::RuntimeErrorCode::TypeMismatch);
                    return 0;
                }
            }
        } else {
            crate::error::set_last_error(format!("Tensor '{}' not found in map.", key_str), crate::error::RuntimeErrorCode::ArgumentError);
            return 0;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_free(ptr: usize) {
    if ptr != 0 {
        unsafe {
            let _ = Box::from_raw(ptr as *mut OpaqueQTensor);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_qtensor_matmul(input: *mut OpaqueTensor, weight: usize) -> *mut OpaqueTensor {
    unsafe {
        let x_t = &(*input).0;
        let weight_ptr = weight as *mut OpaqueQTensor;
        let w_qt = &(*weight_ptr).0;

        // QMatMul::from_arc expects Arc<QTensor>
        let qmatmul = candle_core::quantized::QMatMul::from_arc(w_qt.clone()).unwrap();
        let result = qmatmul.forward(x_t).unwrap();
        make_tensor(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_manager;

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

// Added for Tensor Refactor (Global -> Instance)

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    // tan(x) = sin(x) / cos(x)
    let sin = t.sin().unwrap();
    let cos = t.cos().unwrap();
    let res = sin.div(&cos).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_abs(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.abs().unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sigmoid(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    // sigmoid = 1 / (1 + exp(-x))
    // Manual implementation for Metal compatibility
    let neg_t = t.neg().unwrap();
    let exp_neg = neg_t.exp().unwrap();
    let one = Tensor::ones(t.shape(), t.dtype(), t.device()).unwrap();
    let denom = exp_neg.add(&one).unwrap();
    let res = one.div(&denom).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tanh(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.tanh().unwrap();
    make_tensor(res)
}

// Reductions

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.flatten_all().unwrap().max(0).unwrap(); // dim 0 of flat
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max_dim(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = if keep_dim {
        t.max_keepdim(dim).unwrap()
    } else {
        t.max(dim).unwrap()
    };
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.flatten_all().unwrap().min(0).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min_dim(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = if keep_dim {
        t.min_keepdim(dim).unwrap()
    } else {
        t.min(dim).unwrap()
    };
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = t.flatten_all().unwrap().mean(0).unwrap();
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean_dim(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = if keep_dim {
        t.mean_keepdim(dim).unwrap()
    } else {
        t.mean(dim).unwrap()
    };
    make_tensor(res)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmin(
    t: *mut OpaqueTensor,
    dim: usize,
    keep_dim: bool,
) -> *mut OpaqueTensor {
    let t = unsafe { &(*t).0 };
    let res = if keep_dim {
        t.argmin_keepdim(dim).unwrap()
    } else {
        t.argmin(dim).unwrap()
    };
    // Ensure result is I64
    let res_i64 = res.to_dtype(candle_core::DType::I64).unwrap_or(res);
    make_tensor(res_i64)
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    padding: i64,
    stride: i64,
) -> *mut OpaqueTensor {
    unsafe {
        // candle_core::Tensor has conv2d method?
        // Or candle_nn::ops::conv2d is deprecated?
        // Actually, let's use the tensor method if available, or candle_nn::conv2d (function).

        // Checking candle docs: tensor.conv2d(weight, padding, stride, dilation, groups)
        let res = (*input)
            .0
            .conv2d(
                &(*weight).0,
                padding as usize,
                stride as usize,
                1, // dilation
                1, // groups
            )
            .unwrap();
        make_tensor(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_clamp(
    t: *mut OpaqueTensor,
    min_val: f32,
    max_val: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        // clamp works element-wise
        let res = tensor.clamp(min_val, max_val).unwrap();
        make_tensor(res)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_ones(
    rank: usize,
    shape: *const usize,
    req_grad: bool,
) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_data: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, rank).to_vec() };

    let device = get_device();
    let t = Tensor::ones(&shape_data[..], candle_core::DType::F32, &device).unwrap();

    if req_grad {
        let var = candle_core::Var::from_tensor(&t).unwrap();
        make_var(var)
    } else {
        make_tensor(t)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_exists(path: *const std::os::raw::c_char) -> bool {
    if path.is_null() {
        return false;
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_str().unwrap_or("") };
    if path_str.is_empty() {
        return false;
    }

    let path_buf = if path_str.starts_with("~") {
        if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(path_str.replace("~", &home))
        } else {
            std::path::PathBuf::from(path_str)
        }
    } else {
        std::path::PathBuf::from(path_str)
    };

    path_buf.exists()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_exists_i64(path: *const std::os::raw::c_char) -> i64 {
    if tl_file_exists(path) {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_download_file(
    url: *const std::os::raw::c_char,
    path: *const std::os::raw::c_char,
) -> i64 {
    let url_str = unsafe { std::ffi::CStr::from_ptr(url).to_string_lossy() };
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_string_lossy() };

    // Handle home directory expansion
    let path_buf = if path_str.starts_with("~") {
        if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(path_str.replace("~", &home))
        } else {
            std::path::PathBuf::from(path_str.as_ref())
        }
    } else {
        std::path::PathBuf::from(path_str.as_ref())
    };

    println!("Downloading from: {}", url_str);
    println!("Saving to: {:?}", path_buf);

    match reqwest::blocking::get(url_str.as_ref()) {
        Ok(mut response) => {
            if !response.status().is_success() {
                println!("Download failed: HTTP {}", response.status());
                return 0;
            }

            // Create directories if needed
            if let Some(parent) = path_buf.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    println!("Failed to create directories: {}", e);
                    return 0;
                }
            }

            match std::fs::File::create(&path_buf) {
                Ok(mut file) => {
                    let total_size = response.content_length();
                    if let Some(len) = total_size {
                        println!("Total size: {} bytes", len);
                    } else {
                        println!("Total size: unknown");
                    }

                    if let Err(e) = std::io::copy(&mut response, &mut file) {
                        println!("Error writing to file: {}", e);
                        return 0;
                    }
                    println!("Download complete!");
                    1
                }
                Err(e) => {
                    println!("Failed to create file: {}", e);
                    0
                }
            }
        }
        Err(e) => {
            println!("Request failed: {}", e);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_read_file(path: *const std::os::raw::c_char) -> *const std::os::raw::c_char {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_string_lossy() };
    if let Ok(content) = std::fs::read_to_string(path_str.as_ref()) {
        let c_string = std::ffi::CString::new(content.trim()).unwrap();
        c_string.into_raw()
    } else {
        std::ptr::null()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_write_file(path: *const std::os::raw::c_char, content: *const std::os::raw::c_char) -> i64 {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_string_lossy() };
    let content_str = unsafe { std::ffi::CStr::from_ptr(content).to_string_lossy() };
    
    if let Ok(_) = std::fs::write(path_str.as_ref(), content_str.as_ref()) {
        1
    } else {
        0
    }
}

// --- Specialized Vec Support ---

// I64
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_i64_len(ptr: *mut Vec<i64>) -> usize {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_i64_push(ptr: *mut Vec<i64>, val: i64) {
    if ptr.is_null() { return; }
    unsafe { (*ptr).push(val); }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_i64_get(ptr: *mut Vec<i64>, idx: usize) -> i64 {
    if ptr.is_null() { return 0; }
    unsafe {
        let vec = &*ptr;
        if idx < vec.len() { vec[idx] } else { 0 }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_i64_free(ptr: *mut Vec<i64>) {
    if !ptr.is_null() { 
        tl_log_free(
            ptr as *const std::ffi::c_void,
            std::ptr::null(),
            0
        );
        unsafe { let _ = Box::from_raw(ptr); } 
    }
}

// F32
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_f32_len(ptr: *mut Vec<f32>) -> usize {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_f32_push(ptr: *mut Vec<f32>, val: f32) {
    if ptr.is_null() { return; }
    unsafe { (*ptr).push(val); }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_f32_get(ptr: *mut Vec<f32>, idx: usize) -> f32 {
    if ptr.is_null() { return 0.0; }
    unsafe {
        let vec = &*ptr;
        if idx < vec.len() { vec[idx] } else { 0.0 }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_f32_free(ptr: *mut Vec<f32>) {
    if !ptr.is_null() { 
        tl_log_free(
            ptr as *const std::ffi::c_void,
            std::ptr::null(),
            0
        );
        unsafe { let _ = Box::from_raw(ptr); } 
    }
}

// Ptr (for Structs, Tensors, Strings etc.)
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_ptr_len(ptr: *mut Vec<*mut std::ffi::c_void>) -> usize {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_ptr_push(ptr: *mut Vec<*mut std::ffi::c_void>, val: *mut std::ffi::c_void) {
    if ptr.is_null() { return; }
    unsafe { 
        (*ptr).push(val);
        // Acquire reference for the vector (strong reference)
        memory_manager::tl_ptr_acquire(val);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_ptr_get(ptr: *mut Vec<*mut std::ffi::c_void>, idx: usize) -> *mut std::ffi::c_void {
    if ptr.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let vec = &*ptr;
        if idx < vec.len() { 
            let val = vec[idx];
            // Register in current scope (return owned reference)
            memory_manager::tl_mem_register_ptr(val);
            val 
        } else { 
            std::ptr::null_mut() 
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_ptr_free(ptr: *mut Vec<*mut std::ffi::c_void>) {
    if !ptr.is_null() { 
        tl_log_free(
            ptr as *const std::ffi::c_void,
            std::ptr::null(),
            0
        );
        // Try release via Manager
        let processed = memory_manager::tl_ptr_release_bool(ptr as *mut std::ffi::c_void);
        if !processed {
             // Not registered (Local variable). Manual Cleanup.
             unsafe {
                let v = std::ptr::read(ptr);
                for elem in &v {
                    memory_manager::tl_ptr_release(*elem);
                }
                drop(v);
                libc::free(ptr as *mut std::ffi::c_void);
             }
        }
    }
}

// Constructors
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_i64_new() -> *mut Vec<i64> {
   let ptr = Box::into_raw(Box::new(Vec::new()));
   tl_log_alloc(
        ptr as *const std::ffi::c_void,
        0, 
        std::ptr::null(),
        0
   );
   ptr
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_f32_new() -> *mut Vec<f32> {
   let ptr = Box::into_raw(Box::new(Vec::new()));
   tl_log_alloc(
        ptr as *const std::ffi::c_void,
        0, 
        std::ptr::null(),
        0
   );
   ptr
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_vec_ptr_new() -> *mut Vec<*mut std::ffi::c_void> {
   let ptr = Box::into_raw(Box::new(Vec::new()));
   tl_log_alloc(
        ptr as *const std::ffi::c_void,
        0, 
        std::ptr::null(),
        0
   );
   ptr
}
