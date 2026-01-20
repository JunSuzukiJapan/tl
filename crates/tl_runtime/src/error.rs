use crate::OpaqueTensor;
use std::ffi::{c_char, CString};
use std::ptr;

#[derive(Debug)]
pub enum RuntimeError {
    ShapeMismatch(String),
    DeviceError(String),
    AllocationError(String),
    NullPointerError(String),
    InternalError(String),
    // Add more as needed
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::ShapeMismatch(msg) => write!(f, "ShapeMismatch: {}", msg),
            RuntimeError::DeviceError(msg) => write!(f, "DeviceError: {}", msg),
            RuntimeError::AllocationError(msg) => write!(f, "AllocationError: {}", msg),
            RuntimeError::NullPointerError(msg) => write!(f, "NullPointerError: {}", msg),
            RuntimeError::InternalError(msg) => write!(f, "InternalError: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Result struct for FFI functions returning a Tensor.
/// If success, `tensor` is non-null and `error` is null.
/// If failure, `tensor` is null and `error` points to a C string describing the error.
/// The caller is responsible for freeing the error string if it is not null (though in our compiler usage we might just print and exit).
/// Actually, to simplify memory management for the JIT:
/// We will keep the error string in a thread-local or static buffer if possible, or leak it if we are going to exit anyway.
/// But cleaner is to return a pointer that the caller *should* free, but typically won't if stopping.
#[repr(C)]
pub struct CTensorResult {
    pub tensor: *mut OpaqueTensor,
    pub error: *const c_char,
}

impl CTensorResult {
    pub fn ok(tensor: *mut OpaqueTensor) -> Self {
        CTensorResult {
            tensor,
            error: ptr::null(),
        }
    }

    pub fn err(e: RuntimeError) -> Self {
        let c_str =
            CString::new(e.to_string()).unwrap_or_else(|_| CString::new("Unknown Error").unwrap());
        // We leak the string so it persists after function return.
        // In a long-running application, this would be a leak if handled and continued.
        // Ideally we should have a `tl_free_error(msg)` function.
        // For now, since errors are likely fatal or rare, leaking is acceptable for simplicity in FFI.
        CTensorResult {
            tensor: ptr::null_mut(),
            error: c_str.into_raw(),
        }
    }
}

// Convert Result to CTensorResult
impl From<Result<*mut OpaqueTensor, RuntimeError>> for CTensorResult {
    fn from(res: Result<*mut OpaqueTensor, RuntimeError>) -> Self {
        match res {
            Ok(t) => CTensorResult::ok(t),
            Err(e) => CTensorResult::err(e),
        }
    }
}
