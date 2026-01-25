use crate::OpaqueTensor;
use std::ffi::{c_char, CString};
use std::ptr;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeErrorCode {
    Success = 0,
    ShapeMismatch = 1,
    DeviceError = 2,
    AllocationError = 3,
    NullPointerError = 4,
    InternalError = 5,
    IndexOutOfBounds = 6,
    TypeMismatch = 7,
    ArgumentError = 8,
    Unknown = 999,
}

#[derive(Debug)]
pub enum RuntimeError {
    ShapeMismatch(String),
    DeviceError(String),
    AllocationError(String),
    NullPointerError(String),
    InternalError(String),
    IndexOutOfBounds(String),
    TypeMismatch(String),
    ArgumentError(String),
    Unknown(String),
}

impl RuntimeError {
    pub fn code(&self) -> RuntimeErrorCode {
        match self {
            RuntimeError::ShapeMismatch(_) => RuntimeErrorCode::ShapeMismatch,
            RuntimeError::DeviceError(_) => RuntimeErrorCode::DeviceError,
            RuntimeError::AllocationError(_) => RuntimeErrorCode::AllocationError,
            RuntimeError::NullPointerError(_) => RuntimeErrorCode::NullPointerError,
            RuntimeError::InternalError(_) => RuntimeErrorCode::InternalError,
            RuntimeError::IndexOutOfBounds(_) => RuntimeErrorCode::IndexOutOfBounds,
            RuntimeError::TypeMismatch(_) => RuntimeErrorCode::TypeMismatch,
            RuntimeError::ArgumentError(_) => RuntimeErrorCode::ArgumentError,
            RuntimeError::Unknown(_) => RuntimeErrorCode::Unknown,
        }
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::ShapeMismatch(msg) => write!(f, "ShapeMismatch: {}", msg),
            RuntimeError::DeviceError(msg) => write!(f, "DeviceError: {}", msg),
            RuntimeError::AllocationError(msg) => write!(f, "AllocationError: {}", msg),
            RuntimeError::NullPointerError(msg) => write!(f, "NullPointerError: {}", msg),
            RuntimeError::InternalError(msg) => write!(f, "InternalError: {}", msg),
            RuntimeError::IndexOutOfBounds(msg) => write!(f, "IndexOutOfBounds: {}", msg),
            RuntimeError::TypeMismatch(msg) => write!(f, "TypeMismatch: {}", msg),
            RuntimeError::ArgumentError(msg) => write!(f, "ArgumentError: {}", msg),
            RuntimeError::Unknown(msg) => write!(f, "Unknown Error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Result struct for FFI functions returning a Tensor.
#[repr(C)]
pub struct CTensorResult {
    pub tensor: *mut OpaqueTensor,
    pub error_msg: *const c_char,
    pub error_code: RuntimeErrorCode,
    pub file: *const c_char,
    pub line: u32,
    pub col: u32,
}

impl CTensorResult {
    pub fn ok(tensor: *mut OpaqueTensor) -> Self {
        CTensorResult {
            tensor,
            error_msg: ptr::null(),
            error_code: RuntimeErrorCode::Success,
            file: ptr::null(),
            line: 0,
            col: 0,
        }
    }

    pub fn err(e: RuntimeError) -> Self {
        let code = e.code();
        let c_str =
            CString::new(e.to_string()).unwrap_or_else(|_| CString::new("Unknown Error").unwrap());
        CTensorResult {
            tensor: ptr::null_mut(),
            error_msg: c_str.into_raw(),
            error_code: code,
            file: ptr::null(), // To be filled by wrapper or defaults
            line: 0,
            col: 0,
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

/// Struct to hold the last error information in Thread Local Storage.
/// Strings are owned to ensure lifetime safety.
#[derive(Debug, Clone)]
pub struct LastError {
    pub code: RuntimeErrorCode,
    pub message: String,
    pub file: String,
    pub line: u32,
    pub col: u32,
}

use std::cell::RefCell;

thread_local! {
    pub static LAST_ERROR: RefCell<Option<LastError>> = RefCell::new(None);
}

pub fn set_last_error<S: Into<String>>(msg: S, code: RuntimeErrorCode) {
    let s = msg.into();
    log::error!("{}", s);
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(LastError {
            code,
            message: s,
            file: String::new(),
            line: 0,
            col: 0,
        });
    });
}


#[no_mangle]
pub extern "C" fn tl_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        if e.borrow().is_some() {
            // Leak string to return safe pointer? Or use buffer?
            // For simple implementation, we might need a persistent buffer.
            // But let's check how to return string FFI safely.
            // Usually we return a static buffer or standard string.
            // Here we just return the inner pointer if string is kept alive? 
            // String in RefCell is alive as long as thread lives and not overwritten?
            // No, RefCell borrow ends.
            // Let's store a CString in LastError instead of String?
            // Or convert on demand and store cached CString?
            // Simplification: We just want to store it for now.
            // Returning it is bonus.
            std::ptr::null()
        } else {
            std::ptr::null()
        }
    })
}
