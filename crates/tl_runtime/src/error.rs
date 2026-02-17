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

impl From<u32> for RuntimeErrorCode {
    fn from(code: u32) -> Self {
        match code {
            0 => RuntimeErrorCode::Success,
            1 => RuntimeErrorCode::ShapeMismatch,
            2 => RuntimeErrorCode::DeviceError,
            3 => RuntimeErrorCode::AllocationError,
            4 => RuntimeErrorCode::NullPointerError,
            5 => RuntimeErrorCode::InternalError,
            6 => RuntimeErrorCode::IndexOutOfBounds,
            7 => RuntimeErrorCode::TypeMismatch,
            8 => RuntimeErrorCode::ArgumentError,
            _ => RuntimeErrorCode::Unknown,
        }
    }
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

impl From<tl_backend::BackendError> for RuntimeError {
    fn from(e: tl_backend::BackendError) -> Self {
        match e {
            tl_backend::BackendError::ShapeMismatch(msg) => RuntimeError::ShapeMismatch(msg),
            tl_backend::BackendError::DeviceError(msg) => RuntimeError::DeviceError(msg),
            tl_backend::BackendError::AllocationError(msg) => RuntimeError::AllocationError(msg),
            tl_backend::BackendError::NullPointerError(msg) => RuntimeError::NullPointerError(msg),
            tl_backend::BackendError::InternalError(msg) => RuntimeError::InternalError(msg),
            tl_backend::BackendError::IndexOutOfBounds(msg) => RuntimeError::IndexOutOfBounds(msg),
            tl_backend::BackendError::TypeMismatch(msg) => RuntimeError::TypeMismatch(msg),
            tl_backend::BackendError::ArgumentError(msg) => RuntimeError::ArgumentError(msg),
            tl_backend::BackendError::Unknown(msg) => RuntimeError::Unknown(msg),
        }
    }
}

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

pub fn set_backend_error(e: tl_backend::BackendError) {
    let runtime_err: RuntimeError = e.into();
    set_last_error(runtime_err.to_string(), runtime_err.code());
}

#[unsafe(no_mangle)]
/// @ffi_sig () -> CTensorResult
pub extern "C" fn tl_get_last_error() -> CTensorResult {
    LAST_ERROR.with(|e| {
        if let Some(err) = e.borrow().as_ref() {
            let c_msg = CString::new(err.message.clone()).unwrap();
            let c_file = CString::new(err.file.clone()).unwrap();

            CTensorResult {
                tensor: ptr::null_mut(),
                error_msg: c_msg.into_raw(),
                error_code: err.code,
                file: c_file.into_raw(),
                line: err.line,
                col: err.col,
            }
        } else {
            CTensorResult::ok(ptr::null_mut())
        }
    })
}

#[unsafe(no_mangle)]
/// @ffi_sig (i8*, i32, i32) -> void
pub extern "C" fn tl_report_runtime_error_loc(
    file: *const c_char,
    line: i32,
    col: i32,
) {
    let file_str = if file.is_null() {
        "unknown".to_string()
    } else {
        unsafe {
            std::ffi::CStr::from_ptr(file)
                .to_string_lossy()
                .into_owned()
        }
    };

    LAST_ERROR.with(|e| {
        let borrow = e.borrow();
        if let Some(err) = borrow.as_ref() {
             eprintln!("\n[Runtime Error] Code: {:?} ({})", err.code, err.code as u32);
             eprintln!("  Message: {}", err.message);
             eprintln!("  Location: {}:{}:{}", file_str, line, col);
        } else {
             eprintln!("\n[Runtime Error] Unknown Error (Null Pointer Returned)");
             eprintln!("  Location: {}:{}:{}", file_str, line, col);
        }
    });
    std::process::exit(1);
}
