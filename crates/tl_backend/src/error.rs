//! エラー定義

use std::fmt;

/// バックエンドエラー
#[derive(Debug)]
pub enum BackendError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    DTypeMismatch { expected: String, got: String },
    DeviceError(String),
    UnsupportedOperation(String),
    OutOfMemory,
    Other(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            BackendError::DTypeMismatch { expected, got } => {
                write!(f, "DType mismatch: expected {}, got {}", expected, got)
            }
            BackendError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            BackendError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            BackendError::OutOfMemory => write!(f, "Out of GPU memory"),
            BackendError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

pub type Result<T> = std::result::Result<T, BackendError>;
