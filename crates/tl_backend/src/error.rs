//! エラー定義

use std::fmt;

/// バックエンドエラー
#[derive(Debug)]
pub enum BackendError {
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

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::ShapeMismatch(msg) => write!(f, "ShapeMismatch: {}", msg),
            BackendError::DeviceError(msg) => write!(f, "DeviceError: {}", msg),
            BackendError::AllocationError(msg) => write!(f, "AllocationError: {}", msg),
            BackendError::NullPointerError(msg) => write!(f, "NullPointerError: {}", msg),
            BackendError::InternalError(msg) => write!(f, "InternalError: {}", msg),
            BackendError::IndexOutOfBounds(msg) => write!(f, "IndexOutOfBounds: {}", msg),
            BackendError::TypeMismatch(msg) => write!(f, "TypeMismatch: {}", msg),
            BackendError::ArgumentError(msg) => write!(f, "ArgumentError: {}", msg),
            BackendError::Unknown(msg) => write!(f, "Unknown Error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

pub type Result<T> = std::result::Result<T, BackendError>;
