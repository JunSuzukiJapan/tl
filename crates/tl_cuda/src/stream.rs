//! CUDA ストリーム管理
//!
//! Metal の command_stream.rs に対応する CUDA ストリーム管理モジュール。

use crate::cuda_sys::{self, cudaStream_t, CUDA_SUCCESS};
use std::sync::OnceLock;

/// CUDA ストリームのラッパー
pub struct CudaStreamWrapper {
    stream: cudaStream_t,
}

unsafe impl Send for CudaStreamWrapper {}
unsafe impl Sync for CudaStreamWrapper {}

impl CudaStreamWrapper {
    /// デフォルトストリーム（NULL ストリーム）を使用
    pub fn new_default() -> Self {
        CudaStreamWrapper {
            stream: std::ptr::null_mut(),
        }
    }

    /// 新しいストリームを作成
    pub fn new() -> Result<Self, String> {
        let mut stream: cudaStream_t = std::ptr::null_mut();
        let err = unsafe { cuda_sys::cudaStreamCreate(&mut stream) };
        if err != CUDA_SUCCESS {
            return Err(format!("cudaStreamCreate failed: {}", err));
        }
        Ok(CudaStreamWrapper { stream })
    }

    /// ストリームハンドルを取得
    pub fn raw(&self) -> cudaStream_t {
        self.stream
    }

    /// ストリームの完了を待機
    pub fn synchronize(&self) -> Result<(), String> {
        if self.stream.is_null() {
            // デフォルトストリーム → cudaDeviceSynchronize
            cuda_check!(cuda_sys::cudaDeviceSynchronize())
        } else {
            cuda_check!(cuda_sys::cudaStreamSynchronize(self.stream))
        }
    }
}

impl Drop for CudaStreamWrapper {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                cuda_sys::cudaStreamDestroy(self.stream);
            }
        }
    }
}

/// グローバルストリームインスタンス
static GLOBAL_STREAM: OnceLock<CudaStreamWrapper> = OnceLock::new();

/// グローバルストリームを取得
pub fn get_stream() -> &'static CudaStreamWrapper {
    GLOBAL_STREAM.get_or_init(|| CudaStreamWrapper::new_default())
}

/// GPU 演算の完了を同期
pub fn sync_stream() {
    if let Err(e) = get_stream().synchronize() {
        eprintln!("CUDA sync_stream error: {}", e);
    }
}
