//! CUDA デバイス管理

use crate::cuda_sys::{self, cudaDeviceProp, CUDA_SUCCESS};
use std::sync::Arc;

/// CUDA デバイスを管理
pub struct CudaDevice {
    device_id: i32,
    device_name: String,
}

impl CudaDevice {
    /// システムのデフォルト CUDA デバイスを取得
    pub fn new() -> Option<Self> {
        let mut count: i32 = 0;
        let err = unsafe { cuda_sys::cudaGetDeviceCount(&mut count) };
        if err != CUDA_SUCCESS || count == 0 {
            return None;
        }

        // デバイス 0 を使用
        let device_id = 0;
        let err = unsafe { cuda_sys::cudaSetDevice(device_id) };
        if err != CUDA_SUCCESS {
            return None;
        }

        // デバイス名を取得
        let mut prop = cudaDeviceProp::default();
        let err = unsafe { cuda_sys::cudaGetDeviceProperties(&mut prop, device_id) };
        let device_name = if err == CUDA_SUCCESS {
            let name_bytes = &prop.name;
            let len = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            String::from_utf8_lossy(&name_bytes[..len]).to_string()
        } else {
            "Unknown CUDA Device".to_string()
        };

        Some(CudaDevice {
            device_id,
            device_name,
        })
    }

    /// デバイス ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// デバイス名
    pub fn name(&self) -> &str {
        &self.device_name
    }

    /// GPU メモリを確保
    pub fn allocate_buffer(&self, size: usize) -> Result<*mut std::ffi::c_void, String> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // 0バイトの場合でも最低1バイト確保（cudaMalloc(0) の挙動回避）
        let alloc_size = if size == 0 { 1 } else { size };
        let err = unsafe { cuda_sys::cudaMalloc(&mut ptr, alloc_size) };
        if err != CUDA_SUCCESS {
            let msg = crate::cuda_sys::check_cuda(err).unwrap_err();
            return Err(format!("cudaMalloc({} bytes) failed: {}", alloc_size, msg));
        }
        Ok(ptr)
    }
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new().expect("No CUDA device available")
    }
}

/// グローバルデバイスインスタンス
static GLOBAL_DEVICE: std::sync::OnceLock<Arc<CudaDevice>> = std::sync::OnceLock::new();

/// グローバルデバイスを取得
pub fn get_device() -> Arc<CudaDevice> {
    GLOBAL_DEVICE
        .get_or_init(|| Arc::new(CudaDevice::default()))
        .clone()
}

/// グローバルデバイスが初期化済みの場合のみ取得
pub fn try_get_device() -> Option<Arc<CudaDevice>> {
    GLOBAL_DEVICE.get().cloned()
}
