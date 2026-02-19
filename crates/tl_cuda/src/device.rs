//! CUDA デバイス管理

use std::sync::Arc;

/// CUDA デバイスを管理
pub struct CudaDevice {
    // TODO: CUDA デバイスハンドル
}

impl CudaDevice {
    /// システムのデフォルト CUDA デバイスを取得
    pub fn new() -> Option<Self> {
        unimplemented!("CudaDevice::new")
    }

    /// デバイス名
    pub fn name(&self) -> &str {
        unimplemented!("CudaDevice::name")
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
