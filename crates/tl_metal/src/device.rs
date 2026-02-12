//! Metal デバイス管理

use metal::{CommandQueue, Device, MTLResourceOptions};
use std::sync::Arc;

/// Metal デバイスとコマンドキューを管理
pub struct MetalDevice {
    device: Device,
    command_queue: CommandQueue,
}

impl MetalDevice {
    /// システムのデフォルト Metal デバイスを取得
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();
        Some(MetalDevice {
            device,
            command_queue,
        })
    }

    /// 内部デバイスへの参照
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// コマンドキューへの参照
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// バッファを確保
    pub fn allocate_buffer(&self, size: usize, options: MTLResourceOptions) -> metal::Buffer {
        self.device.new_buffer(size as u64, options)
    }

    /// デバイス名
    pub fn name(&self) -> &str {
        self.device.name()
    }
}

impl Default for MetalDevice {
    fn default() -> Self {
        Self::new().expect("No Metal device available")
    }
}

/// グローバルデバイスインスタンス
static GLOBAL_DEVICE: std::sync::OnceLock<Arc<MetalDevice>> = std::sync::OnceLock::new();

/// グローバルデバイスを取得
pub fn get_device() -> Arc<MetalDevice> {
    GLOBAL_DEVICE
        .get_or_init(|| Arc::new(MetalDevice::default()))
        .clone()
}

/// グローバルデバイスが初期化済みの場合のみ取得（初期化を引き起こさない）
pub fn try_get_device() -> Option<Arc<MetalDevice>> {
    GLOBAL_DEVICE.get().cloned()
}
