//! GPU バッファプール
//!
//! Metal バッファを再利用し、メモリ割り当てのオーバーヘッドを削減する。

use metal::{Buffer, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

/// バッファプールのキー: (サイズ, リソースオプション)
type PoolKey = (usize, MTLResourceOptions);

/// Metal バッファプール
pub struct MetalBufferPool {
    /// フリーリスト: 再利用可能なバッファ
    free_buffers: HashMap<PoolKey, Vec<Arc<Buffer>>>,
    /// 統計: 取得成功数
    pub hits: u64,
    /// 統計: 新規確保数
    pub misses: u64,
}

impl MetalBufferPool {
    pub fn new() -> Self {
        MetalBufferPool {
            free_buffers: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// プールからバッファを取得（なければ None）
    pub fn acquire(&mut self, size: usize, options: MTLResourceOptions) -> Option<Arc<Buffer>> {
        let key = (size, options);
        if let Some(list) = self.free_buffers.get_mut(&key) {
            if let Some(buffer) = list.pop() {
                self.hits += 1;
                return Some(buffer);
            }
        }
        self.misses += 1;
        None
    }

    /// バッファをプールに返却
    pub fn release(&mut self, buffer: Arc<Buffer>) {
        let size = buffer.length() as usize;
        // Note: MTLResourceOptions は Buffer から直接取得できないため、
        // 現時点では StorageModeShared を仮定
        let options = MTLResourceOptions::StorageModeShared;
        let key = (size, options);
        self.free_buffers.entry(key).or_default().push(buffer);
    }

    /// プール内のバッファ数
    pub fn free_count(&self) -> usize {
        self.free_buffers.values().map(|v| v.len()).sum()
    }

    /// 命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// 統計をダンプ
    pub fn dump_stats(&self) {
        eprintln!("=== MetalBufferPool Stats ===");
        eprintln!("Hits: {}, Misses: {}", self.hits, self.misses);
        eprintln!("Hit rate: {:.2}%", self.hit_rate() * 100.0);
        eprintln!("Free buffers: {}", self.free_count());
        eprintln!("=============================");
    }
}

impl Default for MetalBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// グローバルバッファプール
pub static BUFFER_POOL: LazyLock<Mutex<MetalBufferPool>> =
    LazyLock::new(|| Mutex::new(MetalBufferPool::new()));

/// プールからバッファを取得
pub fn pool_acquire(size: usize, options: MTLResourceOptions) -> Option<Arc<Buffer>> {
    BUFFER_POOL.lock().ok()?.acquire(size, options)
}

/// バッファをプールに返却
pub fn pool_release(buffer: Arc<Buffer>) {
    if let Ok(mut pool) = BUFFER_POOL.lock() {
        pool.release(buffer);
    }
}
