//! GPU バッファプール (CUDA)
//!
//! Metal の MetalBufferPool と同じパターン。サイズベースでバッファを再利用。

use crate::tensor::CudaBuffer;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

/// CUDA バッファプール
pub struct CudaBufferPool {
    /// フリーリスト: 再利用可能なバッファ（キーはバイトサイズ）
    pub free_buffers: HashMap<usize, Vec<Arc<CudaBuffer>>>,
    /// 統計: 取得成功数
    pub hits: u64,
    /// 統計: 新規確保数
    pub misses: u64,
}

impl CudaBufferPool {
    pub fn new() -> Self {
        CudaBufferPool {
            free_buffers: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// プールからバッファを取得（なければ None）
    pub fn acquire(&mut self, size: usize) -> Option<Arc<CudaBuffer>> {
        if let Some(list) = self.free_buffers.get_mut(&size) {
            if let Some(buffer) = list.pop() {
                self.hits += 1;
                return Some(buffer);
            }
        }
        self.misses += 1;
        None
    }

    /// バッファをプールに返却
    pub fn release(&mut self, buffer: Arc<CudaBuffer>) {
        const MAX_BUFFERS_PER_SIZE: usize = 4;

        let size = buffer.size();
        let list = self.free_buffers.entry(size).or_default();
        if list.len() < MAX_BUFFERS_PER_SIZE {
            list.push(buffer);
        }
        // else: buffer は drop → cudaFree
    }

    /// プール内のバッファ数
    pub fn free_count(&self) -> usize {
        self.free_buffers.values().map(|v| v.len()).sum()
    }

    /// 命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// 統計をダンプ
    pub fn dump_stats(&self) {
        eprintln!("=== CudaBufferPool Stats ===");
        eprintln!("Hits: {}, Misses: {}", self.hits, self.misses);
        eprintln!("Hit rate: {:.2}%", self.hit_rate() * 100.0);
        eprintln!("Free buffers: {}", self.free_count());
        eprintln!("============================");
    }
}

impl Default for CudaBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// グローバルバッファプール
pub static BUFFER_POOL: LazyLock<Mutex<CudaBufferPool>> =
    LazyLock::new(|| Mutex::new(CudaBufferPool::new()));

/// プールからバッファを取得
pub fn pool_acquire(size: usize) -> Option<Arc<CudaBuffer>> {
    // 再利用するバッファが前のGPU操作で使われていた可能性があるので同期
    crate::stream::sync_stream();
    BUFFER_POOL.lock().ok()?.acquire(size)
}

/// バッファをプールに返却
pub fn pool_release(buffer: Arc<CudaBuffer>) {
    if let Ok(mut pool) = BUFFER_POOL.lock() {
        pool.release(buffer);
    }
}
