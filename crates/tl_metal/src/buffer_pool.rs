//! GPU バッファプール
//!
//! Metal バッファを再利用し、メモリ割り当てのオーバーヘッドを削減する。

use metal::{Buffer, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, LazyLock};

/// バッファプールのキー: サイズのみ
/// Metal の resource_options() は作成時に指定した MTLResourceOptions と
/// 異なるビットフラグを返す場合がある (ドライバ内部の追加フラグ)。
/// そのため size のみをキーとして使用し、options 不一致による miss を防ぐ。
type PoolKey = usize;

/// Metal バッファプール
pub struct MetalBufferPool {
    /// フリーリスト: 再利用可能なバッファ
    pub free_buffers: HashMap<PoolKey, Vec<Arc<Buffer>>>,
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
    pub fn acquire(&mut self, size: usize, _options: MTLResourceOptions) -> Option<Arc<Buffer>> {
        let key = size;
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
    /// 各サイズバケットの最大保持数を超えた場合、バッファは drop されて
    /// OS にメモリが返される（GPU メモリリーク防止）。
    pub fn release(&mut self, buffer: Arc<Buffer>) {
        let size = buffer.length() as usize;
        let key = size;
        let list = self.free_buffers.entry(key).or_default();
        
        // Persistent GPU Pool 戦略: 
        // ただしサイズバケットあたり最大 32 バッファに制限し、無限蓄積を防止
        if list.len() < 32 {
            list.push(buffer);
        }
        // 32 を超えた場合は drop して OS に返す
    }

    /// プール内のバッファを強制解放する（OSにメモリを返却）
    pub fn purge(&mut self) {
        self.free_buffers.clear();
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
/// バッファプールからの再利用前にGPUストリームを同期し、
/// 前の操作が完了していないバッファの再利用を防ぐ。
pub fn pool_acquire(size: usize, options: MTLResourceOptions) -> Option<Arc<Buffer>> {
    // 再利用するバッファが前のGPU操作で使われていた可能性があるので、
    // ストリームに未コミットのコマンドがあれば同期する
    if crate::command_stream::get_stream().needs_sync() {
        crate::command_stream::sync_stream();
    }
    BUFFER_POOL.lock().ok()?.acquire(size, options)
}

/// バッファをプールに返却
pub fn pool_release(buffer: Arc<Buffer>) {
    if let Ok(mut pool) = BUFFER_POOL.lock() {
        pool.release(buffer);
    }
}

/// プール内のバッファを強制解放する（OSにメモリを返却）
pub fn pool_purge() {
    if let Ok(mut pool) = BUFFER_POOL.lock() {
        pool.purge();
    }
}
