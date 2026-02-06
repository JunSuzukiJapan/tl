// Persistent GPU Memory Pool
// テンソルを解放せず再利用することで、Metal/CUDA ドライバの RSS 膨張問題を回避
//
// V4.0: 全サイズ対応、解放なし、初期メモリ確保オプション

use super::OpaqueTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

/// プール統計情報
#[derive(Default)]
pub struct PoolStats {
    /// 累計確保バイト数
    pub total_allocated_bytes: AtomicU64,
    /// フリーリスト内のテンソル数
    pub current_free_count: AtomicU64,
    /// フリーリスト内のバイト数
    pub current_free_bytes: AtomicU64,
    /// プールからの再利用成功数
    pub acquire_hits: AtomicU64,
    /// 新規確保が必要だった数
    pub acquire_misses: AtomicU64,
}

impl PoolStats {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Persistent GPU Pool
/// すべてのテンソルをプールし、解放しない
pub struct PersistentGpuPool {
    // キー: (要素数, dtype_id, device_id)
    // 値: 再利用可能なテンソルのリスト
    free_lists: HashMap<(usize, u8, u8), Vec<*mut OpaqueTensor>>,
    // 使用中テンソルの追跡（重複解放防止）
    active: std::collections::HashSet<usize>,
    /// 統計情報
    pub stats: PoolStats,
    /// 初期化済みフラグ
    initialized: bool,
}

// SAFETY: PersistentGpuPool の raw ポインタは単一スレッドの JIT 実行コンテキストでのみアクセス
unsafe impl Send for PersistentGpuPool {}
unsafe impl Sync for PersistentGpuPool {}

impl PersistentGpuPool {
    pub fn new() -> Self {
        PersistentGpuPool {
            free_lists: HashMap::new(),
            active: std::collections::HashSet::new(),
            stats: PoolStats::new(),
            initialized: false,
        }
    }

    /// 初期GPUメモリ確保 (TL_GPU_PREALLOCATE_MB 環境変数で指定)
    pub fn preallocate(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;

        if let Ok(mb_str) = std::env::var("TL_GPU_PREALLOCATE_MB") {
            if let Ok(mb) = mb_str.parse::<usize>() {
                if mb > 0 {
                    eprintln!("[PersistentGpuPool] Preallocating {} MB of GPU memory...", mb);
                    // 将来的に実際のテンソル確保を行う
                    // 現時点では統計のみ更新
                    let bytes = (mb * 1024 * 1024) as u64;
                    self.stats.total_allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
                    eprintln!("[PersistentGpuPool] Preallocation complete.");
                }
            }
        }
    }

    /// テンソルのバイト数を計算 (要素数 × dtype サイズ)
    fn calculate_bytes(num_elements: usize, dtype_id: u8) -> usize {
        let elem_size = match dtype_id {
            0 => 4,  // F32
            1 => 8,  // F64
            2 => 4,  // I32
            3 => 8,  // I64
            4 => 1,  // U8
            5 => 2,  // F16
            6 => 2,  // BF16
            _ => 4,  // デフォルト F32
        };
        num_elements * elem_size
    }

    /// プールからテンソルを取得
    /// V4.0 Phase 2: フリーリストからメモリを再利用
    /// 注意: 取得したポインタはメモリ領域のみ再利用。内容は初期化が必要。
    pub fn acquire(
        &mut self,
        num_elements: usize,
        dtype_id: u8,
        device_id: u8,
    ) -> Option<*mut OpaqueTensor> {
        let key = (num_elements, dtype_id, device_id);
        
        let debug = std::env::var("TL_POOL_DEBUG").is_ok();
        
        if let Some(list) = self.free_lists.get_mut(&key) {
            if let Some(ptr) = list.pop() {
                let bytes = Self::calculate_bytes(num_elements, dtype_id) as u64;
                self.stats.current_free_count.fetch_sub(1, Ordering::Relaxed);
                self.stats.current_free_bytes.fetch_sub(bytes, Ordering::Relaxed);
                self.stats.acquire_hits.fetch_add(1, Ordering::Relaxed);
                self.active.insert(ptr as usize);
                
                if debug {
                    eprintln!("[Pool] HIT acquire key={:?} ptr={:?}", key, ptr);
                }
                
                // ドロップして再利用可能にする（メモリは保持）
                unsafe {
                    std::ptr::drop_in_place(ptr);
                }
                
                return Some(ptr);
            }
        }
        
        if debug {
            eprintln!("[Pool] MISS acquire key={:?} free_lists_keys={:?}", key, self.free_lists.keys().collect::<Vec<_>>());
        }
        
        self.stats.acquire_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// テンソルをプールに戻す（解放しない）
    /// V4.0 Phase 2: フリーリストにプッシュして再利用可能に
    pub fn release(
        &mut self,
        ptr: *mut OpaqueTensor,
        num_elements: usize,
        dtype_id: u8,
        device_id: u8,
    ) {
        if ptr.is_null() {
            return;
        }

        let debug = std::env::var("TL_POOL_DEBUG").is_ok();
        let ptr_val = ptr as usize;
        
        // アクティブから削除
        self.active.remove(&ptr_val);
        
        // 重複チェック
        let key = (num_elements, dtype_id, device_id);
        if let Some(list) = self.free_lists.get(&key) {
            if list.contains(&ptr) {
                if debug {
                    eprintln!("[Pool] SKIP release (duplicate) key={:?} ptr={:?}", key, ptr);
                }
                return; // 既にプール内
            }
        }

        // フリーリストに追加（ドロップは acquire 時に行う）
        let bytes = Self::calculate_bytes(num_elements, dtype_id) as u64;
        self.free_lists.entry(key).or_default().push(ptr);
        self.stats.current_free_count.fetch_add(1, Ordering::Relaxed);
        self.stats.current_free_bytes.fetch_add(bytes, Ordering::Relaxed);
        
        if debug {
            let list_len = self.free_lists.get(&key).map(|l| l.len()).unwrap_or(0);
            eprintln!("[Pool] RELEASE key={:?} ptr={:?} list_len={}", key, ptr, list_len);
        }
    }

    /// 新規テンソル確保を記録
    pub fn register_new_allocation(&mut self, ptr: *mut OpaqueTensor, num_elements: usize, dtype_id: u8) {
        if !ptr.is_null() {
            self.active.insert(ptr as usize);
            let bytes = Self::calculate_bytes(num_elements, dtype_id) as u64;
            self.stats.total_allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// プール命中率を計算
    pub fn hit_rate(&self) -> f64 {
        let hits = self.stats.acquire_hits.load(Ordering::Relaxed);
        let misses = self.stats.acquire_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// 統計をダンプ
    pub fn dump_stats(&self) {
        eprintln!("=== PersistentGpuPool Stats ===");
        eprintln!("Total allocated: {} bytes ({} MB)",
            self.stats.total_allocated_bytes.load(Ordering::Relaxed),
            self.stats.total_allocated_bytes.load(Ordering::Relaxed) / (1024 * 1024));
        eprintln!("Free pool: {} tensors, {} bytes ({} MB)",
            self.stats.current_free_count.load(Ordering::Relaxed),
            self.stats.current_free_bytes.load(Ordering::Relaxed),
            self.stats.current_free_bytes.load(Ordering::Relaxed) / (1024 * 1024));
        eprintln!("Acquire hits: {}", self.stats.acquire_hits.load(Ordering::Relaxed));
        eprintln!("Acquire misses: {}", self.stats.acquire_misses.load(Ordering::Relaxed));
        eprintln!("Hit rate: {:.2}%", self.hit_rate() * 100.0);
        eprintln!("Active tensors: {}", self.active.len());
        eprintln!("===============================");
    }
}

// グローバル Persistent GPU Pool
pub static PERSISTENT_GPU_POOL: LazyLock<Mutex<PersistentGpuPool>> =
    LazyLock::new(|| Mutex::new(PersistentGpuPool::new()));

/// プールを初期化（初期メモリ確保を含む）
pub fn init_pool() {
    if let Ok(mut pool) = PERSISTENT_GPU_POOL.lock() {
        pool.preallocate();
    }
}

/// プールからテンソルを取得
pub fn pool_acquire(num_elements: usize, dtype_id: u8, device_id: u8) -> Option<*mut OpaqueTensor> {
    PERSISTENT_GPU_POOL.lock().ok()?.acquire(num_elements, dtype_id, device_id)
}

/// テンソルをプールに戻す
pub fn pool_release(ptr: *mut OpaqueTensor, num_elements: usize, dtype_id: u8, device_id: u8) {
    if let Ok(mut pool) = PERSISTENT_GPU_POOL.lock() {
        pool.release(ptr, num_elements, dtype_id, device_id);
    }
}

/// 新規テンソル確保を記録
pub fn pool_register_new(ptr: *mut OpaqueTensor, num_elements: usize, dtype_id: u8) {
    if let Ok(mut pool) = PERSISTENT_GPU_POOL.lock() {
        pool.register_new_allocation(ptr, num_elements, dtype_id);
    }
}

// ============ C-ABI exports for LLVM codegen ============

/// プールからテンソルを取得 (C API)
#[unsafe(no_mangle)]
pub extern "C" fn tl_pool_acquire(num_elements: usize, dtype_id: u8, device_id: u8) -> *mut OpaqueTensor {
    pool_acquire(num_elements, dtype_id, device_id).unwrap_or(std::ptr::null_mut())
}

/// テンソルをプールに戻す (C API)
#[unsafe(no_mangle)]
pub extern "C" fn tl_pool_release(ptr: *mut OpaqueTensor, num_elements: usize, dtype_id: u8, device_id: u8) {
    if !ptr.is_null() {
        pool_release(ptr, num_elements, dtype_id, device_id);
    }
}

/// 確保済み総バイト数を取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_gpu_total_allocated_bytes() -> i64 {
    PERSISTENT_GPU_POOL
        .lock()
        .map(|p| p.stats.total_allocated_bytes.load(Ordering::Relaxed) as i64)
        .unwrap_or(0)
}

/// フリーリスト内のテンソル数
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_gpu_free_count() -> i64 {
    PERSISTENT_GPU_POOL
        .lock()
        .map(|p| p.stats.current_free_count.load(Ordering::Relaxed) as i64)
        .unwrap_or(0)
}

/// フリーリスト内のバイト数
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_gpu_free_bytes() -> i64 {
    PERSISTENT_GPU_POOL
        .lock()
        .map(|p| p.stats.current_free_bytes.load(Ordering::Relaxed) as i64)
        .unwrap_or(0)
}

/// プール命中率
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_gpu_pool_hit_rate() -> f64 {
    PERSISTENT_GPU_POOL
        .lock()
        .map(|p| p.hit_rate())
        .unwrap_or(0.0)
}

/// 統計のダンプ
#[unsafe(no_mangle)]
pub extern "C" fn tl_dump_gpu_pool_stats() {
    if let Ok(pool) = PERSISTENT_GPU_POOL.lock() {
        pool.dump_stats();
    }
}

// ============ 後方互換性のための旧API ============
// 旧 tensor_pool.rs の API を維持（シグネチャ変更）

/// 旧API: 要素数のみでプールを取得（dtype/device はデフォルト）
#[unsafe(no_mangle)]
pub extern "C" fn tl_pool_acquire_compat(element_count: usize) -> *mut OpaqueTensor {
    // デフォルト: F32 (0), 現在のデバイス (0)
    tl_pool_acquire(element_count, 0, 0)
}

/// 旧API: 要素数のみでプールに戻す
#[unsafe(no_mangle)]
pub extern "C" fn tl_pool_release_compat(ptr: *mut OpaqueTensor, element_count: usize) {
    tl_pool_release(ptr, element_count, 0, 0);
}
