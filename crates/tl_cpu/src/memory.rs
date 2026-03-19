use std::cell::UnsafeCell;
use std::sync::Arc;
use crate::tensor::CpuTensor;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Once;

// メモリログフラグ (TL_MEM_LOG 環境変数で有効化、--mem_log CLI フラグ経由)
static MEM_LOG_ENABLED: AtomicBool = AtomicBool::new(false);
static MEM_LOG_INIT: Once = Once::new();

pub fn is_mem_log_enabled() -> bool {
    MEM_LOG_INIT.call_once(|| {
        if std::env::var("TL_MEM_LOG").is_ok() {
            MEM_LOG_ENABLED.store(true, Ordering::Relaxed);
        }
    });
    MEM_LOG_ENABLED.load(Ordering::Relaxed)
}

// ========== メモリ統計カウンタ (TL_MEM_STATS=1 で有効化) ==========

static MEM_STATS_ENABLED: AtomicBool = AtomicBool::new(false);
static MEM_STATS_INIT: Once = Once::new();

/// テンソル割り当て回数
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// テンソル解放回数
static RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);
/// 累計割り当てバイト数
static TOTAL_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
/// 累計解放バイト数
static TOTAL_RELEASED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn is_mem_stats_enabled() -> bool {
    MEM_STATS_INIT.call_once(|| {
        if std::env::var("TL_MEM_STATS").is_ok() {
            MEM_STATS_ENABLED.store(true, Ordering::Relaxed);
        }
    });
    MEM_STATS_ENABLED.load(Ordering::Relaxed)
}

pub fn track_alloc(bytes: usize) {
    TOTAL_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn track_free(bytes: usize) {
    TOTAL_RELEASED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// テンソル割り当て1回をカウント (make_tensor から呼ぶ)
pub fn count_tensor_alloc() {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// テンソル解放1回をカウント (release_tensor の最終 drop 時に呼ぶ)
pub fn count_tensor_release() {
    RELEASE_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub fn get_total_allocated() -> usize {
    TOTAL_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

pub fn get_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

pub fn get_release_count() -> usize {
    RELEASE_COUNT.load(Ordering::Relaxed)
}

pub fn get_live_count() -> usize {
    let alloc = ALLOC_COUNT.load(Ordering::Relaxed);
    let release = RELEASE_COUNT.load(Ordering::Relaxed);
    alloc.saturating_sub(release)
}

pub fn get_total_released() -> usize {
    TOTAL_RELEASED_BYTES.load(Ordering::Relaxed)
}

/// メモリ統計レポートを stderr に出力
pub fn mem_stats_report() {
    let alloc_c = ALLOC_COUNT.load(Ordering::Relaxed);
    let release_c = RELEASE_COUNT.load(Ordering::Relaxed);
    let live = alloc_c.saturating_sub(release_c);
    let alloc_mb = TOTAL_ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let release_mb = TOTAL_RELEASED_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let rss_mb = get_rss_bytes() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[MEM] alloc={} release={} live={} alloc_mb={:.1} release_mb={:.1} rss_mb={:.1}",
        alloc_c, release_c, live, alloc_mb, release_mb, rss_mb
    );
}

/// プロセスの RSS (Resident Set Size) をバイト単位で取得
fn get_rss_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: i32,
                task_info_out: *mut std::ffi::c_void,
                task_info_outCnt: *mut u32,
            ) -> i32;
        }
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
        let kr = unsafe {
            task_info(
                mach_task_self(),
                20, // MACH_TASK_BASIC_INFO
                &mut info as *mut _ as *mut std::ffi::c_void,
                &mut count,
            )
        };
        if kr == 0 {
            info.resident_size as usize
        } else {
            0
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(resident) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = resident.parse::<usize>() {
                    return pages * 4096;
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

// ========== スコープ管理 (V6: 廃止 → No-op) ==========
// V5 では CPU 固有の SCOPE_STACK で全テンソルを追跡していたが、
// Metal/CUDA と統一するため廃止。テンソル寿命は Arc RC のみで管理。
// codegen の emit_cleanup_vars_in_scope → tl_tensor_release_safe が唯一の解放パス。

pub fn enter_scope() {
    // No-op: Metal/CUDA と統一
}

pub fn exit_scope() {
    // No-op: Metal/CUDA と統一
}

pub fn register_tensor(_t: *mut CpuTensor<f32>) {
    // No-op: Metal/CUDA と統一。make_tensor でのスコープ登録を廃止。
}

pub fn promote_tensor(_t: *mut CpuTensor<f32>) {
    // No-op: Metal/CUDA と統一
}

// Diagnostics
pub fn get_pool_size() -> usize {
    0
}

/// Arc ベースでテンソルを解放する (RC-1)。
/// RC が 0 になれば CpuTensor（autograd グラフ含む）が自然に Drop される。
/// Metal/CUDA の release_if_live と同等の処理。
pub fn release_tensor(t: *mut CpuTensor<f32>) {
    if t.is_null() { return; }
    unsafe {
        let arc_ref = Arc::from_raw(t as *const UnsafeCell<CpuTensor<f32>>);
        if is_mem_log_enabled() {
            let rc = Arc::strong_count(&arc_ref);
            eprintln!("[RELEASE] Ptr: {:p} (RC={}, {})", t, rc, if rc == 1 { "DROP" } else { "RC-1" });
        }
        if Arc::strong_count(&arc_ref) == 1 {
            count_tensor_release();
            // track_free は CpuTensor::Drop で呼ばれるため、ここでは呼ばない。
        }
        drop(arc_ref);
    }
}

