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

// ========== スコープ管理 ==========

thread_local! {
    // Stack of scopes. Each scope contains a list of tensors allocated within it.
    // スコープスタック: 各スコープはそのスコープ内で割り当てられた Arc の生ポインタを保持。
    // スコープ脱出時に codegen が tl_tensor_release_safe を個別に呼ぶ。
    static SCOPE_STACK: std::cell::RefCell<Vec<Vec<*mut CpuTensor<f32>>>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub fn enter_scope() {
    SCOPE_STACK.with(|stack| {
        stack.borrow_mut().push(Vec::new());
    });
}

pub fn exit_scope() {
    // スコープスタックをポップするのみ。
    // テンソルの解放は codegen の emit_cleanup_vars_in_scope → tl_tensor_release_safe が個別に行う。
    SCOPE_STACK.with(|stack| {
        let _ = stack.borrow_mut().pop();
    });
}

pub fn register_tensor(t: *mut CpuTensor<f32>) {
    if t.is_null() { return; }
    SCOPE_STACK.with(|stack| {
        let mut stack_ref = stack.borrow_mut();
        if let Some(current_scope) = stack_ref.last_mut() {
            current_scope.push(t);
        }
    });
}

/// スコープスタックからテンソルを除去（release_tensor から呼ばれる）
fn unregister_from_scope(t: *mut CpuTensor<f32>) {
    SCOPE_STACK.with(|stack| {
        let mut stack_ref = stack.borrow_mut();
        // 最も内側のスコープから探して除去
        for scope in stack_ref.iter_mut().rev() {
            if let Some(pos) = scope.iter().rposition(|&x| x == t) {
                scope.remove(pos);
                return;
            }
        }
    });
}

pub fn promote_tensor(t: *mut CpuTensor<f32>) {
    if t.is_null() { return; }
    SCOPE_STACK.with(|stack| {
        let mut stack_ref = stack.borrow_mut();
        if let Some(current_scope) = stack_ref.last_mut() {
             if let Some(pos) = current_scope.iter().rposition(|&x| x == t) {
                 current_scope.remove(pos);
             }
        }
    });
}

// Diagnostics
pub fn get_pool_size() -> usize {
    0  // プールは Arc 化で廃止（Arc の参照カウントが管理）
}

/// Arc ベースでテンソルを解放する (RC-1)。
/// RC が 0 になれば CpuTensor（autograd グラフ含む）が自然に Drop される。
pub fn release_tensor(t: *mut CpuTensor<f32>) {
    if t.is_null() { return; }
    // スコープスタックからこのテンソルを除去（exit_scope での二重解放防止）
    unregister_from_scope(t);
    unsafe {
        let arc_ref = Arc::from_raw(t as *const UnsafeCell<CpuTensor<f32>>);
        if is_mem_log_enabled() {
            let rc = Arc::strong_count(&arc_ref);
            eprintln!("[RELEASE] Ptr: {:p} (RC={}, {})", t, rc, if rc == 1 { "DROP" } else { "RC-1" });
        }
        // track_free は CpuTensor の Drop 内で呼ばれるべきだが、
        // ここで RC==1 (最終 drop) の場合にカウント
        if Arc::strong_count(&arc_ref) == 1 {
            count_tensor_release();
            // UnsafeCell::get() は *mut CpuTensor<f32> を返す
            let tensor_ptr = arc_ref.get();
            let data_len = unsafe { (*tensor_ptr).data.capacity() };
            let bytes = data_len * std::mem::size_of::<f32>();
            if bytes > 0 {
                track_free(bytes);
            }
        }
        drop(arc_ref);
    }
}
