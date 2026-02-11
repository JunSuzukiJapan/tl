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

// Global memory tracker
static TOTAL_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn track_alloc(bytes: usize) {
    TOTAL_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn track_free(bytes: usize) {
    TOTAL_ALLOCATED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

pub fn get_total_allocated() -> usize {
    TOTAL_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

thread_local! {
    // Stack of scopes. Each scope contains a list of tensors allocated within it.
    // スコープスタック: 各スコープはそのスコープ内で割り当てられた Arc の生ポインタを保持。
    // スコープ脱出時に codegen が tl_tensor_release_safe を個別に呼ぶ。
    static SCOPE_STACK: std::cell::RefCell<Vec<Vec<*mut CpuTensor>>> = const { std::cell::RefCell::new(Vec::new()) };
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

pub fn register_tensor(t: *mut CpuTensor) {
    if t.is_null() { return; }
    SCOPE_STACK.with(|stack| {
        let mut stack_ref = stack.borrow_mut();
        if let Some(current_scope) = stack_ref.last_mut() {
            current_scope.push(t);
        }
    });
}

pub fn promote_tensor(t: *mut CpuTensor) {
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
pub fn release_tensor(t: *mut CpuTensor) {
    if t.is_null() { return; }
    unsafe {
        let arc_ref = Arc::from_raw(t as *const UnsafeCell<CpuTensor>);
        if is_mem_log_enabled() {
            let rc = Arc::strong_count(&arc_ref);
            eprintln!("[RELEASE] Ptr: {:p} (RC={}, {})", t, rc, if rc == 1 { "DROP" } else { "RC-1" });
        }
        drop(arc_ref);
    }
}
