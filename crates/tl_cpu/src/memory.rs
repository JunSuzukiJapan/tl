use std::cell::RefCell;
use std::sync::Mutex;
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

// Global Tensor Pool to reduce allocation overhead
// Stores released tensors for reuse.
#[allow(clippy::vec_box)]
static TENSOR_POOL: Mutex<Vec<Box<CpuTensor>>> = Mutex::new(Vec::new());

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
    // When a scope exits, all tensors in it are returned to the global pool.
    #[allow(clippy::vec_box)]
    static SCOPE_STACK: RefCell<Vec<Vec<*mut CpuTensor>>> = const { RefCell::new(Vec::new()) };
}

pub fn enter_scope() {
    SCOPE_STACK.with(|stack| {
        stack.borrow_mut().push(Vec::new());
    });
}

pub fn exit_scope() {
    // スコープスタックをポップするのみ。
    // テンソルの解放は codegen の emit_cleanup_vars_in_scope → tl_tensor_release_safe が個別に行う。
    // ここで Box::from_raw するとcodegen 側の release_safe と二重解放になる。
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
        } else {
            // No scope active (e.g., global scope or top-level script without scope)
            // In this case, we might leak or need a default global scope.
            // For now, let's assume there's always a scope if `tl_mem_function_enter` is called.
            // If not, we could warn or just ignore (leak).
            // Better to leak than crash.
        }
    });
}

pub fn promote_tensor(t: *mut CpuTensor) {
    if t.is_null() { return; }
    SCOPE_STACK.with(|stack| {
        let mut stack_ref = stack.borrow_mut();
        if let Some(current_scope) = stack_ref.last_mut() {
            // Remove 't' from the current scope.
            // Use retain or finding index. Since we expect 'promote' to be called
            // on return values which are usually recent, finding it should be fast.
            // However, verify reverse search or O(N).
            // For now, simple retain/filter or position check.
             if let Some(pos) = current_scope.iter().rposition(|&x| x == t) {
                 current_scope.remove(pos);
             }
        }
    });
}

pub fn recycle_tensor() -> Option<Box<CpuTensor>> {
    let mut pool = TENSOR_POOL.lock().unwrap();
    pool.pop()
}

// Diagnostics

pub fn get_pool_size() -> usize {
    let pool = TENSOR_POOL.lock().unwrap();
    pool.len()
}

pub fn return_to_pool(mut t: Box<CpuTensor>) {
    if is_mem_log_enabled() {
        eprintln!("[FREE] Ptr: {:p} at {}:{}", t, file!(), line!());
    }
    t.data_f32.clear(); // Keep capacity
    t.data_i64 = None;
    t.shape.clear();
    
    // Recycle gradient tensor if it exists (prevents pool drain/churn)
    if let Some(meta) = t.autograd.take() {
        if let Some(grad) = meta.grad {
            let boxed_grad = Box::new(grad);
            if is_mem_log_enabled() {
                eprintln!("[ALLOC] (GradRecycle) Ptr: {:p} at {}:{}", boxed_grad, file!(), line!());
            }
            return_to_pool(boxed_grad);
        }
    }
    
    let mut pool = TENSOR_POOL.lock().unwrap();
    pool.push(t);
}
