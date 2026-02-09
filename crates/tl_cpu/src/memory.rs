use std::cell::RefCell;
use std::sync::Mutex;
use crate::tensor::CpuTensor;

// Global Tensor Pool to reduce allocation overhead
// Stores released tensors for reuse.
static TENSOR_POOL: Mutex<Vec<Box<CpuTensor>>> = Mutex::new(Vec::new());

thread_local! {
    // Stack of scopes. Each scope contains a list of tensors allocated within it.
    // When a scope exits, all tensors in it are returned to the global pool.
    static SCOPE_STACK: RefCell<Vec<Vec<*mut CpuTensor>>> = RefCell::new(Vec::new());
}

pub fn enter_scope() {
    SCOPE_STACK.with(|stack| {
        stack.borrow_mut().push(Vec::new());
    });
}

pub fn exit_scope() {
    let tensors_to_free = SCOPE_STACK.with(|stack| {
        stack.borrow_mut().pop()
    });

    if let Some(tensors) = tensors_to_free {
        let mut pool = TENSOR_POOL.lock().unwrap();
        for ptr in tensors {
            if ptr.is_null() { continue; }
            // Safety: The tensor is owned by the scope and we are the only ones
            // who should be freeing it (except for promoted ones which are removed).
            // We convert it back to Box to own it, then push to pool.
            let mut t = unsafe { Box::from_raw(ptr) };
            
            // Reset tensor for reuse (keep capacity)
            t.data_f32.clear();
            t.data_i64 = None;
            t.shape.clear();
            t.autograd = None;

            pool.push(t);
        }
    }
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
