use std::sync::atomic::{AtomicI64, AtomicI32, Ordering};
use std::sync::Arc;

// --- AtomicI64 ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_new(val: i64) -> *mut Arc<AtomicI64> {
    let arc = Arc::new(AtomicI64::new(val));
    Box::into_raw(Box::new(arc))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_load(ptr: *const Arc<AtomicI64>) -> i64 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.load(Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_store(ptr: *const Arc<AtomicI64>, val: i64) {
    if ptr.is_null() { return; }
    let arc = unsafe { &*ptr };
    arc.store(val, Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_add(ptr: *const Arc<AtomicI64>, val: i64) -> i64 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.fetch_add(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_sub(ptr: *const Arc<AtomicI64>, val: i64) -> i64 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.fetch_sub(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_swap(ptr: *const Arc<AtomicI64>, val: i64) -> i64 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.swap(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_compare_exchange(ptr: *const Arc<AtomicI64>, current: i64, new: i64) -> bool {
    if ptr.is_null() { return false; }
    let arc = unsafe { &*ptr };
    arc.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst).is_ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_clone(ptr: *const Arc<AtomicI64>) -> *mut Arc<AtomicI64> {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let arc = unsafe { &*ptr };
    Box::into_raw(Box::new(arc.clone()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i64_free(ptr: *mut Arc<AtomicI64>) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

// --- AtomicI32 ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_new(val: i32) -> *mut Arc<AtomicI32> {
    let arc = Arc::new(AtomicI32::new(val));
    Box::into_raw(Box::new(arc))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_load(ptr: *const Arc<AtomicI32>) -> i32 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.load(Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_store(ptr: *const Arc<AtomicI32>, val: i32) {
    if ptr.is_null() { return; }
    let arc = unsafe { &*ptr };
    arc.store(val, Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_add(ptr: *const Arc<AtomicI32>, val: i32) -> i32 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.fetch_add(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_sub(ptr: *const Arc<AtomicI32>, val: i32) -> i32 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.fetch_sub(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_swap(ptr: *const Arc<AtomicI32>, val: i32) -> i32 {
    if ptr.is_null() { return 0; }
    let arc = unsafe { &*ptr };
    arc.swap(val, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_compare_exchange(ptr: *const Arc<AtomicI32>, current: i32, new: i32) -> bool {
    if ptr.is_null() { return false; }
    let arc = unsafe { &*ptr };
    arc.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst).is_ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_clone(ptr: *const Arc<AtomicI32>) -> *mut Arc<AtomicI32> {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let arc = unsafe { &*ptr };
    Box::into_raw(Box::new(arc.clone()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_atomic_i32_free(ptr: *mut Arc<AtomicI32>) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}
