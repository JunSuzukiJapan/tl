// Tensor Pool for small-to-medium sized tensors
// This reduces allocation overhead for frequently created/destroyed tensors

use super::OpaqueTensor;
use std::cell::RefCell;

const MAX_POOL_SIZE: usize = 32; // Max tensors per size bucket
const MAX_ELEMENT_COUNT: usize = 64; // Only pool tensors with ≤64 elements

/// Size bucket for pooling (based on element count)
fn size_bucket(element_count: usize) -> Option<usize> {
    match element_count {
        1..=4 => Some(0),
        5..=8 => Some(1),
        9..=16 => Some(2),
        17..=32 => Some(3),
        33..=64 => Some(4),
        _ => None,
    }
}

pub struct TensorPool {
    // 5 buckets: 1-4, 5-8, 9-16, 17-32, 33-64 elements
    pools: [Vec<*mut OpaqueTensor>; 5],
}

impl TensorPool {
    pub fn new() -> Self {
        TensorPool {
            pools: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Try to acquire a tensor from the pool
    pub fn acquire(&mut self, element_count: usize) -> Option<*mut OpaqueTensor> {
        if let Some(bucket) = size_bucket(element_count) {
            self.pools[bucket].pop()
        } else {
            None
        }
    }

    /// Release a tensor back to the pool
    pub fn release(&mut self, ptr: *mut OpaqueTensor, element_count: usize) {
        if let Some(bucket) = size_bucket(element_count) {
            if self.pools[bucket].len() < MAX_POOL_SIZE {
                self.pools[bucket].push(ptr);
            } else {
                // Pool is full, actually free the tensor
                unsafe {
                    let _ = Box::from_raw(ptr);
                }
            }
        } else {
            // Not poolable, actually free
            unsafe {
                let _ = Box::from_raw(ptr);
            }
        }
    }

    /// Clear all pools (for cleanup)
    pub fn clear(&mut self) {
        for pool in self.pools.iter_mut() {
            for ptr in pool.drain(..) {
                unsafe {
                    let _ = Box::from_raw(ptr);
                }
            }
        }
    }
}

impl Drop for TensorPool {
    fn drop(&mut self) {
        self.clear();
    }
}

// Thread-local tensor pool
thread_local! {
    static TENSOR_POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

/// Try to acquire a pooled tensor (internal Rust API)
pub fn pool_acquire(element_count: usize) -> Option<*mut OpaqueTensor> {
    if element_count > MAX_ELEMENT_COUNT {
        return None;
    }
    TENSOR_POOL.with(|pool| pool.borrow_mut().acquire(element_count))
}

/// Release a tensor back to the pool (internal Rust API)
pub fn pool_release(ptr: *mut OpaqueTensor, element_count: usize) {
    if element_count > MAX_ELEMENT_COUNT {
        // Not poolable, free directly
        unsafe {
            let _ = Box::from_raw(ptr);
        }
        return;
    }
    TENSOR_POOL.with(|pool| pool.borrow_mut().release(ptr, element_count));
}

// C-ABI exports for LLVM codegen

/// Try to acquire a pooled tensor (C API)
#[no_mangle]
pub extern "C" fn tl_pool_acquire(element_count: usize) -> *mut OpaqueTensor {
    pool_acquire(element_count).unwrap_or(std::ptr::null_mut())
}

/// Release a tensor back to the pool (C API)
#[no_mangle]
pub extern "C" fn tl_pool_release(ptr: *mut OpaqueTensor, element_count: usize) {
    if !ptr.is_null() {
        pool_release(ptr, element_count);
    }
}
