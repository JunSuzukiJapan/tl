use std::ffi::c_void;
use std::sync::Mutex;

use super::OpaqueTensor;

/// Type of allocation being tracked
#[derive(Debug, Clone, Copy)]
enum AllocationType {
    Struct, // malloc'd struct - needs simple free()
    Tensor, // OpaqueTensor* - needs tl_tensor_free()
}

/// Record of a single allocation
#[derive(Debug)]
struct AllocationRecord {
    ptr: *mut c_void,
    alloc_type: AllocationType,
}

/// Scope-based memory manager
/// Tracks allocations per scope and automatically frees them on scope exit
pub struct MemoryManager {
    // Stack of scopes, each scope contains its allocations
    scopes: Vec<Vec<AllocationRecord>>,
}

// SAFETY: MemoryManager contains raw pointers but they are only accessed
// from C code in a single-threaded context (LLVM JIT execution)
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        MemoryManager { scopes: Vec::new() }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    /// Exit current scope and free ALL allocations in that scope
    /// CRITICAL: This MUST free all unfreed memory in the scope
    pub fn exit_scope(&mut self) {
        if let Some(allocations) = self.scopes.pop() {
            // Free all allocations in reverse order (LIFO)
            for record in allocations.into_iter().rev() {
                unsafe {
                    match record.alloc_type {
                        AllocationType::Struct => {
                            // Simple malloc'd struct - use libc free
                            libc::free(record.ptr);
                        }
                        AllocationType::Tensor => {
                            // OpaqueTensor - use tl_tensor_free
                            let tensor_ptr = record.ptr as *mut OpaqueTensor;
                            super::tl_tensor_free(tensor_ptr);
                        }
                    }
                }
            }
        }
    }

    /// Register a struct allocation in the current scope
    pub fn register_struct(&mut self, ptr: *mut c_void) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(AllocationRecord {
                ptr,
                alloc_type: AllocationType::Struct,
            });
        }
    }

    /// Register a tensor allocation in the current scope
    pub fn register_tensor(&mut self, ptr: *mut OpaqueTensor) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(AllocationRecord {
                ptr: ptr as *mut c_void,
                alloc_type: AllocationType::Tensor,
            });
        }
    }

    /// Unregister a pointer (e.g., when it's reassigned)
    /// The memory won't be freed on scope exit
    pub fn unregister(&mut self, ptr: *mut c_void) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.retain(|record| record.ptr != ptr);
        }
    }
}

// Global memory manager instance
lazy_static::lazy_static! {
    static ref MEMORY_MANAGER: Mutex<MemoryManager> = Mutex::new(MemoryManager::new());
}

// C-ABI functions for LLVM codegen

/// Enter a new memory scope
#[no_mangle]
pub extern "C" fn tl_mem_enter_scope() {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.enter_scope();
}

/// Exit current scope and free all allocations in it
#[no_mangle]
pub extern "C" fn tl_mem_exit_scope() {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.exit_scope();
}

/// Register a struct allocation
#[no_mangle]
pub extern "C" fn tl_mem_register_struct(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.register_struct(ptr);
    }
}

/// Register a tensor allocation (Internal Rust API)
pub fn register_tensor_global(ptr: *mut OpaqueTensor) {
    if ptr.is_null() {
        eprintln!("WARNING: Attempted to register null tensor pointer");
        return;
    }

    let mut mgr = MEMORY_MANAGER.lock().unwrap();

    // Check if we have an active scope
    if mgr.scopes.is_empty() {
        // No active scope
        // This should not happen in normal usage but prevents crash
        eprintln!("WARNING: Registering tensor without active scope");
        return;
    }

    mgr.register_tensor(ptr);
}

/// Register a tensor allocation (C API)
#[no_mangle]
pub extern "C" fn tl_mem_register_tensor(ptr: *mut OpaqueTensor) {
    register_tensor_global(ptr);
}

/// Unregister a pointer (won't be freed on scope exit)
#[no_mangle]
pub extern "C" fn tl_mem_unregister(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.unregister(ptr);
    }
}
