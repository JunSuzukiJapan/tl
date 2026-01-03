use std::ffi::c_void;

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
    // Stack of arena offsets corresponding to each scope
    arena_offsets: Vec<usize>,
}

// SAFETY: MemoryManager contains raw pointers but they are only accessed
// from C code in a single-threaded context (LLVM JIT execution)
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        MemoryManager {
            scopes: Vec::new(),
            arena_offsets: Vec::new(),
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scopes.push(Vec::new());
        // Save current arena offset
        let offset = super::arena::tl_arena_get_offset();
        self.arena_offsets.push(offset);
        // println!(
        //     "DEBUG: Enter Scope. Depth: {}. Arena Offset: {}",
        //     self.scopes.len(),
        //     offset
        // );
    }

    /// Exit current scope and free ALL allocations in that scope
    /// CRITICAL: This MUST free all unfreed memory in the scope
    pub fn exit_scope(&mut self) {
        if self.scopes.is_empty() {
            eprintln!("WARNING: exit_scope called on empty scope stack");
            return;
        }

        // Restore arena offset (freeing all arena allocations in this scope)
        if let Some(offset) = self.arena_offsets.pop() {
            super::arena::tl_arena_set_offset(offset);
            // println!("DEBUG: Exit Scope. Restored Arena Offset: {}", offset);
        }

        if let Some(allocations) = self.scopes.pop() {
            // println!("DEBUG: Freeing {} allocations in scope", allocations.len());
            // Free all allocations in reverse order (LIFO)
            for record in allocations.into_iter().rev() {
                unsafe {
                    match record.alloc_type {
                        AllocationType::Struct => {
                            // eprintln!("DEBUG: freeing struct at {:?}", record.ptr);
                            libc::free(record.ptr);
                        }
                        AllocationType::Tensor => {
                            let tensor_ptr = record.ptr as *mut OpaqueTensor;
                            // eprintln!("DEBUG: freeing tensor at {:?}", tensor_ptr);
                            super::free_tensor_resources(tensor_ptr);
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
        for scope in self.scopes.iter_mut().rev() {
            if let Some(pos) = scope.iter().position(|r| r.ptr == ptr) {
                scope.remove(pos);
                return;
            }
        }
    }
}

// Global memory manager instance
lazy_static::lazy_static! {
    static ref MEMORY_MANAGER: std::sync::Mutex<MemoryManager> = std::sync::Mutex::new(MemoryManager::new());
}

// C-ABI functions for LLVM codegen

static ENTER_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static EXIT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Enter a new memory scope
#[no_mangle]
pub extern "C" fn tl_mem_enter_scope() {
    // let count = ENTER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // if count % 10000 == 0 {
    //     let exits = EXIT_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    //     println!(
    //         "DEBUG SCOPE: Enter={}, Exit={}, Net={}",
    //         count,
    //         exits,
    //         count as isize - exits as isize
    //     );
    // }
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.enter_scope();
}

/// Exit current scope and free all allocations in it
#[no_mangle]
pub extern "C" fn tl_mem_exit_scope() {
    // EXIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

/// Register a tensor allocation (internal)
pub fn register_tensor_global(ptr: *mut OpaqueTensor) {
    if ptr.is_null() {
        return;
    }

    let mut mgr = MEMORY_MANAGER.lock().unwrap();

    // Check if we have an active scope
    if mgr.scopes.is_empty() {
        // No active scope
        // This should not happen in normal usage but prevents crash
        // eprintln!("WARNING: Registering tensor without active scope");
        return;
    }

    mgr.register_tensor(ptr);
}

/// Register a tensor allocation (C API)
#[no_mangle]
pub extern "C" fn tl_mem_register_tensor(ptr: *mut OpaqueTensor) {
    register_tensor_global(ptr);
}

/// Unregister a pointer (e.g. from reassignment or return)
#[no_mangle]
pub extern "C" fn tl_mem_unregister(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.unregister(ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    // Helper to create a dummy pointer (for testing only)
    fn dummy_ptr(offset: usize) -> *mut c_void {
        offset as *mut c_void
    }

    fn dummy_tensor_ptr(offset: usize) -> *mut OpaqueTensor {
        offset as *mut OpaqueTensor
    }

    #[test]
    fn test_basic_scope() {
        let mut mgr = MemoryManager::new();
        assert_eq!(mgr.scopes.len(), 0);
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 1);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 0);
    }

    #[test]
    fn test_nested_scopes() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 1);
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 2);
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 3);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 2);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 1);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 0);
    }

    #[test]
    fn test_register_struct() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        let ptr = dummy_ptr(0x1000);
        mgr.register_struct(ptr);
        assert_eq!(mgr.scopes[0].len(), 1);
        assert_eq!(mgr.scopes[0][0].ptr, ptr);
        assert!(matches!(
            mgr.scopes[0][0].alloc_type,
            AllocationType::Struct
        ));
    }

    #[test]
    fn test_register_tensor() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        let ptr = dummy_tensor_ptr(0x2000);
        mgr.register_tensor(ptr);
        assert_eq!(mgr.scopes[0].len(), 1);
        assert_eq!(mgr.scopes[0][0].ptr, ptr as *mut c_void);
        assert!(matches!(
            mgr.scopes[0][0].alloc_type,
            AllocationType::Tensor
        ));
    }

    #[test]
    fn test_unregister() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        let ptr1 = dummy_ptr(0x1000);
        let ptr2 = dummy_ptr(0x2000);
        let ptr3 = dummy_ptr(0x3000);
        mgr.register_struct(ptr1);
        mgr.register_struct(ptr2);
        mgr.register_struct(ptr3);
        assert_eq!(mgr.scopes[0].len(), 3);
        mgr.unregister(ptr2);
        assert_eq!(mgr.scopes[0].len(), 2);
        assert_eq!(mgr.scopes[0][0].ptr, ptr1);
        assert_eq!(mgr.scopes[0][1].ptr, ptr3);
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        mgr.register_struct(dummy_ptr(0x1000));
        mgr.unregister(dummy_ptr(0x9999));
        assert_eq!(mgr.scopes[0].len(), 1);
    }

    #[test]
    fn test_exit_scope_empty() {
        let mut mgr = MemoryManager::new();
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 0);
    }

    #[test]
    fn test_c_api_functions() {
        tl_mem_enter_scope();
        tl_mem_enter_scope();
        tl_mem_exit_scope();
        tl_mem_exit_scope();
    }
}
