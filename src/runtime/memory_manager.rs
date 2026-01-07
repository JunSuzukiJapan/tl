use std::collections::HashMap;
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
    // Tensor reference counts: ptr -> refcount
    // When refcount reaches 0, tensor is freed
    tensor_refcounts: HashMap<*mut c_void, usize>,
}

// SAFETY: MemoryManager contains raw pointers but they are only accessed
// from C code in a single-threaded context (LLVM JIT execution)
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new() -> Self {
        MemoryManager {
            scopes: vec![Vec::new()], // Start with Global Scope
            arena_offsets: Vec::new(),
            tensor_refcounts: HashMap::new(),
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scopes.push(Vec::new());
        println!("DEBUG: Enter Scope. Depth: {}", self.scopes.len());
        // Save current arena offset
        let offset = super::arena::tl_arena_get_offset();
        self.arena_offsets.push(offset);
    }

    /// Exit current scope and free ALL allocations in that scope
    /// CRITICAL: This MUST free all unfreed memory in the scope
    pub fn exit_scope(&mut self) {
        println!("DEBUG: Exit Scope. Start Depth: {}", self.scopes.len());
        if self.scopes.len() <= 1 {
            return;
        }

        if let Some(allocations) = self.scopes.pop() {
            // Free all allocations in reverse order (LIFO)
            for record in allocations.into_iter().rev() {
                unsafe {
                    match record.alloc_type {
                        AllocationType::Struct => {
                            // Structs are simple mallocs, just free
                            libc::free(record.ptr);
                        }
                        AllocationType::Tensor => {
                            // Decrement refcount for scope ownership
                            self.release_tensor_ptr(record.ptr);
                        }
                    }
                }
            }
        }

        if let Some(offset) = self.arena_offsets.pop() {
            super::arena::tl_arena_set_offset(offset);
        }
    }

    /// Check if pointer is already registered in ANY scope
    fn is_registered(&self, ptr: *mut c_void) -> bool {
        for scope in &self.scopes {
            if scope.iter().any(|r| r.ptr == ptr) {
                return true;
            }
        }
        return false;
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
            // Initial refcount = 1 (owned by scope)
            let count = self.tensor_refcounts.entry(ptr as *mut c_void).or_insert(0);
            if *count == 0 {
                *count = 1;
                println!("DEBUG: Register Tensor {:p} (New).", ptr);
            } else {
                println!(
                    "DEBUG: Register Tensor {:p} (Existing count: {}).",
                    ptr, *count
                );
            }

            scope.push(AllocationRecord {
                ptr: ptr as *mut c_void,
                alloc_type: AllocationType::Tensor,
            });
        } else {
            println!("DEBUG: register_tensor called but scopes empty!");
        }
    }

    /// Increase reference count
    pub fn acquire_tensor_ptr(&mut self, ptr: *mut c_void) {
        if ptr.is_null() {
            println!("Warning: Attempt to acquire NULL tensor ptr");
            return;
        }
        let count = self.tensor_refcounts.entry(ptr).or_insert(0);
        *count += 1;
        println!("Acquire Tensor {:p}, count: {}", ptr, *count);
    }

    /// Decrease reference count and free if 0
    pub fn release_tensor_ptr(&mut self, ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        if let Some(count) = self.tensor_refcounts.get_mut(&ptr) {
            *count -= 1;
            println!("DEBUG: Release Tensor {:p}, new count: {}", ptr, *count);
            if *count == 0 {
                println!("DEBUG: Freeing Tensor {:p}", ptr);
                self.tensor_refcounts.remove(&ptr);
                super::free_tensor_resources(ptr as *mut OpaqueTensor);
            }
        }
    }

    /// Unregister a pointer (e.g., when it's reassigned)
    /// The memory won't be freed on scope exit
    /// Unregister a pointer from the CURRENT scope only (for move semantics)
    pub fn unregister(&mut self, ptr: *mut c_void) {
        // Iterate scopes in reverse order to find the pointer (most recent first)
        for scope in self.scopes.iter_mut().rev() {
            // Use rposition to find the NEWEST record (handling duplicates/reuse correctly)
            if let Some(pos) = scope.iter().rposition(|r| r.ptr == ptr) {
                scope.remove(pos);
                // CRITICAL FIX: Do NOT release refcount.
                // Unregister means "Remove from Scope Ownership".
                // The RefCount (1) is transferred to the caller (Variable, Struct, etc).
                // If we release, we drop RefCount to 0 and Free!
                println!("DEBUG: Unregistered {:p} from scope (Move)", ptr);
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

/// Register a tensor allocation (internal)
pub fn register_tensor_global(ptr: *mut OpaqueTensor) {
    if ptr.is_null() {
        return;
    }

    let mut mgr = MEMORY_MANAGER.lock().unwrap();

    // Check if we have an active scope
    if mgr.scopes.is_empty() {
        // No active scope
        println!("WARNING: Registering tensor without active scope");
        return;
    }
    // Check if already registered in ANY scope to prevent double-free
    // BUT only if it is still alive (tracked in refcounts)
    if mgr.is_registered(ptr as *mut c_void) {
        if mgr.tensor_refcounts.contains_key(&(ptr as *mut c_void)) {
            return;
        }
        // If not in refcounts, it's a stale record (freed). Allow address reuse.
        println!(
            "DEBUG: Address reuse detected for {:p}, purging stale record.",
            ptr
        );
        // Remove the stale record to prevent confusion
        mgr.unregister(ptr as *mut c_void);
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
        println!("DEBUG: Unregistering {:p}", ptr);
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.unregister(ptr);
    }
}

/// Increase tensor reference count
#[no_mangle]
pub extern "C" fn tl_tensor_acquire(ptr: *mut OpaqueTensor) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.acquire_tensor_ptr(ptr as *mut c_void);
    }
}

/// Decrease tensor reference count
#[no_mangle]
pub extern "C" fn tl_tensor_release(ptr: *mut OpaqueTensor) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.release_tensor_ptr(ptr as *mut c_void);
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
        assert_eq!(mgr.scopes.len(), 1); // Global scope
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 2);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 1);
    }

    #[test]
    fn test_nested_scopes() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 2);
        mgr.enter_scope();
        assert_eq!(mgr.scopes.len(), 3);
        mgr.enter_scope(); // Extra push from original code??
        assert_eq!(mgr.scopes.len(), 4);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 3);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 2);
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 1);
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
