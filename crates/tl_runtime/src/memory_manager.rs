use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_char;

use super::OpaqueTensor;

#[derive(Clone)]
pub struct AllocationMeta {
    pub ctx: String,
    pub bytes: usize,
    pub dtype: String,
    pub elems: usize,
    pub shape: String,
    pub device: String,
    pub loc_file: String,
    pub loc_line: u32,
    pub pooled: bool,
}

/// Tensor pool for memory reuse
/// Key: (num_elements, dtype_id, device_id)
/// This avoids frequent malloc/free by reusing freed tensors of the same size
pub struct TensorPool {
    // (num_elements, dtype_id, device_id) -> Vec<*mut OpaqueTensor>
    free_list: HashMap<(usize, u8, u8), Vec<*mut OpaqueTensor>>,
    base_max_per_size: usize,
}

// SAFETY: TensorPool contains raw pointers but they are only accessed
// from C code in a single-threaded context (LLVM JIT execution)
unsafe impl Send for TensorPool {}
unsafe impl Sync for TensorPool {}

#[derive(Debug, PartialEq, Eq)]
pub enum PoolOutcome {
    Pooled,
    Full,
    Duplicate,
}

impl TensorPool {
    pub fn new() -> Self {
        TensorPool {
            free_list: HashMap::new(),
            base_max_per_size: 32, // Base max tensors per size bucket
        }
    }

    /// Calculate effective max_per_size based on KV cache memory usage
    /// When KV cache uses more memory, pool size is reduced
    fn effective_max_per_size(&self) -> usize {
        // Import the function from llm module
        let kv_bytes = super::llm::tl_kv_cache_get_memory_usage() as usize;

        // Calculate reduction based on KV cache usage
        // For every 100MB of KV cache, reduce max_per_size by 4
        let reduction = kv_bytes / (100 * 1024 * 1024) * 4;
        let effective = self.base_max_per_size.saturating_sub(reduction);

        // Minimum of 4 to ensure some pooling still happens
        effective.max(4)
    }

    /// Try to acquire a tensor from the pool
    /// Returns None if no matching tensor is available
    pub fn acquire(
        &mut self,
        num_elements: usize,
        dtype_id: u8,
        device_id: u8,
    ) -> Option<*mut OpaqueTensor> {
        let key = (num_elements, dtype_id, device_id);
        if let Some(list) = self.free_list.get_mut(&key) {
            if let Some(ptr) = list.pop() {
                return Some(ptr);
            }
        }
        None
    }

    /// Release a tensor to the pool
    /// Returns PoolOutcome to indicate status
    pub fn release(
        &mut self,
        ptr: *mut OpaqueTensor,
        num_elements: usize,
        dtype_id: u8,
        device_id: u8,
    ) -> PoolOutcome {
        // Calculate effective max before borrowing free_list mutably
        let effective_max = self.effective_max_per_size();

        let key = (num_elements, dtype_id, device_id);
        let list = self.free_list.entry(key).or_insert_with(Vec::new);

        // CRITICAL FIX: Check for duplicate pointer to prevent double-pool
        // This can happen when the same tensor is released multiple times
        // due to address reuse or reference counting bugs
        if list.contains(&ptr) {
            // println!("WARNING: Attempted to pool duplicate pointer {:p}, ignoring", ptr);
            return PoolOutcome::Duplicate;
        }

        if list.len() < effective_max {
            list.push(ptr);
            PoolOutcome::Pooled
        } else {
            // Pool is full, tensor should be freed
            PoolOutcome::Full
        }
    }

    /// Clear all tensors from the pool (actually free them)
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for (_, list) in self.free_list.drain() {
            for ptr in list {
                unsafe {
                    // println!("TensorPool clearing {:p}", ptr);
                    let _ = Box::from_raw(ptr);
                }
            }
        }
    }

    /// Total number of pooled tensors
    pub fn total_count(&self) -> usize {
        self.free_list.values().map(|v| v.len()).sum()
    }
}

// Global tensor pool instance
lazy_static::lazy_static! {
    pub static ref TENSOR_POOL: std::sync::Mutex<TensorPool> = std::sync::Mutex::new(TensorPool::new());
}

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
    // Object reference counts: ptr -> refcount
    // When refcount reaches 0, object is freed
    refcounts: HashMap<*mut c_void, usize>,
    // Track type of each pointer for generic release
    ptr_types: HashMap<*mut c_void, AllocationType>,
    // Allocation metadata for logging (ptr -> meta)
    tensor_meta: HashMap<*mut c_void, AllocationMeta>,
    
    // Stack of Function Frames for reused buffers
    call_stack_frames: Vec<FunctionFrame>,
}

struct FunctionFrame {
    // Frame: Vec<Slot>, Slot: Option<(Pointer, Capacity)>
    slots: Vec<Option<(*mut c_void, usize)>>,
    // Old buffers kept alive until frame exit to prevent Use-After-Free
    trash: Vec<*mut c_void>,
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
            refcounts: HashMap::new(),
            ptr_types: HashMap::new(),
            tensor_meta: HashMap::new(),
            call_stack_frames: Vec::new(),
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scopes.push(Vec::new());
        // println!("Enter Scope. Depth: {}", self.scopes.len());
        // Save current arena offset
        let offset = super::arena::tl_arena_get_offset();
        self.arena_offsets.push(offset);
        if crate::mem_log_enabled() {
            eprintln!(
                "[TL_MEM] enter_scope depth={}",
                self.scopes.len()
            );
        }
    }

    /// Exit current scope and free ALL allocations in that scope
    /// CRITICAL: This MUST free all unfreed memory in the scope
    pub fn exit_scope(&mut self) {
        // println!("Exit Scope. Start Depth: {}", self.scopes.len());
        if self.scopes.len() <= 1 {
            return;
        }

        if let Some(allocations) = self.scopes.pop() {
            let mut total_allocs = 0usize;
            let mut tensor_allocs = 0usize;
            let mut struct_allocs = 0usize;
            for record in allocations.iter() {
                total_allocs += 1;
                match record.alloc_type {
                    AllocationType::Struct => struct_allocs += 1,
                    AllocationType::Tensor => tensor_allocs += 1,
                }
            }
            if crate::mem_log_enabled() {
                eprintln!(
                    "[TL_MEM] exit_scope begin depth={} allocs_total={} allocs_tensor={} allocs_struct={}",
                    self.scopes.len() + 1,
                    total_allocs,
                    tensor_allocs,
                    struct_allocs
                );
            }
            // Free all allocations in reverse order (LIFO)
            for record in allocations.into_iter().rev() {
                 // Unified release (decrements refcount)
                 self.release_ptr(record.ptr);
            }
        }

        if let Some(offset) = self.arena_offsets.pop() {
            super::arena::tl_arena_set_offset(offset);
        }
        if crate::mem_log_enabled() {
            let pool_count = if let Ok(pool) = TENSOR_POOL.lock() {
                pool.total_count()
            } else {
                0
            };
            eprintln!(
                "[TL_MEM] exit_scope end depth={} refcount_count={} pool_count={}",
                self.scopes.len(),
                self.refcounts.len(),
                pool_count
            );
        }
    }

    /// Check if pointer is already registered in ANY scope
    #[allow(dead_code)]
    fn is_registered(&self, ptr: *mut c_void) -> bool {
        for scope in &self.scopes {
            if scope.iter().any(|r| r.ptr == ptr) {
                return true;
            }
        }
        return false;
    }

    pub fn register_struct_named(&mut self, ptr: *mut c_void, name: *const c_char) {
        if ptr.is_null() { return; }
        
        let name_str = if !name.is_null() {
            unsafe { std::ffi::CStr::from_ptr(name).to_string_lossy().to_string() }
        } else {
            "struct".to_string()
        };

        if crate::mem_log_enabled() {
             eprintln!("[TL_MEM] register_struct_named ptr={:p} name='{}'", ptr, name_str);
        }

        // DEBUG: Check if we are registering an Arena pointer as a Struct
        if super::arena::tl_arena_contains(ptr) {
            eprintln!("[TL_MEM] WARNING: Registering Arena Pointer {:p} (name='{}') as Struct! This will cause Double Free/Bad Free.", ptr, name_str);
        }

        self.ptr_types.entry(ptr).or_insert(AllocationType::Struct);

        if let Some(scope) = self.scopes.last_mut() {
             let count = self.refcounts.entry(ptr).or_insert(0);
             if *count == 0 {
                 *count = 1;
             } else {
                 *count += 1;
             }
             scope.push(AllocationRecord {
                 ptr,
                 alloc_type: AllocationType::Struct,
             });
        }
        if crate::mem_log_enabled() {
            eprintln!(
                "[TL_MEM] register_struct name={} ptr={:p} depth={}",
                name_str,
                ptr,
                self.scopes.len()
            );
        }
    }

    /// Register a struct allocation in the current scope
    pub fn register_struct(&mut self, ptr: *mut c_void) {
        self.register_struct_named(ptr, std::ptr::null());
    }

    /// Register a tensor allocation in the current scope
    pub fn register_tensor(&mut self, ptr: *mut OpaqueTensor) {
        let ptr_c = ptr as *mut c_void;
        self.ptr_types.entry(ptr_c).or_insert(AllocationType::Tensor);
        
        if let Some(scope) = self.scopes.last_mut() {
            // Initial refcount = 1 (owned by scope)
            let count = self.refcounts.entry(ptr_c).or_insert(0);
            if *count == 0 {
                *count = 1;
            } else {
                *count += 1;
            }
            // ALWAYS add to scope
            scope.push(AllocationRecord {
                ptr: ptr_c,
                alloc_type: AllocationType::Tensor,
            });
            
            if crate::mem_log_enabled() {
                eprintln!(
                    "[TL_MEM] register_tensor ptr={:p} refcount={} depth={}",
                    ptr,
                    *count,
                    self.scopes.len()
                );
            }
        } else {
            // No active scope, this should ideally not happen if scopes are managed correctly
        }
    }

    pub fn register_tensor_meta(&mut self, ptr: *mut c_void, meta: AllocationMeta) {
        if ptr.is_null() {
            return;
        }
        self.tensor_meta.insert(ptr, meta);
    }

    /// Increase reference count (Generic)
    pub fn acquire_ptr(&mut self, ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        let count = self.refcounts.entry(ptr).or_insert(0);
        *count += 1;
        if crate::mem_log_enabled() {
            eprintln!(
                "[TL_MEM] acquire ptr={:p} refcount={}",
                ptr,
                *count
            );
        }
    }

    /// Decrease reference count and free if 0 (Generic)
    /// Returns true if the pointer was found and processed, false if generic logic should apply (unregistered)
    pub fn release_ptr(&mut self, ptr: *mut c_void) -> bool {
        if ptr.is_null() {
            return false;
        }
        if let Some(count) = self.refcounts.get_mut(&ptr) {
            *count -= 1;
            if crate::mem_log_enabled() {
                eprintln!(
                    "[TL_MEM] release ptr={:p} refcount={}",
                    ptr,
                    *count
                );
            }
            if *count == 0 {
                self.refcounts.remove(&ptr);
                
                // Check type to determine free method
                if let Some(alloc_type) = self.ptr_types.remove(&ptr) {
                    match alloc_type {
                        AllocationType::Struct => {
                             if crate::mem_log_enabled() {
                                 eprintln!("[TL_MEM] free_struct ptr={:p} in_arena={}", ptr, super::arena::tl_arena_contains(ptr));
                             }
                             // CRITICAL FIX: Ensure we don't free Arena pointers treated as Structs
                             if !super::arena::tl_arena_contains(ptr) {
                                 unsafe { libc::free(ptr); }
                             }
                        }
                        AllocationType::Tensor => {
                            let _meta = self.tensor_meta.remove(&ptr);
                            let outcome = super::free_tensor_resources(ptr as *mut OpaqueTensor);
                             if crate::mem_log_enabled() {
                                 let outcome_str = match outcome {
                                     crate::FreeOutcome::ArenaDrop => "arena_drop",
                                     crate::FreeOutcome::Pooled => "pooled",
                                     crate::FreeOutcome::Freed => "freed",
                                 };
                                 // Logging... (Simplified for brevity as original was long)
                                 eprintln!("[TL_MEM] freed tensor outcome={}", outcome_str);
                             }
                        }
                    }
                } else {
                     // Assume Struct if unknown (fallback?) or just leak/warn?
                     // If it was registered, it should have a type.
                     // If we are here, something went wrong or it was just a raw ptr.
                     // Safer to free as Struct (simple free) if refcount was tracked?
                     // But we removed from refcounts. 
                }
            }
            return true;
        }
        false
    }
    
    // Legacy specialized method (delegate)
    pub fn release_tensor_ptr(&mut self, ptr: *mut c_void) {
        self.release_ptr(ptr);
    }



    /// Register a pointer with known type (Internal)
    pub fn register_any_ptr(&mut self, ptr: *mut c_void) {
        if ptr.is_null() { return; }
        
        let alloc_type = if let Some(t) = self.ptr_types.get(&ptr) {
             *t
        } else {
             return; 
        };

        if let Some(scope) = self.scopes.last_mut() {
             let count = self.refcounts.entry(ptr).or_insert(0);
             if *count == 0 {
                  *count = 1;
             } else {
                  *count += 1;
             }
             scope.push(AllocationRecord { ptr, alloc_type });
        }
    }

    /// Enter a function frame with N slots
    pub fn function_enter(&mut self, num_slots: usize) {
        // Create new scope for automatic cleanup of registered tensors
        self.enter_scope();
        
        let frame = FunctionFrame {
            slots: vec![None; num_slots],
            trash: Vec::new(),
        };
        self.call_stack_frames.push(frame);
        if crate::mem_log_enabled() {
             eprintln!("[TL_MEM] function_enter slots={} depth={}", num_slots, self.scopes.len());
        }
    }

    /// Exit function frame and free all slot buffers
    pub fn function_exit(&mut self) {
        // Free registered tensors in this scope
        self.exit_scope();

        if let Some(frame) = self.call_stack_frames.pop() {
            // Free active slots
            for (i, slot) in frame.slots.into_iter().enumerate() {
                if let Some((ptr, size)) = slot {
                    unsafe { libc::free(ptr); }
                    if crate::mem_log_enabled() {
                         eprintln!("[TL_MEM] function_exit free_slot id={} ptr={:p} size={}", i, ptr, size);
                    }
                }
            }
            // Free trash (deferred)
            for ptr in frame.trash {
                unsafe { libc::free(ptr); }
                if crate::mem_log_enabled() {
                     eprintln!("[TL_MEM] function_exit free_trash ptr={:p}", ptr);
                }
            }
        }
    }

    /// Get a buffer for a slot, reallocating if necessary
    pub fn get_buffer(&mut self, slot_id: usize, min_size: usize) -> *mut c_void {
        if let Some(frame) = self.call_stack_frames.last_mut() {
            if slot_id < frame.slots.len() {
                if let Some((ptr, cap)) = frame.slots[slot_id] {
                    if cap >= min_size {
                        // Reuse
                        if crate::mem_log_enabled() {
                             eprintln!("[TL_MEM] reuse_buffer slot={} ptr={:p} cap={}", slot_id, ptr, cap);
                        }
                        return ptr;
                    } else {
                        // Grow: Alloc New + Defere Free Old (No Copy)
                        let new_ptr = unsafe { libc::malloc(min_size) };
                        if new_ptr.is_null() {
                            panic!("[TL_MEM] Slot buffer malloc failed");
                        }
                        
                        // Move old ptr to trash (Deferred Free)
                         // We do NOT copy data. Caller must handle initialization.
                        frame.trash.push(ptr);
                        
                        // Update slot
                        frame.slots[slot_id] = Some((new_ptr, min_size));
                        
                        if crate::mem_log_enabled() {
                             eprintln!("[TL_MEM] expand_buffer_nocopy slot={} old_cap={} new_cap={} old_ptr={:p} new_ptr={:p}", slot_id, cap, min_size, ptr, new_ptr);
                        }
                        return new_ptr;
                    }
                } else {
                    // Start Alloc
                    let new_ptr = unsafe { libc::malloc(min_size) };
                    frame.slots[slot_id] = Some((new_ptr, min_size));
                    if crate::mem_log_enabled() {
                         eprintln!("[TL_MEM] alloc_buffer slot={} size={} ptr={:p}", slot_id, min_size, new_ptr);
                    }
                    return new_ptr;
                }
            }
            eprintln!("[TL_MEM] ERROR: Slot ID {} out of bounds (frame size {})", slot_id, frame.slots.len());
        } else {
            eprintln!("[TL_MEM] ERROR: No function frame active for get_buffer");
        }
        std::ptr::null_mut()
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
                if crate::mem_log_enabled() {
                    let refcount = self.refcounts.get(&ptr).cloned().unwrap_or(0);
                    eprintln!(
                        "[TL_MEM] unregister ptr={:p} refcount={} depth={}",
                        ptr,
                        refcount,
                        self.scopes.len()
                    );
                }
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
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_enter_scope() {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.enter_scope();
}

/// Exit current scope and free all allocations in it
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_exit_scope() {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.exit_scope();
}

/// Register a struct allocation
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.register_struct(ptr);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_struct_named(ptr: *mut c_void, name: *const std::os::raw::c_char) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.register_struct_named(ptr, name);
    }
}

pub fn register_tensor_global(ptr: *mut OpaqueTensor) {
    if ptr.is_null() {
        return;
    }

    let mut mgr = MEMORY_MANAGER.lock().unwrap();

    // Check if we have an active scope
    if mgr.scopes.is_empty() {
        return;
    }
    // unified logic in register_tensor
    mgr.register_tensor(ptr);
}

/// Generic register for any pointer (looks up type or assumes struct if new?)
/// Actually we need this for Vec::get. 
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_ptr(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.register_any_ptr(ptr);
    }
}




pub fn register_tensor_meta_global(ptr: *mut OpaqueTensor, meta: AllocationMeta) {
    if ptr.is_null() {
        return;
    }
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.register_tensor_meta(ptr as *mut c_void, meta);
}

/// Register a tensor allocation (C API)
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_register_tensor(ptr: *mut OpaqueTensor) {
    register_tensor_global(ptr);
}

/// Unregister a pointer (e.g. from reassignment or return)
#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_unregister(ptr: *mut c_void) {
    if !ptr.is_null() {
        if crate::mem_log_enabled() {
            eprintln!("[TL_MEM] tl_mem_unregister called for ptr={:p}", ptr);
        }
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.unregister(ptr);
    } else {
        // ...
    }
}

/// Increase tensor reference count
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_acquire(ptr: *mut OpaqueTensor) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.acquire_ptr(ptr as *mut c_void);
    }
}

/// Increase reference count (Generic)
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_acquire(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.acquire_ptr(ptr);
    }
}

/// Decrement refcount and return true if it should be freed (Ref==0)
/// Removes from refcounts if 0 to prepare for free.
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_dec_ref(ptr: *mut c_void) -> i32 {
    if ptr.is_null() {
        return 0; // False
    }
    if crate::mem_log_enabled() {
        eprintln!("[TL_MEM] dec_ref check ptr={:p}", ptr);
    }
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    if let Some(count) = mgr.refcounts.get_mut(&ptr) {
        if *count > 1 {
            *count -= 1;
            if crate::mem_log_enabled() {
                eprintln!("[TL_MEM] dec_ref decrement ptr={:p} new_count={}", ptr, *count);
            }
            return 0; // False (Still shared)
        } else {
            // Count == 1 -> Going to 0
            mgr.refcounts.remove(&ptr);
            mgr.ptr_types.remove(&ptr); 
            mgr.tensor_meta.remove(&ptr);
            if crate::mem_log_enabled() {
                eprintln!("[TL_MEM] dec_ref zero ptr={:p}", ptr);
            }
            return 1; // True (Should free)
        }
    }
    // Not found -> Assume unmanaged -> True
    if crate::mem_log_enabled() {
        eprintln!("[TL_MEM] dec_ref not_found ptr={:p}", ptr);
    }
    1
}

/// Increase reference count (Generic)
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_inc_ref(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    
    // If we have track of it, increment.
    // If not, it means we are converting from Single Ownership (1, implicit) to Shared (2).
    // But wait, our system assumes unregistered pointers have refcount 1 if they are managed.
    // Actually, `register_struct` calls put it in Refcounts? No.
    // Tensors are registered in `tl_mem_register_tensor`.
    // Structs are registered in `tl_mem_register_struct_named`.
    
    if let Some(count) = mgr.refcounts.get_mut(&ptr) {
        *count += 1;
        if crate::mem_log_enabled() {
            eprintln!("[TL_MEM] inc_ref ptr={:p} new_count={}", ptr, *count);
        }
    } else {
        // It's not in refcounts.
        // If it's a managed pointer (in scopes), it implicity has usage 1.
        // We upgrading it to 2.
        // However, we need to know if it's actually managed.
        // If it was allocated by malloc or arena, we track it?
        // Let's assume valid pointer provided by compiler needs tracking.
        mgr.refcounts.insert(ptr, 2);
        if crate::mem_log_enabled() {
            eprintln!("[TL_MEM] inc_ref new_entry ptr={:p} count=2", ptr);
        }
    }
}

/// Decrease reference count (Generic) - Returns bool
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_release_bool(ptr: *mut c_void) -> bool {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.release_ptr(ptr)
    } else {
        false
    }
}

/// Decrease reference count (Generic)
#[unsafe(no_mangle)]
pub extern "C" fn tl_ptr_release(ptr: *mut c_void) {
    if !ptr.is_null() {
        let mut mgr = MEMORY_MANAGER.lock().unwrap();
        mgr.release_ptr(ptr);
    }
}

/// Decrease tensor reference count
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_release(ptr: *mut OpaqueTensor) {
    tl_ptr_release(ptr as *mut c_void);
}

/// Get number of tensors currently stored in the pool
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_pool_count() -> i64 {
    if let Ok(pool) = TENSOR_POOL.lock() {
        pool.total_count() as i64
    } else {
        -1
    }
}

/// Get number of live tensor refcount entries
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_refcount_count() -> i64 {
    let mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.refcounts.len() as i64
}

/// Get current scope depth (including global scope)
#[unsafe(no_mangle)]
pub extern "C" fn tl_get_scope_depth() -> i64 {
    let mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.scopes.len() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_enter(num_slots: i64) {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.function_enter(num_slots as usize);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_function_exit() {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.function_exit();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_mem_get_buffer(slot_id: i64, min_size: i64) -> *mut c_void {
    let mut mgr = MEMORY_MANAGER.lock().unwrap();
    mgr.get_buffer(slot_id as usize, min_size as usize)
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
        assert_eq!(mgr.scopes[1].len(), 1);
        assert_eq!(mgr.scopes[1][0].ptr, ptr);
        assert!(matches!(
            mgr.scopes[1][0].alloc_type,
            AllocationType::Struct
        ));
    }

    #[test]
    fn test_register_tensor() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        let ptr = dummy_tensor_ptr(0x2000);
        mgr.register_tensor(ptr);
        assert_eq!(mgr.scopes[1].len(), 1);
        assert_eq!(mgr.scopes[1][0].ptr, ptr as *mut c_void);
        assert!(matches!(
            mgr.scopes[1][0].alloc_type,
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
        assert_eq!(mgr.scopes[1].len(), 3);
        mgr.unregister(ptr2);
        assert_eq!(mgr.scopes[1].len(), 2);
        assert_eq!(mgr.scopes[1][0].ptr, ptr1);
        assert_eq!(mgr.scopes[1][1].ptr, ptr3);
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut mgr = MemoryManager::new();
        mgr.enter_scope();
        mgr.register_struct(dummy_ptr(0x1000));
        mgr.unregister(dummy_ptr(0x9999));
        assert_eq!(mgr.scopes[1].len(), 1);
    }

    #[test]
    fn test_exit_scope_empty() {
        let mut mgr = MemoryManager::new();
        mgr.exit_scope();
        assert_eq!(mgr.scopes.len(), 1);
    }

    #[test]
    fn test_c_api_functions() {
        tl_mem_enter_scope();
        tl_mem_enter_scope();
        tl_mem_exit_scope();
        tl_mem_exit_scope();
    }
}
