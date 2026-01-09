// use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ffi::c_void;

// use super::OpaqueTensor;

thread_local! {
    static ARENA: RefCell<Option<Arena>> = RefCell::new(None);
}

pub struct Arena {
    _storage: Box<[u8]>, // Key ownership
    buffer: *mut u8,
    capacity: usize,
    offset: usize,
}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        let mut storage = vec![0u8; capacity].into_boxed_slice();
        let buffer = storage.as_mut_ptr();
        Arena {
            _storage: storage,
            buffer,
            capacity,
            offset: 0,
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        // Align offset to requested alignment
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.capacity {
            return std::ptr::null_mut();
        }

        let ptr = unsafe { self.buffer.add(aligned_offset) };
        self.offset = aligned_offset + size;
        ptr
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn contains(&self, ptr: *const u8) -> bool {
        let start = self.buffer as *const u8;
        let end = unsafe { self.buffer.add(self.capacity) } as *const u8;
        ptr >= start && ptr < end
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn set_offset(&mut self, offset: usize) {
        if offset <= self.capacity {
            self.offset = offset;
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // Box<[u8]> drops automatically
    }
}

// C-ABI Functions

/// Initialize arena with specified capacity in bytes
#[no_mangle]
pub extern "C" fn tl_arena_init(capacity: i64) {
    println!("DEBUG: tl_arena_init called with capacity={}", capacity);
    if capacity <= 0 {
        panic!("Arena capacity must be positive, got {}", capacity);
    }

    ARENA.with(|arena| {
        let mut arena_ref = arena.borrow_mut();
        if arena_ref.is_some() {
            eprintln!("Warning: Arena already initialized, reinitializing");
        }
        let arena = Arena::new(capacity as usize);
        *arena_ref = Some(arena);
        println!("DEBUG: tl_arena_init done");
    });
}

/// Allocate memory from arena for OpaqueTensor (testing purpose)
/// Returns address as i64 logic
#[no_mangle]
pub extern "C" fn tl_arena_alloc(size: i64) -> i64 {
    tl_arena_malloc(size) as i64
}

/// Generic arena allocation
#[no_mangle]
pub extern "C" fn tl_arena_malloc(size: i64) -> *mut c_void {
    if size <= 0 {
        return std::ptr::null_mut();
    }

    ARENA.with(|arena| {
        let mut borrow = arena.borrow_mut();
        if let Some(ref mut a) = *borrow {
            // Check overflow before allocate to avoid getting null from allocate()
            // if we want to handle it here.
            let aligned_offset = (a.offset + 15) & !15;
            if aligned_offset + size as usize > a.capacity {
                return std::ptr::null_mut();
            }
            let ptr = a.allocate(size as usize, 16);
            ptr as *mut c_void
        } else {
            std::ptr::null_mut()
        }
    })
}

/// Free arena (deallocate entire buffer)
#[no_mangle]
pub extern "C" fn tl_arena_free() {
    ARENA.with(|arena| {
        *arena.borrow_mut() = None;
    });
}

/// Check if arena is currently active
#[no_mangle]
pub extern "C" fn tl_arena_is_active() -> bool {
    ARENA.with(|arena| arena.borrow().is_some())
}

/// Reset arena (keep buffer, reset offset to 0)
/// Useful for reusing arena across multiple function calls
#[no_mangle]
pub extern "C" fn tl_arena_reset() {
    ARENA.with(|arena| {
        if let Some(ref mut a) = *arena.borrow_mut() {
            a.reset();
        }
    });
}

/// Check if a pointer belongs to the arena
#[no_mangle]
pub extern "C" fn tl_arena_contains(ptr: *mut c_void) -> bool {
    ARENA.with(|arena| {
        if let Some(ref a) = *arena.borrow() {
            a.contains(ptr as *const u8)
        } else {
            false
        }
    })
}

#[no_mangle]
pub extern "C" fn tl_arena_get_offset() -> usize {
    ARENA.with(|arena| {
        if let Some(ref a) = *arena.borrow() {
            a.offset()
        } else {
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn tl_arena_get_capacity() -> usize {
    ARENA.with(|arena| {
        if let Some(ref a) = *arena.borrow() {
            a.capacity
        } else {
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn tl_arena_set_offset(offset: usize) {
    ARENA.with(|arena| {
        if let Some(ref mut a) = *arena.borrow_mut() {
            a.set_offset(offset);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = Arena::new(1024);
        let ptr1 = arena.allocate(64, 16);
        assert!(!ptr1.is_null());
        assert_eq!(ptr1 as usize % 16, 0); // Check alignment

        let ptr2 = arena.allocate(128, 16);
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new(1024);
        let ptr1 = arena.allocate(512, 16);
        arena.reset();
        let ptr2 = arena.allocate(512, 16);
        // After reset, should allocate from beginning again
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = Arena::new(100);
        let ptr = arena.allocate(200, 16);
        assert!(ptr.is_null()); // Should return null, not panic
    }

    #[test]
    fn test_c_api() {
        tl_arena_init(2048);
        assert!(tl_arena_is_active());

        let ptr = tl_arena_alloc(256);
        assert!(ptr != 0);

        tl_arena_free();
        assert!(!tl_arena_is_active());
    }
}
