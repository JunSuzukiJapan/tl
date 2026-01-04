use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ffi::c_void;

use super::OpaqueTensor;

thread_local! {
    static ARENA: RefCell<Option<Arena>> = RefCell::new(None);
}

pub struct Arena {
    buffer: *mut u8,
    capacity: usize,
    offset: usize,
}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        unsafe {
            let layout = Layout::from_size_align_unchecked(capacity, 16);
            let buffer = alloc(layout);
            if buffer.is_null() {
                panic!("Arena allocation failed: out of memory");
            }
            Arena {
                buffer,
                capacity,
                offset: 0,
            }
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        // Align offset to requested alignment
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.capacity {
            panic!(
                "Arena overflow! Requested {} bytes (aligned to {}), have {} remaining. \
                 Total capacity: {}, current offset: {}",
                size,
                aligned_offset,
                self.capacity.saturating_sub(aligned_offset),
                self.capacity,
                self.offset
            );
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
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.capacity, 16);
            dealloc(self.buffer, layout);
        }
    }
}

// C-ABI Functions

/// Initialize arena with specified capacity in bytes
#[no_mangle]
pub extern "C" fn tl_arena_init(capacity: i64) {
    if capacity <= 0 {
        panic!("Arena capacity must be positive, got {}", capacity);
    }

    ARENA.with(|arena| {
        let mut arena_ref = arena.borrow_mut();
        if arena_ref.is_some() {
            eprintln!("Warning: Arena already initialized, reinitializing");
        }
        *arena_ref = Some(Arena::new(capacity as usize));
    });
}

/// Allocate memory from arena for OpaqueTensor
/// Returns null if arena is not initialized
#[no_mangle]
pub extern "C" fn tl_arena_alloc(size: i64) -> *mut OpaqueTensor {
    if size <= 0 {
        return std::ptr::null_mut();
    }

    ARENA.with(|arena| {
        let mut borrow = arena.borrow_mut();
        if let Some(ref mut a) = *borrow {
            // Check overflow before allocate to avoid panic in allocate()
            if a.offset + size as usize > a.capacity {
                return std::ptr::null_mut();
            }
            let ptr = a.allocate(size as usize, 16);
            ptr as *mut OpaqueTensor
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
    #[should_panic(expected = "Arena overflow")]
    fn test_arena_overflow() {
        let mut arena = Arena::new(100);
        arena.allocate(200, 16); // Should panic
    }

    #[test]
    fn test_c_api() {
        tl_arena_init(2048);
        assert!(tl_arena_is_active());

        let ptr = tl_arena_alloc(256);
        assert!(!ptr.is_null());

        tl_arena_free();
        assert!(!tl_arena_is_active());
    }
}
