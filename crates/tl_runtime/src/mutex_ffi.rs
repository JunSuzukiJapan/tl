use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::copy_nonoverlapping;

/// Opaque wrapper for Mutex data
pub struct TlMutexState {
    pub data_ptr: *mut c_void,
    pub layout: Layout,
}

// We implement Send and Sync because data_ptr points to heap allocated value that is 
// synchronized through the Mutex wrapper itself.
unsafe impl Send for TlMutexState {}
unsafe impl Sync for TlMutexState {}

impl Drop for TlMutexState {
    fn drop(&mut self) {
        if !self.data_ptr.is_null() && self.layout.size() > 0 {
            unsafe {
                dealloc(self.data_ptr as *mut u8, self.layout);
            }
        }
    }
}

/// Global registry for TL Mutexes
static MUTEX_REGISTRY: Lazy<Mutex<HashMap<i64, Arc<Mutex<TlMutexState>>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_MUTEX_ID: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(1));

/// @ffi_sig (i64, *mut c_void) -> i64
/// Creates a new Mutex containing a copy of the data.
/// Returns the Mutex ID.
#[unsafe(no_mangle)]
pub extern "C" fn tl_mutex_new(size: i64, data_ptr: *mut c_void) -> i64 {
    let size = size as usize;
    
    // Create new heap allocation for the data
    let layout = if size > 0 {
        Layout::array::<u8>(size).unwrap()
    } else {
        Layout::from_size_align(0, 1).unwrap()
    };
    
    let new_ptr = if size > 0 && !data_ptr.is_null() {
        unsafe {
            let p = alloc(layout) as *mut c_void;
            copy_nonoverlapping(data_ptr, p, size);
            p
        }
    } else {
        std::ptr::null_mut()
    };

    let state = TlMutexState {
        data_ptr: new_ptr,
        layout,
    };
    
    let arc_mutex = Arc::new(Mutex::new(state));
    
    let id = {
        let mut id_lock = NEXT_MUTEX_ID.lock().unwrap();
        let current_id = *id_lock;
        *id_lock += 1;
        current_id
    };

    let mut registry = MUTEX_REGISTRY.lock().unwrap();
    registry.insert(id, arc_mutex);

    id
}


// The closure returns a pointer (to the modified data) which might be an Opaque pointer or small value packed into pointer space?
// Wait, generic TL closures return their direct value. But since LLVM doesn't know the generic 
// return type in advance perfectly for C FFI, it's easier to pass pointers out, or we can just 
// have the closure take `(*mut c_void env, *mut c_void param_ptr)` and write the result into `param_ptr` directly if we pass it by ref.

// For maximum compatibility with TL's generic ABI:
// A closure for `modify(|x| -> T)` will be compiled as:
// `fn(env_ptr: *mut c_void, current_value_ptr: *mut T) -> T`
// But we actually want to write the new `T` back to the mutex's memory.
// Let's design the FFI closure as:
// `fn(env_ptr: *mut c_void, value_ptr: *mut c_void, out_ptr: *mut c_void)`
// where `value_ptr` takes the current data, and `out_ptr` receives the new data.
type TlModifyClosure = extern "C" fn(*mut c_void, *mut c_void, *mut c_void);

#[unsafe(no_mangle)]
pub extern "C" fn tl_mutex_modify(
    id: i64,
    fn_ptr: TlModifyClosure,
    env_ptr: *mut c_void,
) {
    let arc_mutex = {
        let registry = MUTEX_REGISTRY.lock().unwrap();
        if let Some(m) = registry.get(&id) {
            Arc::clone(m)
        } else {
            return;
        }
    };

    // Lock the mutex!
    let mut guard = arc_mutex.lock().unwrap();
    
    // We need a temporary buffer (out_ptr) to receive the closure's updated value
    let size = guard.layout.size();
    if size > 0 {
        unsafe {
            let out_ptr = alloc(guard.layout) as *mut c_void;
            
            // Call the JIT closure: fn(env, current_val_ptr, out_ptr)
            // Safety: Transmuting execution environment across JIT boundary.
            let f: TlModifyClosure = std::mem::transmute(fn_ptr);
            f(env_ptr, guard.data_ptr, out_ptr);
            
            // Free the old data and set new data
            dealloc(guard.data_ptr as *mut u8, guard.layout);
            guard.data_ptr = out_ptr;
        }
    } else {
        unsafe {
            // For Zero-Sized Types, just call it.
            let f: TlModifyClosure = std::mem::transmute(fn_ptr);
            f(env_ptr, std::ptr::null_mut(), std::ptr::null_mut());
        }
    }
}

type TlReadClosure = extern "C" fn(*mut c_void, *mut c_void, *mut c_void);

#[unsafe(no_mangle)]
pub extern "C" fn tl_mutex_read(
    id: i64,
    fn_ptr: TlReadClosure,
    env_ptr: *mut c_void,
    out_ret_ptr: *mut c_void, // User provides a pointer to receive the read result (e.g. alloca)
) {
    let arc_mutex = {
        let registry = MUTEX_REGISTRY.lock().unwrap();
        if let Some(m) = registry.get(&id) {
            Arc::clone(m)
        } else {
            return;
        }
    };

    // Lock the mutex!
    let guard = arc_mutex.lock().unwrap();
    
    unsafe {
        // fn(env_ptr, mutex_value_ptr, user_provided_out_ptr)
        let f: TlReadClosure = std::mem::transmute(fn_ptr);
        f(env_ptr, guard.data_ptr, out_ret_ptr);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_mutex_release(id: i64) {
    let mut registry = MUTEX_REGISTRY.lock().unwrap();
    registry.remove(&id);
}
