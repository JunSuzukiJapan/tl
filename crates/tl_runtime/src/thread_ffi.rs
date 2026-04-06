use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// A wrapper to securely send a dynamically allocated pointer across OS threads.
#[derive(Debug, Clone, Copy)]
pub struct SendablePtr(pub *mut c_void);
unsafe impl Send for SendablePtr {}
unsafe impl Sync for SendablePtr {}

/// A global registry to hold JoinHandles of spawned threads.
static THREAD_REGISTRY: Lazy<Mutex<HashMap<i64, std::thread::JoinHandle<SendablePtr>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_THREAD_ID: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(1));

/// The signature of a JIT-compiled closure function in TL (Returns heap allocated return buffer ptr)
type ThreadFn = extern "C" fn(*mut c_void) -> *mut c_void;

/// @ffi_sig (ThreadFn, *mut c_void) -> i64
/// Spawns a new OS background thread.
/// Takes a JIT-compiled closure function pointer and its environment pointer.
#[unsafe(no_mangle)]
pub extern "C" fn tl_thread_spawn(fn_ptr: ThreadFn, env_ptr: *mut c_void) -> i64 {
    // Generate a new ID
    let id = {
        let mut id_lock = NEXT_THREAD_ID.lock().unwrap();
        let current_id = *id_lock;
        *id_lock += 1;
        current_id
    };

    let fn_ptr_send = fn_ptr as usize;
    let env_ptr_send = env_ptr as usize;

    let handle = std::thread::spawn(move || {
        let f: ThreadFn = unsafe { std::mem::transmute(fn_ptr_send) };
        let env = env_ptr_send as *mut c_void;
        // Invoke the TL function inside the new thread!
        let result_ptr = f(env);
        SendablePtr(result_ptr)
    });

    // Store handle in registry
    let mut registry = THREAD_REGISTRY.lock().unwrap();
    registry.insert(id, handle);

    id
}

/// @ffi_sig (i64) -> void*
/// Joins the thread by its ID and returns the result pointer.
/// Returns null pointer if joining fails (e.g. panic).
#[unsafe(no_mangle)]
pub extern "C" fn tl_thread_join(id: i64) -> *mut c_void {
    let handle_opt = {
        let mut registry = THREAD_REGISTRY.lock().unwrap();
        registry.remove(&id)
    };

    if let Some(handle) = handle_opt {
        match handle.join() {
            Ok(res) => res.0,
            Err(e) => {
                eprintln!("[Thread Error] Panicked during thread execution: {:?}", e);
                std::ptr::null_mut()
            }
        }
    } else {
        // Handle gracefully if invalid ID
        std::ptr::null_mut()
    }
}
