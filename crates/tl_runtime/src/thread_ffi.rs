use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// A global registry to hold JoinHandles of spawned threads.
static THREAD_REGISTRY: Lazy<Mutex<HashMap<i64, std::thread::JoinHandle<i64>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_THREAD_ID: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(1));

/// The signature of a JIT-compiled closure function in TL
type ThreadFn = extern "C" fn(*mut c_void) -> i64;

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

    // Note: The JIT engine function pointers are essentially static.
    // The environment pointer (env_ptr) must trace back to safe memory (ARC/Arena).
    // The compiler MUST ensure the closure is Send-safe.
    
    // Safety: we transmute the lifetime of the extern "C" function to 'static 
    // because JIT code usually lives until the ExecutionEngine is destroyed.
    // Also env_ptr needs to be raw-sent to the new thread.
    let fn_ptr_send = fn_ptr as usize;
    let env_ptr_send = env_ptr as usize;

    let handle = std::thread::spawn(move || {
        let f: ThreadFn = unsafe { std::mem::transmute(fn_ptr_send) };
        let env = env_ptr_send as *mut c_void;
        // Invoke the TL function inside the new thread!
        f(env)
    });

    // Store handle in registry
    let mut registry = THREAD_REGISTRY.lock().unwrap();
    registry.insert(id, handle);

    id
}

/// @ffi_sig (i64) -> i64
/// Joins the thread by its ID and returns the result (i64).
/// Returns -1 or appropriate error fallback if joining fails.
#[unsafe(no_mangle)]
pub extern "C" fn tl_thread_join(id: i64) -> i64 {
    let handle_opt = {
        let mut registry = THREAD_REGISTRY.lock().unwrap();
        registry.remove(&id)
    };

    if let Some(handle) = handle_opt {
        match handle.join() {
            Ok(res) => res,
            Err(e) => {
                eprintln!("[Thread Error] Panicked during thread execution: {:?}", e);
                -1
            }
        }
    } else {
        // Handle gracefully if invalid ID
        -1
    }
}
