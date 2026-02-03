use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_void;

/// Wrapper struct for HashMap to be used in TL.
/// Keys are Strings, Values are opaque pointers (void*).
/// This allows storing Tensors, other Structs, or boxed primitives.
pub struct TLHashMap {
    pub inner: HashMap<String, *mut c_void>,
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_new() -> u64 {
    let map = TLHashMap {
        inner: HashMap::new(),
    };
    let ptr = Box::into_raw(Box::new(map));
    
    // Log allocation for leak detection
    // crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);

    // Register with MemoryManager for automatic cleanup
    crate::memory_manager::tl_mem_register_custom(ptr as *mut c_void, hashmap_dtor_shim);
    
    ptr as u64
}

extern "C" fn hashmap_dtor_shim(ptr: *mut c_void) {
    tl_hashmap_free(ptr as u64);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_insert(map_handle: u64, key: *mut crate::StringStruct, value: u64) {
    println!("DEBUG: Insert handle: {}, key: {:p}, val: {}", map_handle, key, value);
    let map = map_handle as *mut TLHashMap;
    if map.is_null() || key.is_null() {
        println!("DEBUG: Map or Key is null");
        return;
    }
    unsafe {
        if (*key).ptr.is_null() { 
             println!("DEBUG: Key inner ptr is null");
             return; 
        }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy().into_owned();
        println!("DEBUG: Key string: {}", key_str);
        println!("DEBUG: Map addr: {:p}", map);
        // data race if multi-threaded, but single threaded test
        println!("DEBUG: Current Len: {}", (*map).inner.len());
        
        println!("DEBUG: Reserving capacity...");
        (*map).inner.reserve(1);
        println!("DEBUG: Capacity reserved. Inserting...");
        
        (*map).inner.insert(key_str, value as *mut c_void);
        println!("DEBUG: Insert done.");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_get(map_handle: u64, key: *mut crate::StringStruct) -> u64 {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() || key.is_null() {
        return 0;
    }
    
    unsafe {
        if (*key).ptr.is_null() { return 0; }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        match (*map).inner.get(key_str.as_ref()) {
            Some(&val) => val as u64,
            None => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_remove(map_handle: u64, key: *mut crate::StringStruct) -> u64 {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() || key.is_null() {
        return 0;
    }
    
    unsafe {
        if (*key).ptr.is_null() { return 0; }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        match (*map).inner.remove(key_str.as_ref()) {
            Some(val) => val as u64,
            None => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_contains_key(map_handle: u64, key: *mut crate::StringStruct) -> bool {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() || key.is_null() {
        return false;
    }
    
    unsafe {
        if (*key).ptr.is_null() { return false; }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        (*map).inner.contains_key(key_str.as_ref())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_len(map_handle: u64) -> i64 {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() {
        return 0;
    }
    unsafe {
        (*map).inner.len() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_clear(map_handle: u64) {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() {
        return;
    }
    unsafe {
        (*map).inner.clear();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_free(map_handle: u64) {
    let map = map_handle as *mut TLHashMap;
    if map.is_null() {
        return;
    }
    
    // Log free before dropping
    // crate::tl_log_free(map as *const c_void, std::ptr::null(), 0);
    
    unsafe {
        let _ = Box::from_raw(map);
    }
}
