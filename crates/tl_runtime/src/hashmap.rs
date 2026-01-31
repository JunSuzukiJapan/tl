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
pub extern "C" fn tl_hashmap_new() -> *mut TLHashMap {
    let map = TLHashMap {
        inner: HashMap::new(),
    };
    let ptr = Box::into_raw(Box::new(map));
    
    // Log allocation for leak detection
    // Size is unknown/dynamic, pass 0 represents "managed object"
    // crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);

    // Register with MemoryManager for automatic cleanup
    crate::memory_manager::tl_mem_register_custom(ptr as *mut c_void, hashmap_dtor_shim);
    
    ptr
}

/// Shim to match extern "C" fn(*mut c_void) signature for generic destructor
extern "C" fn hashmap_dtor_shim(ptr: *mut c_void) {
    tl_hashmap_free(ptr as *mut TLHashMap);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_insert(map: *mut TLHashMap, key: *mut crate::StringStruct, value: *mut c_void) {
    if map.is_null() || key.is_null() {
        return;
    }
    unsafe {
        if (*key).ptr.is_null() { return; }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy().into_owned();
        (*map).inner.insert(key_str, value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_get(map: *mut TLHashMap, key: *mut crate::StringStruct) -> *mut c_void {
    if map.is_null() || key.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        if (*key).ptr.is_null() { return std::ptr::null_mut(); }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        match (*map).inner.get(key_str.as_ref()) {
            Some(&val) => val,
            None => std::ptr::null_mut(),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_remove(map: *mut TLHashMap, key: *mut crate::StringStruct) -> *mut c_void {
    if map.is_null() || key.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        if (*key).ptr.is_null() { return std::ptr::null_mut(); }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        match (*map).inner.remove(key_str.as_ref()) {
            Some(val) => val,
            None => std::ptr::null_mut(),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_contains_key(map: *mut TLHashMap, key: *mut crate::StringStruct) -> bool {
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
pub extern "C" fn tl_hashmap_len(map: *mut TLHashMap) -> i64 {
    if map.is_null() {
        return 0;
    }
    unsafe {
        (*map).inner.len() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_clear(map: *mut TLHashMap) {
    if map.is_null() {
        return;
    }
    unsafe {
        (*map).inner.clear();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_free(map: *mut TLHashMap) {
    if map.is_null() {
        return;
    }
    
    // Log free before dropping
    // Log free before dropping
    // crate::tl_log_free(map as *const c_void, std::ptr::null(), 0);
    
    unsafe {
        let _ = Box::from_raw(map);
    }
}
