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
/// @ffi_sig () -> u64
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
/// @ffi_sig (u64, StringStruct, u64) -> void
pub extern "C" fn tl_hashmap_insert(map_handle: u64, key: *mut crate::StringStruct, value: u64) {
// println!("DEBUG: Insert handle: {}, key: {:p}, val: {}", map_handle, key, value);
    let map = map_handle as *mut TLHashMap;
    if map.is_null() || key.is_null() {
// println!("DEBUG: Map or Key is null");
        return;
    }
    unsafe {
        if (*key).ptr.is_null() { 
             // println!("DEBUG: Key inner ptr is null");
             return; 
        }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy().into_owned();
        // println!("DEBUG: Key string: {}", key_str);
        // println!("DEBUG: Map addr: {:p}", map);
        // data race if multi-threaded, but single threaded test
        // println!("DEBUG: Current Len: {}", (*map).inner.len());
        
        // println!("DEBUG: Reserving capacity...");
        (*map).inner.reserve(1);
        // println!("DEBUG: Capacity reserved. Inserting...");
        
        (*map).inner.insert(key_str, value as *mut c_void);
        // println!("DEBUG: Insert done.");
    }
}

#[unsafe(no_mangle)]
/// @ffi_sig (u64, StringStruct) -> u64
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
/// @ffi_sig (u64, StringStruct) -> u64
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
/// @ffi_sig (u64, StringStruct) -> bool
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
/// @ffi_sig (u64) -> i64
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
/// @ffi_sig (u64) -> void
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
/// @ffi_sig (u64) -> void
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

/// HashMap.keys() -> Vec<String> — 全キーを StringStruct ポインタの Vec として返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_keys(map_handle: u64) -> *mut crate::string_ffi::VecStruct {
    use crate::string_ffi::VecStruct;
    let map = map_handle as *mut TLHashMap;

    unsafe {
        let layout = std::alloc::Layout::new::<VecStruct>();
        let vec_ptr = std::alloc::alloc(layout) as *mut VecStruct;

        if map.is_null() {
            (*vec_ptr).ptr = std::ptr::null_mut();
            (*vec_ptr).cap = 0;
            (*vec_ptr).len = 0;
            return vec_ptr;
        }

        let keys: Vec<&String> = (*map).inner.keys().collect();
        let count = keys.len() as i64;

        let ptr_size = std::mem::size_of::<*mut crate::StringStruct>();
        let array_layout = std::alloc::Layout::from_size_align(
            ptr_size * keys.len().max(1), 8
        ).unwrap();
        let array_ptr = std::alloc::alloc(array_layout) as *mut *mut crate::StringStruct;

        for (i, key) in keys.iter().enumerate() {
            let key_cstr = std::ffi::CString::new(key.as_str()).unwrap_or_default();
            let str_struct = Box::into_raw(Box::new(crate::StringStruct {
                ptr: key_cstr.into_raw(),
                len: key.len() as i64,
            }));
            *array_ptr.add(i) = str_struct;
        }

        (*vec_ptr).ptr = array_ptr as *mut u8;
        (*vec_ptr).cap = count;
        (*vec_ptr).len = count;
        vec_ptr
    }
}

/// HashMap.values() -> Vec<i64> — 全バリュー (u64) を Vec に格納
#[unsafe(no_mangle)]
pub extern "C" fn tl_hashmap_values(map_handle: u64) -> *mut crate::string_ffi::VecStruct {
    use crate::string_ffi::VecStruct;
    let map = map_handle as *mut TLHashMap;

    unsafe {
        let layout = std::alloc::Layout::new::<VecStruct>();
        let vec_ptr = std::alloc::alloc(layout) as *mut VecStruct;

        if map.is_null() {
            (*vec_ptr).ptr = std::ptr::null_mut();
            (*vec_ptr).cap = 0;
            (*vec_ptr).len = 0;
            return vec_ptr;
        }

        let values: Vec<u64> = (*map).inner.values().map(|&v| v as u64).collect();
        let count = values.len() as i64;

        let elem_size = std::mem::size_of::<u64>();
        let array_layout = std::alloc::Layout::from_size_align(
            elem_size * values.len().max(1), 8
        ).unwrap();
        let array_ptr = std::alloc::alloc(array_layout) as *mut u64;

        for (i, &val) in values.iter().enumerate() {
            *array_ptr.add(i) = val;
        }

        (*vec_ptr).ptr = array_ptr as *mut u8;
        (*vec_ptr).cap = count;
        (*vec_ptr).len = count;
        vec_ptr
    }
}
