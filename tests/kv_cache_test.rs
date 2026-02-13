use serial_test::serial;
use tl_runtime::system::*;
use tl_runtime::*;

// Helper to check if a tensor is not null
fn assert_tensor_valid(t: *mut OpaqueTensor) {
    assert!(!t.is_null(), "Tensor pointer is null");
}

#[test]
#[serial]
fn test_kv_cache_lifecycle() {
    // 1. New
    let raw_ptr = tl_kv_cache_new(2);
    assert_ne!(raw_ptr, 0, "KVCache pointer should not be null/0");

    // 2. Free
    tl_kv_cache_free(raw_ptr);
}

#[test]
#[serial]
fn test_kv_cache_update_get() {
    let raw_ptr = tl_kv_cache_new(2);
    
    // Create dummy tensors
    let data = [1.0, 2.0, 3.0, 4.0];
    let shape = [2, 2];
    
    // We need valid OpaqueTensor pointers. 
    // Since this is an integration test, we can use tl_metal to create them if available.
    // Or we can rely on the fact that we are linking against tl_runtime which uses Metal.
    // tests/runtime_test.rs uses tl_metal::ffi_ops::tl_metal_new.
    
    let k_ptr = tl_metal::ffi_ops::tl_metal_new(data.as_ptr(), 2, shape.as_ptr());
    let v_ptr = tl_metal::ffi_ops::tl_metal_new(data.as_ptr(), 2, shape.as_ptr());
    
    assert_tensor_valid(k_ptr);
    assert_tensor_valid(v_ptr);

    // Update layer 0
    tl_kv_cache_update(raw_ptr, 0, k_ptr, v_ptr);

    // Get layer 0
    let k_out = tl_kv_cache_get_k(raw_ptr, 0);
    let v_out = tl_kv_cache_get_v(raw_ptr, 0);
    
    assert_tensor_valid(k_out);
    assert_tensor_valid(v_out);
    
    // Verify content (optional, but good)
    // We can use tl_metal_get_f32_md
    let val = tl_metal::ffi_ops::tl_metal_get_f32_md(k_out, 0, 0);
    assert_eq!(val, 1.0);

    // Free everything
    // Note: k_ptr and v_ptr were cloned inside cache. 
    // We should free our local handles.
    tl_metal::ffi_ops::tl_metal_free(k_ptr);
    tl_metal::ffi_ops::tl_metal_free(v_ptr);
    
    tl_kv_cache_free(raw_ptr);
    // k_out and v_out are pointers to the cache's internal tensors (or clones).
    // Wait, get_k/get_v returns raw pointer from cache.
    // In current system.rs implementation:
    // cache.layers[idx].0.unwrap_or(null)
    // It returns the pointer stored in the cache.
    // Since cache owns it (via Arc/Box logic in OpaqueTensor?), 
    // actually OpaqueTensor is MetalTensor.
    // system.rs update does:
    // let k_clone = ... tl_device_tensor_clone(k) ...
    // So cache owns a clone.
    // get_k returns that cloned pointer.
    // When we free cache, it should release its clones.
    // We don't need to free k_out/v_out explicitly if they are just borrowed pointers 
    // from the cache's perspective, OR if they are new Arc handles.
    // system.rs get_k returns *mut OpaqueTensor.
    // It returns the raw pointer stored in the Vec.
    // So it behaves like a borrow. 
    // Accessing k_out after tl_kv_cache_free(raw_ptr) would be UAF if valid.
    // But since OpaqueKVCache Drop calls tl_tensor_release_safe (RC-1),
    // and that might drop the tensor if RC=0.
    // So yes, after cache free, k_out is invalid.
}

#[test]
#[serial]
fn test_kv_cache_expansion() {
    let raw_ptr = tl_kv_cache_new(1);
    
    // Update layer 5
    // Pass nulls just to check expansion logic without tensor overhead
    tl_kv_cache_update(raw_ptr, 5, std::ptr::null_mut(), std::ptr::null_mut());
    
    // Get layer 5 (should be null but exists in vector)
    let k_out = tl_kv_cache_get_k(raw_ptr, 5);
    assert!(k_out.is_null());
    
    // Check internal vector size indirectly? 
    // We can't access struct fields from here.
    // But no crash means success.
    
    tl_kv_cache_free(raw_ptr);
}
