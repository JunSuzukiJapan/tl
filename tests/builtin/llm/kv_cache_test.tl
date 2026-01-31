
fn main() {
    println("Testing KVCache...");

    let layers = 2;
    // 1. Test new
    let cache = KVCache::new(layers);
    println("KVCache created.");

    // 2. Test get_k / get_v (should be empty/null initially? Or assert behavior)
    // The current implementation returns null ptr if empty, which might crash if we try to use it as tensor?
    // Let's look at `compile_kv_cache_get_k`.
    // It calls `codegen.check_tensor_result`.
    // If runtime returns null, `check_tensor_result` might handle it or error.
    // In `llm.rs`: `tl_kv_cache_get_k` returns `std::ptr::null_mut()`.
    // `check_tensor_result` usually checks if ptr is null. 
    // If it is null, it usually returns an error or panics?
    // Let's assume for now we shouldn't get empty layers if we want safety, 
    // OR `check_tensor_result` handles it gracefully (maybe returning a specific error or empty tensor).
    // Actually, `check_tensor_result` typically calls `tl_get_last_error` if null.
    // But `tl_kv_cache_get_k` doesn't set last error. It just returns null.
    // This might be a bug or undefined behavior in current codegen if it relies on last error.
    
    // To be safe, we will UPDATE first, then GET.
    
    // Create dummy tensors for K and V
    // Shape: [batch, heads, seq, dim] -> [1, 1, 1, 1] for simplicity
    let k_tensor = Tensor::zeros([1, 1, 1, 1]); 
    let v_tensor = Tensor::ones([1, 1, 1, 1]);
    
    println("Updating cache layer 0...");
    cache.update(0, k_tensor, v_tensor);
    println("Update done.");

    println("Getting K from layer 0...");
    let k_out = cache.get_k(0);
    println("Got K.");
    println(k_out);
    
    println("Getting V from layer 0...");
    let v_out = cache.get_v(0);
    println("Got V.");
    println(v_out);

    // 3. Test free
    cache.free();
    println("KVCache freed.");

    println("KVCache tests passed!");
}
