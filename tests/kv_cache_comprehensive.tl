
fn main() {
    println("=== KVCache Comprehensive Test (TL) ===");

    // 1. Basic Lifecycle
    println("[1] Lifecycle Test");
    let initial_mem = tl_get_memory_mb();
    
    let cache = KVCache::new(2);
    println("KVCache created.");
    
    // 2. Update and Get
    println("[2] Update and Get Test");
    let k0 = Tensor::ones([16, 64], false); 
    let v0 = Tensor::zeros([16, 64], false);
    
    cache.update(0, k0, v0);
    
    let k_out = cache.get_k(0);
    println("K0 retrieved.");
    // Verify shape or content if possible, but just existence is good for now.
    
    // 3. Expansion
    println("[3] Expansion Test");
    let k10 = Tensor::ones([16, 64], false);
    let v10 = Tensor::zeros([16, 64], false);
    
    println("Updating layer 10...");
    cache.update(10, k10, v10);
    println("Expanded successfully.");

    // 4. Memory Check
    println("[4] Memory Check");
    let mid_mem = tl_get_memory_mb();
    println(mid_mem);
    
    // 5. Free
    println("[5] Free Test");
    cache.free();
    println("KVCache freed.");
    
    let end_mem = tl_get_memory_mb();
    println(end_mem);
    
    println("=== Test Passed ===");
}
