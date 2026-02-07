// Minimal kv_cache crash reproduction
fn main() {
    println("Step 1: KVCache::new");
    let cache = KVCache::new(1);
    println("Step 2: Tensor::zeros");
    let k = Tensor::zeros([1, 1, 1, 1]);
    let v = Tensor::ones([1, 1, 1, 1]);
    println("Step 3: cache.update");
    cache.update(0, k, v);
    println("Step 4: cache.get_k");
    let k_out = cache.get_k(0);
    println("Step 5: after get_k");
    print(k_out);
    println("\nStep 6: cache.free");
    cache.free();
    println("DONE");
}
