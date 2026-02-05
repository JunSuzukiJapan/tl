// Test with pool stats to analyze where the leak occurs
fn main() {
    println("Memory leak analysis with pool/refcount stats");
    Param::set_device(Device::Auto);
    
    let mut x = Tensor::randn([10, 10], true);
    
    for i in 0..200 {
        let y = x.softmax(1);
        let z = (y - 0.5).pow(2);
        let loss = z.sumall();
        
        loss.backward();
        
        let g = x.grad();
        x = x - g * 0.1;
        x = x.detach();
        x.enable_grad();
        
        Tensor::clear_grads();
        
        if i % 50 == 0 {
            let pool = System::get_pool_count();
            let refs = System::get_refcount_count();
            let depth = System::get_scope_depth();
            println("Iter {} - Mem: {} MB | Pool: {} | Refs: {} | Depth: {}", 
                i, System::memory_mb(), pool, refs, depth);
        }
    }
    
    println("After 200 iters - Mem: {} MB", System::memory_mb());
}
