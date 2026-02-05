// Minimal reproduction for memory leak with autograd - WITH clear_grads fix
// Run: cargo run --release -- --device=cpu tests/fixtures/debug/mem_leak_autograd_fixed.tl

fn main() {
    println("Memory leak test with autograd + clear_grads - Expect stable memory usage");
    Param::set_device(Device::Auto);
    
    let mut x = Tensor::randn([10, 10], true);  // Requires grad
    
    for i in 0..100 {
        // Operations that create gradient graph
        let y = x.softmax(1);
        let z = (y - 0.5).pow(2);
        let loss = z.sumall();
        
        loss.backward();
        
        let g = x.grad();
        x = x - g * 0.1;
        x = x.detach();
        x.enable_grad();
        
        // FIX: Clear gradient graph to prevent memory leak
        Tensor::clear_grads();
        
        if i % 20 == 0 {
            println("Iter {} - Mem: {} MB", i, System::memory_mb());
        }
    }
    
    println("After loop - Mem: {} MB", System::memory_mb());
}
