// Extended test with more iterations to analyze memory growth rate
fn main() {
    println("Extended Memory leak test with autograd + fixes");
    Param::set_device(Device::Auto);
    
    let mut x = Tensor::randn([10, 10], true);
    
    for i in 0..500 {
        let y = x.softmax(1);
        let z = (y - 0.5).pow(2);
        let loss = z.sumall();
        
        loss.backward();
        
        let g = x.grad();
        x = x - g * 0.1;
        x = x.detach();
        x.enable_grad();
        
        Tensor::clear_grads();
        
        if i % 100 == 0 {
            println("Iter {} - Mem: {} MB", i, System::memory_mb());
        }
    }
    
    println("After 500 iters - Mem: {} MB", System::memory_mb());
}
