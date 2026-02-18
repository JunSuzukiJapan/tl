fn main() {
    let mut x = Tensor::randn([4, 4], true);
    println("=== Start ===");
    
    for i in 0..3 {
        println("--- Iter {} start ---", i);
        println("Before softmax: x = {}", x);
        
        let y = x.softmax(1);
        let loss = y.sumall();
        println("Loss: {}", loss.item());
        
        loss.backward();
        let g = x.grad();
        println("Grad: {}", g);
        
        let step = g * 0.1;
        println("Step: {}", step);
        
        x = x - step;
        println("After sub: x = {}", x);
        
        x = x.detach();
        println("After detach: x = {}", x);
        
        x.enable_grad();
        println("After enable_grad: x = {}", x);
        
        println("Iter {} done (Mem: {} MB)", i, System::memory_bytes() / 1024 / 1024);
    }
    println("Done!");
}
