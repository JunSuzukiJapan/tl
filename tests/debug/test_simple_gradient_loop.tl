// test_simple_gradient_loop.tl
// Minimal test for gradient descent in a for loop

fn main() {
    println("Testing simple gradient descent loop...");

    // Create a simple optimizable tensor
    let x = Tensor::randn([3, 3], true);
    
    // Target: all zeros
    let target = Tensor::zeros([3, 3]);
    
    let lr = 0.1;

    for epoch in 0..10 {
        // Forward: compute MSE loss
        let diff = x - target;
        let loss = (diff * diff).sum();
        
        print("Epoch "); print(epoch); print(" - Loss: "); println(loss.item());
        
        // Backward
        loss.backward();
        
        // Get gradient
        let g = x.grad();
        print("  Grad sum: "); println(g.sum().item());
        
        // Update (use assignment, not let)
        x = x - g * lr;
        x = x.detach();
        x.enable_grad();
    }
    
    println("Final x:");
    println(x);
    println("Done!");
}
