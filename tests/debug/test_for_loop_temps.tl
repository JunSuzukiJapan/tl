// test_for_loop_temps.tl
// Test if for-loop intermediate tensors are freed properly

fn main() {
    println("Testing for-loop intermediate tensor cleanup...");

    let x = Tensor::randn([10, 10], true);

    for i in 0..2000 {
        if i % 100 == 0 {
            print("Iteration "); println(i);
        }
        
        // Create many intermediate tensors each iteration
        let a = x.softmax(1);
        let b = a.sum(0);
        let c = (b - 1.0).pow(2);
        let d = c.sum();
        
        // These temps should be freed at end of iteration
    }
    
    println("Done!");
}
