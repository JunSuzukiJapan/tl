// test_memory_leak_debug.tl
// Minimal test for memory leak in while loop tensor reassignment

fn main() {
    println("Testing memory leak...");

    let x = Tensor::randn([10, 10], true);
    let iterations = 7000;
    let i = 0;

    while i < iterations {
        if i % 10 == 0 {
            print("Iteration "); println(i);
        }
        
        // This should free old x before assigning new x
        x = Tensor::randn([10, 10], true);
        
        i = i + 1;
    }
    
    println("Done! If no memory crash, likely fixed.");
}
