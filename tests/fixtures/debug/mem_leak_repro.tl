// Minimal reproduction for memory leak
// Run: cargo run --release -- tests/fixtures/debug/mem_leak_repro.tl

fn main() {
    println("Memory leak test - Expect stable memory usage");
    
    for i in 0..100 {
        // Create temporary tensors inside loop
        let a = Tensor::randn([10, 10], false);
        let b = Tensor::randn([10, 10], false);
        let c = a + b;  // Temporary tensor from binary op
        let d = c * 2.0;  // Another temporary
        let _sum = d.sumall().item();  // Force evaluation
        
        if i % 20 == 0 {
            println("Iter {} - Mem: {} MB", i, System::memory_mb());
        }
    }
    
    println("After loop - Mem: {} MB", System::memory_mb());
}
